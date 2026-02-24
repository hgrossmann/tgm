import torch, torch.nn as nn
import math

class MarkovSuperpositionModel(torch.nn.Module):
    """
    Superposition wrapper that behaves like your other models.
    Combine outputs as (1 - alpha) * A + alpha * B.
    """
    def __init__(self,
                 model_sde: torch.nn.Module,
                 model_jump: torch.nn.Module,
                 alpha: float, jump_api: str):
        super().__init__()
        self.model_sde = model_sde
        self.model_jump = model_jump
        self.alpha = float(alpha)
        self.sqrt_alpha = math.sqrt(self.alpha)
        self.sigma = model_sde.sigma
        self.rho = model_sde.rho
        self.memory_length = model_sde.memory_length
        self.device = model_sde.device
        self.data_dim = model_jump.data_dim
        self.jump_api = jump_api

    @torch.no_grad()
    def sample_unif(self, x0, no_bridges, t_start, t_end, stepsize, no_samples=100):
        """
        Vectorized Euler–Maruyama with 'no_bridges' equal-length bridges,
        starting in x0 at 't_start' with no memory and ending at 't_end'.
        
        Memory at each step uses the last 'memory_length' bridge endpoints
        up to and including the startpoint of the current bridge.
        
        If t_end - t_start / no_bridges is no multiple of stepsize,
        the closest feasible stepsize is used.
        
        x0 : [no_samples, state_dim] or [state_dim]
        All other scalars can be Python floats or 0-d torch tensors.
        
        """
        if x0.ndim == 1: x0 = x0.unsqueeze(0).expand(no_samples, -1)
        
        dtype  = x0.dtype if x0.is_floating_point() else torch.float32
        
        t_start  = torch.as_tensor(t_start,  device=self.device, dtype=dtype)
        t_end    = torch.as_tensor(t_end,    device=self.device, dtype=dtype)
        stepsize = torch.as_tensor(stepsize, device=self.device, dtype=dtype)
        sigma    = torch.as_tensor(self.sigma, device=self.device, dtype=dtype)
        
        bridge_length = (t_end - t_start) / float(no_bridges)
        disc_steps = int(torch.round(bridge_length / stepsize).clamp(min=1).item())
        stepsize = bridge_length / disc_steps
        t_bridges = torch.linspace(t_start, t_end, steps=no_bridges+1, device=self.device, dtype=dtype)
        
        idcs_bridge_endpoints = torch.arange(no_bridges + 1, device=self.device, dtype=torch.long) * disc_steps
                
        traj = torch.zeros((no_samples, disc_steps * no_bridges + 1, x0.shape[-1]), 
                           device=self.device, dtype=dtype)
        traj[:,0,:] = x0
        
        x_mem = x0.unsqueeze(1).expand(-1, self.memory_length, -1).clone().to(self.device)  # [N,M,D]
        t_mem = t_start.view(1, 1, 1).expand(no_samples, self.memory_length, 1).clone()     # [N,M,1]
        
        for k in range(no_bridges):
            
            if k > 0: # update memory window
                x_mem = x_mem.roll(shifts=-1, dims=1)
                t_mem = t_mem.roll(shifts=-1, dims=1)
                x_mem[:, -1, :] = traj[:, k * disc_steps, :]
                t_mem[:, -1, :] = t_bridges[k]
                
            t2 = t_bridges[k+1].expand(no_samples, 1)
            
            for i in range(disc_steps):
                idx = k * disc_steps + i
                
                x = traj[:, idx, :]
                t = (t_bridges[k] + i * stepsize).expand(no_samples, 1)
                
                drift = self.model_sde.forward(x, t, x_mem, t_mem, t2)
                x_new_drift = x + self.alpha * stepsize * drift + torch.sqrt(stepsize) * self.sqrt_alpha * torch.sqrt(sigma) * torch.randn_like(x)
                
                if self.jump_api == "full_cov":
                    lambd, mu, L = self.model_jump.forward(x, t, x_mem, t_mem, t2)
                    
                    eps = torch.randn(no_samples, self.data_dim, device=self.device) # [B, D] ~ N(0,I)
                    x_new_jump = mu + torch.bmm(eps.unsqueeze(1), L).squeeze(1)   # [B, D] ~ N(mu, LL^T)
                    rt = torch.exp(- (1-self.alpha) * lambd * stepsize)
                    
                elif self.jump_api == "jump":
                    out = self.model_jump.forward(x, t, x_mem, t_mem, t2)     
                    lambda_t = torch.exp(out[:,0:1])  # jump intensity
                    lambda_t.clamp_(0,1000)
                    mu_j = out[:,1:-1]
                    sigma_j = torch.exp(out[:,-1:])
                    sigma_j.clamp_(0,1000)
                    
                    x_new_jump = mu_j + sigma_j * torch.randn(no_samples, self.data_dim, device = self.device)
                    rt = torch.exp(- (1-self.alpha) * lambda_t * stepsize)

                elif self.jump_api == "uncoupled":
                    out = self.model_jump.forward(x, t, x_mem, t_mem, t2)         
                    lambda_t    = torch.exp(out[:,:self.data_dim])  # jump intensity
                    mu_j        = out[:,self.data_dim:2*self.data_dim]
                    sigma_j     = torch.exp(out[:,2*self.data_dim:])
                    
                    x_new_jump = mu_j + sigma_j * torch.randn(no_samples, self.data_dim, device = self.device)
                    rt = torch.exp(- (1-self.alpha) * lambda_t * stepsize)

                else:
                    raise ValueError("Unexpected jump api.")
                    
                m = torch.bernoulli(1 - rt)
                x_new = (1 - m) * x_new_drift + m * x_new_jump
                
                traj[:, idx + 1, :] = x_new
        
        marginals = traj[:,idcs_bridge_endpoints,:]
        times = torch.linspace(t_start, t_end, steps=no_bridges*disc_steps+1, device=self.device, dtype=dtype)
        times = times.unsqueeze(0).expand(no_samples, -1)
        
        return traj, times, marginals