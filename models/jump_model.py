import torch, torch.nn as nn
import numpy as np
import sys
from typing import Optional, Dict, Any
from tgm.utils.network import create_jump_network
from tgm.utils.memory import get_memory
from omegaconf import DictConfig

class JumpModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.sigma = float(model_cfg.sigma)
        self.rho = float(model_cfg.rho)
        self.memory_length = int(model_cfg.memory_length)
        self.device = model_cfg.device
        self.jumpnet = create_jump_network(model_cfg).to(self.device)
        self.time_sampling = model_cfg.time_sampling
        self.data_dim = model_cfg.data_dim
        if self.data_dim == 2:
            self.quadrature = GL2D_JumpMoments(device=self.device)
        self.loss_function = model_cfg.loss_function

    def forward(self, x, t, x_mem, t_mem, t2):
        """
        x of shape [batchsize, state_dim]
        t, t2 of shape [batchsize, 1]
        x_mem of shape [batchsize, memory_length, state_dim]
        t_mem of shape [batchsize, memory_length, 1]
        """    
        batchsize = x.shape[0]
        single_vals = torch.cat([x, t, t2], dim=-1)
        memory_flat = torch.cat([x_mem, t_mem], dim=-1).reshape(batchsize, -1)
        # single_vals = torch.cat([t2, t, x], dim=-1)
        # memory_flat = torch.cat([x_mem.reshape(batchsize, -1), t_mem.reshape(batchsize, -1)], dim=-1)
        inpu = torch.cat([single_vals, memory_flat], dim=-1)
        return self.jumpnet(inpu)
    
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
                
                out = self.forward(x, t, x_mem, t_mem, t2)      
                if not out.isfinite().all().item():
                    print("out no finite")
                
                lambda_t = torch.exp(out[:,0:1])  # jump intensity
                if not lambda_t.isfinite().all().item():
                    print("lambda_t not finite")
                lambda_t.clamp_(0,1000)
                mu_j = out[:,1:-1]
                if not mu_j.isfinite().all().item():
                    print("mu_j not finite")
                sigma_j = torch.exp(out[:,-1:])
                if not sigma_j.isfinite().all().item():
                    print("sigma_j not finite")
                sigma_j.clamp_(0,1000)
                
                z = mu_j + sigma_j * torch.randn(no_samples, self.data_dim, device = self.device)
                rt = torch.exp(- lambda_t * stepsize)
                if not rt.isfinite().all().item():
                    print("rt not finite")
                m = torch.bernoulli(1 - rt)
                
                x_new = (1 - m) * x + m * z
                # x_new = torch.clamp(x_new, 0, 1000) # TODO data dependent choice, adapt!
                
                traj[:, idx + 1, :] = x_new
        
        marginals = traj[:,idcs_bridge_endpoints,:]
        times = torch.linspace(t_start, t_end, steps=no_bridges*disc_steps+1, device=self.device, dtype=dtype)
        times = times.unsqueeze(0).expand(no_samples, -1)
        
        return traj, times, marginals
    
    @torch.no_grad()
    def sample(self, data_given, bridge_length, t_start, t_end, stepsize, no_samples=100):
        """
        Vectorized Euler-Maruyama, using bridges of length bridge_length
        to extrapolate time series data_given with most recent data at t_start
        until t_end in approximate size stepsize.
        
        If bridge_length is no multiple of stepsize,
        the closest feasible stepsize is used.
        
        data_given : [no_samples, n, state_dim]
        t_start, t_end : [no_samples, 1]
        """
        
        raise NotImplementedError()
                    
    def loss(self, batch):
        """
        x, x1, x2 of shape [batchsize, state_dim]
        t, t1, t2 of shape [batchsize, 1]
        x_mem of shape [batchsize, memory_length, state_dim]
        t_mem of shape [batchsize, memory_length, 1]
        """
        data = batch["x"] # [batchsize, trajectory length, state_dim]
        times = batch["t"] # [batchsize, trajectory length]
        mask = None # batch["mask"] # TODO add nonuniform lengths later
 
        batch_size = data.shape[0]
        
        t, idx_prev = self._draw_t(data, times, mask)
        
        # draw x and calculate conditional velocities
        x1 = data[torch.arange(batch_size, device=self.device), idx_prev]
        x2 = data[torch.arange(batch_size, device=self.device), idx_prev + 1]
        t1 = times[torch.arange(batch_size, device=self.device), idx_prev].unsqueeze(1) 
        t2 = times[torch.arange(batch_size, device=self.device), idx_prev + 1].unsqueeze(1)  
        
        # debugging
        # x1 = torch.tensor([[1]], dtype=torch.float32, device= self.device)
        # x2 = torch.tensor([[2.5]], dtype=torch.float32, device= self.device)
        # t1 = torch.tensor([[18]], dtype=torch.float32, device= self.device)
        # t2 = torch.tensor([[27]], dtype=torch.float32, device= self.device)
        # t = torch.tensor([[23.4]], dtype=torch.float32, device= self.device)
        
        taut = self.sigma*(t-t1)*(t2-t)/(t2-t1)**2+self.rho
        mt = (t2-t)/(t2-t1) * x1 + (t-t1)/(t2-t1) * x2
        x =  mt + torch.sqrt(taut) * torch.randn_like(x1)
        
        # predict jump measure parameters
        x_mem, t_mem = get_memory(data, times, idx_prev, self.memory_length)
        out = self.forward(x, t, x_mem, t_mem, t2)
        lambda_t_pred = torch.exp(out[:,0:1]) # [B, 1]
        mean_pred = out[:,1:-1] # [B, D]
        sigma_pred = torch.exp(out[:,-1:]) # [B, 1]
        
        eps = 1e-10 # if self.rho == 0 else 0 # for numerical stability
        # implementation close to paper numerically unstable
        # prefactor = self.sigma * (t1+t2-2*t)/(2*(taut+eps)*(t2-t1)**2) 
        # a_over_sqrt_tau = (t2-t1) * (x2 - x1) / (self.sigma * (t1+t2-2*t)) # explodes!!
        # xi_t = prefactor * (torch.sum((x-mt)**2, dim=1, keepdim=True) / (taut+eps) + 2 * torch.sum(a_over_sqrt_tau * (x-mt), dim=1, keepdim=True) - self.data_dim)
        # lambda_t_cond = nn.functional.relu(-xi_t) # [B, 1]
        # instead to the following
        term = self.sigma*(t2+t1-2*t)/(2*(t2-t1)**2)  -.5*self.sigma*(t2+t1-2*t)*torch.sum((x-mt)**2, dim=1, keepdim=True)/((t2-t1)**2*(taut+eps))-torch.sum((x-mt)*(x2-x1), dim=1, keepdim=True)/(t2-t1)
        
        lambda_t_cond = nn.functional.relu(term) / taut + eps
        
        if not lambda_t_cond.isfinite().all().item():
            print("lambda_t_cond not finite")
               
        mean_cond, trace_cov_cond = self._calc_conditional_mean_cov(x1, x2, t, t1, t2)
            
        if self.loss_function == 'ikl':
            loss_func = loss_ikl(data_dim=self.data_dim)
        elif self.loss_function == 'naive':
            loss_func = loss_naive(data_dim=self.data_dim)
        else:
            raise ValueError(f"Invalid loss '{self.loss_function}'. Expected 'ikl' or 'naive'.")
        
        # debugging
        # lambda_t_pred, mean_pred, sigma_pred = torch.tensor([1], device= self.device), torch.tensor([2], device= self.device), torch.tensor([3], device= self.device)
        x_np = x.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()
        t1_np = t1.detach().cpu().numpy()
        t2_np = t2.detach().cpu().numpy()
        var_np = trace_cov_cond.detach().cpu().numpy()
        mean_np = mean_cond.detach().cpu().numpy()
        lambd_np = lambda_t_cond.detach().cpu().numpy()
        sig_pred_np = sigma_pred.detach().cpu().numpy()
        mean_pred_np = mean_pred.detach().cpu().numpy()
        lambd_pred_np = lambda_t_pred.detach().cpu().numpy()

            
        loss = loss_func(lambda_t_cond, lambda_t_pred, mean_cond, mean_pred, trace_cov_cond, sigma_pred)

        return loss
    
    def _calc_conditional_mean_cov(self, x1, x2, t, t1, t2):
        """
        x, x1, x2 : [batchsize, data_dim]
        t, t1, t2 : [batchsize, 1]
        """
        
        if self.data_dim == 1:
             
            # Compute z_p = z+ and z_n = z-
            taut = self.sigma*(t-t1)*(t2-t)/(t2-t1)**2 + self.rho
            a = (t2-t1) * (x2 - x1) * torch.sqrt(taut) / (self.sigma * sign(t1+t2-2*t) * (t1+t2-2*t).abs().clamp(min=1e-8))

            z_p = - a + torch.sqrt(a**2 + 1)
            z_n = - a - torch.sqrt(a**2 + 1)
            
            # Compute truncated moments times 1 over sqrt(2*pi)
            sq_two_pi = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.float32, device=self.device))
            I_0 = 0.5 * (torch.erf(z_p / 2**0.5) - torch.erf(z_n / 2**0.5))
            I_1 = 1.0 / (sq_two_pi) * (torch.exp(-z_n**2 / 2) - torch.exp(-z_p**2 / 2))
            I_2 = I_0 - 1.0 / (sq_two_pi) * (z_p * torch.exp(-z_p**2 / 2) - z_n * torch.exp(-z_n**2 / 2))
            I_3 = 2*I_1 - 1.0 / (sq_two_pi) * (z_p**2 * torch.exp(-z_p**2 / 2) - z_n**2 * torch.exp(-z_n**2 / 2))
            I_4 = 3*I_2 - 1.0 / (sq_two_pi) * (z_p**3 * torch.exp(-z_p**2 / 2) - z_n**3 * torch.exp(-z_n**2 / 2))
    
            # Apply conditional transformations
            mid_time = (t2+t1)/2
            I_0 = torch.where(t > mid_time, I_0, 1 - I_0)
            I_1 = torch.where(t > mid_time, I_1, - I_1)
            I_2 = torch.where(t > mid_time, I_2, 1 - I_2)
            I_3 = torch.where(t > mid_time, I_3, - I_3)
            I_4 = torch.where(t > mid_time, I_4, 3 - I_4)
            
            # if torch.isnan(I_0).any() or torch.isnan(I_1).any() or torch.isnan(I_2).any() or torch.isnan(I_3).any() or torch.isnan(I_4).any():
            #    print(torch.isnan(I_0).any())
            #    print(torch.isnan(I_1).any())
            #    print(torch.isnan(I_2).any())
            #    print(torch.isnan(I_3).any())
            #    print(torch.isnan(I_4).any())
            #    sys.exit()
            
            # Compute conditional mean and cov, expand denominator and enumerator for numerical stability
            mt = (t2-t)/(t2-t1) * x1 + (t-t1)/(t2-t1) * x2
            
            expand_by = self.sigma * ((t2+t1)-2*t)/(2*(t2-t1)) # note that 2a_t * expand_by = sqrt(tau)*(x2-x1)
            denom = expand_by * (I_2 - I_0) + torch.sqrt(taut) * (x2-x1) * I_1

            # Debugging
            # tau_np = taut.detach().cpu().numpy()
            # a_np = a.detach().cpu().numpy()
            # z_p_np = z_p.detach().cpu().numpy()
            # z_n_np = z_n.detach().cpu().numpy()
            # I_0_np = I_0.detach().cpu().numpy()
            # I_1_np = I_1.detach().cpu().numpy()
            # I_2_np = I_2.detach().cpu().numpy()
            # expand_np = expand_by.detach().cpu().numpy()
            # denom_np = denom.detach().cpu().numpy()
            
            if (denom < 1e-8).any().item(): 
                print("\33[31m OMG denominator<1e-8", "\33[0m",end="")
                print(denom.min().item())
            
            mean_cond = mt + torch.sqrt(taut) * (expand_by * (I_3-I_1) + torch.sqrt(taut) * (x2-x1) * I_2) / denom
            
            var_cond = taut * ((expand_by * (I_4-I_2) + torch.sqrt(taut) * (x2-x1) * I_3) / denom) - (mt - mean_cond)**2
            var_cond = var_cond.clamp(0,1000)
            
            return mean_cond, var_cond
        
        elif self.data_dim == 2:
            
            tau_t = self.sigma*(t-t1)*(t2-t)/(t2-t1)**2 + self.rho
            a_t = (t2-t1) * (x2 - x1) * torch.sqrt(tau_t) / (self.sigma * sign(t1+t2-2*t) * (t1+t2-2*t).abs().clamp(min=1e-8))
            m_t = (t2-t)/(t2-t1) * x1 + (t-t1)/(t2-t1) * x2
            
            tau_t = tau_t.squeeze(1).to(torch.float64)
            a_t = a_t.to(torch.float64)
            m_t = m_t.to(torch.float64)
            
            mean, tr_cov = self.quadrature(a_t, m_t, tau_t)
            
            if (tr_cov < 0).any().item():
                print("Tr(Cov) is negative!")
            
            return mean, tr_cov
        
        else:
            raise NotImplementedError()
    
    def _draw_t(self, data, times, mask):
        batch_size = data.shape[0]
        nb_timepoints = times.shape[1]
        
        if self.time_sampling == "uniform_time":
            # t uniformly distributed in [t_min, t_max]
            tmin = times.min(dim=1).values.unsqueeze(1)          # [B,1]
            tmax = times.max(dim=1).values.unsqueeze(1)          # [B,1]
            t = torch.rand(batch_size, 1, device=self.device) * (tmax - tmin) + tmin
            indices = torch.arange(nb_timepoints, device=self.device).expand(batch_size, nb_timepoints)
            earlier_observations = (times <= t)
            idx_prev_observation = (earlier_observations * indices).max(dim=1).values # [batchsize]
            idx_prev_observation = torch.clamp(idx_prev_observation, 0, nb_timepoints - 2)
        else:
            # idx_prev_observation uniformly distributed (not weighted by interval length)
            random_idx = torch.rand(batch_size, device=self.device) * (nb_timepoints-1)
            idx_prev_observation= torch.floor(random_idx).long()  
            idx_prev_observation = torch.clamp(idx_prev_observation, 0, nb_timepoints - 2)
            t1 = times[torch.arange(batch_size, device=self.device), idx_prev_observation] 
            t2 = times[torch.arange(batch_size, device=self.device), idx_prev_observation + 1] 
            frac = (random_idx % 1)
            t = (t1 + frac * (t2-t1)).unsqueeze(1)
            
        return t, idx_prev_observation
    
# Loss comparing lambda, mu, sigma
class loss_naive(nn.Module):
    def __init__(self, data_dim=1):
        super().__init__()
        self.data_dim = data_dim
        
    def forward(self, lambd_t_cond, lambda_t_net, mean_cond, mean_net, trace_cov_cond, sig_net):
        sig_cond = torch.sqrt(trace_cov_cond/self.data_dim)
        loss = torch.mean((lambda_t_net-lambd_t_cond)**2)
        loss += torch.mean((mean_net-mean_cond)**2)
        loss += torch.mean((sig_net-sig_cond)**2)
        return loss

class loss_ikl(nn.Module):
    def __init__(self, tol=1e-6, data_dim=1):
        super().__init__()
        self.tol = tol
        self.data_dim = data_dim

    def forward(self, lambda_t_cond, lambda_t_net, mean_cond, mean_net, trace_cov_cond, sig_net):
        var_net = sig_net**2
        loss_lambda = lambda_t_net - lambda_t_cond + lambda_t_cond * (torch.log(lambda_t_cond.clamp(min=self.tol)) - torch.log(lambda_t_net.clamp(min=self.tol)))  
        loss_lambda = loss_lambda.mean()
        loss_sig = lambda_t_cond * 0.5 * (self.data_dim * (torch.log(var_net.clamp(min=self.tol)) - torch.log((trace_cov_cond/self.data_dim).clamp(min=self.tol)))  +  (trace_cov_cond - self.data_dim * var_net)/(var_net + self.tol)) 
        loss_sig = loss_sig.mean()
        loss_mean = lambda_t_cond * 0.5 * (torch.sum((mean_cond-mean_net)**2, dim = 1) /(var_net + self.tol))
        loss_mean = loss_mean.mean()
        loss = (loss_lambda + loss_sig + loss_mean) #.mean()
        return loss
    
def sign(x):
    s = torch.sign(x)
    s[s == 0] = 1
    return s


class GL2D_JumpMoments(torch.nn.Module):
    """
    d=2. Computes mu^J (B,2) and tr(Cov(J_t)) (B,)
    from the proposition using fixed Gauss–Laguerre quadrature.
    Entire computation is grad-free.
    """
    def __init__(self, n_quad=128, device=None, dtype=torch.float64):
        super().__init__()
        device = device or torch.device('cpu')
        # Standard Gauss–Laguerre nodes/weights (alpha=0), e^{-x} weight.
        x, w = np.polynomial.laguerre.laggauss(n_quad)  # numpy ok; no autograd
        self.register_buffer('x', torch.as_tensor(x, device=device, dtype=dtype))  # (n,)
        self.register_buffer('w', torch.as_tensor(w, device=device, dtype=dtype))  # (n,)
        self.dtype = dtype

        # constants for d=2
        self.d = 2.0
        self.twopi = torch.tensor(2.0*np.pi, device=device, dtype=dtype)
        self.pi    = torch.tensor(np.pi, device=device, dtype=dtype)
        self.eps   = torch.tensor(1e-12, device=device, dtype=dtype)

    @torch.inference_mode()
    def forward(self, a: torch.Tensor, m_t: torch.Tensor, tau_t: torch.Tensor):
        """
        a   : (B,2)
        m_t : (B,2)
        tau_t: (B,)
        Returns:
            mu_J   : (B,2)
            tr_cov : (B,)
        """
        assert a.dim()==2 and a.size(1)==2, "a must be [B,2]"
        # ----- shapes
        B = a.size(0)
        x = self.x                    # (n,)
        w = self.w                    # (n,)
        r = torch.sqrt(2.0 * x)       # (n,)
        
        # Broadcast to [B,n]
        r_bn = r.unsqueeze(0).expand(B, -1)      # [B,n]
        x_bn = x.unsqueeze(0).expand(B, -1)      # [B,n]
        w_bn = w.unsqueeze(0).expand(B, -1)      # [B,n]
        
        # Norms and r_±
        anorm    = torch.linalg.norm(a, dim=1)                      # [B]
        anorm_c  = torch.clamp(anorm, min=float(self.eps))          # [B]  (for denoms)
        anorm_bn = anorm.unsqueeze(1).expand_as(r_bn)               # [B,n] (for masking/products)
        
        r_plus  = (torch.sqrt(anorm**2 + self.d) + anorm).unsqueeze(1)   # [B,1]
        r_minus = (torch.sqrt(anorm**2 + self.d) - anorm).unsqueeze(1)   # [B,1]
        
        # Region masks
        hi  = (r_bn >= r_plus)                    # [B,n]
        mid = (r_bn >  r_minus) & (r_bn < r_plus) # [B,n]
        # low region is neither hi nor mid; will be zeros
        
        # ----- phi(r,a) only on mid
        denom = 2.0 * anorm_c.unsqueeze(1) * r_bn                    # [B,n]
        arg   = (self.d - r_bn**2) / torch.clamp(denom, min=float(self.eps))
        arg   = torch.clamp(arg, -1.0, 1.0)
        
        phi = torch.zeros_like(r_bn)
        phi = torch.where(mid, torch.arccos(arg), phi)
        
        # Trig helpers (everywhere, zeros outside mid)
        sin_phi  = torch.where(mid, torch.sin(phi), torch.zeros_like(phi))
        sin2_phi = torch.where(mid, torch.sin(2.0*phi), torch.zeros_like(phi))
        sin3_phi = torch.where(mid, torch.sin(3.0*phi), torch.zeros_like(phi))
        
        # ----- f0
        f0_hi  = self.twopi * (r_bn**2 - self.d)                     # [B,n]
        f0_mid = 2.0*phi + 4.0*anorm_bn * r_bn * sin_phi            # [B,n]
        f0 = torch.zeros_like(r_bn)
        f0 = torch.where(mid, f0_mid, f0)
        f0 = torch.where(hi,  f0_hi,  f0)
        
        # ----- f1
        f1_hi  = self.twopi * anorm_bn * r_bn
        f1_mid = 2.0*(r_bn**2 - self.d) * sin_phi \
                 + 2.0*anorm_bn * r_bn * (phi + 0.5*sin2_phi)
        f1 = torch.zeros_like(r_bn)
        f1 = torch.where(mid, f1_mid, f1)
        f1 = torch.where(hi,  f1_hi,  f1)
        
        # ----- f2
        f2_hi  = self.pi * (r_bn**2 - self.d)
        f2_mid = (r_bn**2 - self.d) * (phi + 0.5*sin2_phi) \
                 + 2.0*anorm_bn * r_bn * (1.5*sin_phi + (1.0/6.0)*sin3_phi)
        f2 = torch.zeros_like(r_bn)
        f2 = torch.where(mid, f2_mid, f2)
        f2 = torch.where(hi,  f2_hi,  f2)

        # ===== Quadrature (after x = r^2/2):
        # ∫ e^{-r^2/2} r f0 dr   ->  ∫ e^{-x} f0 dx         -> sum w * f0
        # ∫ e^{-r^2/2} r^2 f1 dr ->  ∫ e^{-x} sqrt(2x) f1 dx-> sum w * sqrt(2x) f1
        # ∫ e^{-r^2/2} r^3 f2 dr ->  ∫ e^{-x} (2x) f2 dx    -> sum w * (2x) f2
        denom = torch.sum(w_bn * f0, dim=1)                                # [B]
        num_mu = torch.sum(w_bn * torch.sqrt(2.0*x_bn) * f1, dim=1)        # [B]
        num_tr = torch.sum(w_bn * (2.0*x_bn) * f2, dim=1)                  # [B]

        # Ratios
        denom_safe = torch.clamp(denom, min=float(self.eps))
        ratio_mu = num_mu / denom_safe     # scalar multiplier for direction a/||a||
        ratio_v  = num_tr / denom_safe     # enters the covariance part

        # Assemble mu^J = m_t + sqrt(tau_t) * (a/||a||) * ratio_mu
        a_hat = a / anorm_c.unsqueeze(1)                     # [B,2]
        delta_mu = torch.sqrt(tau_t).unsqueeze(1) * a_hat * ratio_mu.unsqueeze(1)  # [B,2]
        # When ||a||==0, direction is undefined; set delta_mu=0 (symmetry).
        zero_mask = (anorm <= 0)
        if zero_mask.any():
            delta_mu[zero_mask] = 0.0

        mu_J = m_t + delta_mu                                 # [B,2]

        # tr(Cov) = tau_t * (num_tr/denom) - ||delta_mu||^2
        tr_cov = tau_t * ratio_v - torch.sum(delta_mu**2, dim=1)  # [B]
        
        if not mu_J.isfinite().all().item():
            print("mu_j not finite")
        if not tr_cov.isfinite().all().item():
            print("tr_cov not finite")

        return mu_J.to(dtype=m_t.dtype), tr_cov.to(dtype=tau_t.dtype)
