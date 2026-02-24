import torch, torch.nn as nn
from typing import Optional, Dict, Any
from tgm.utils.network import create_drift_network
from tgm.utils.memory import get_memory
from omegaconf import DictConfig

class TfmModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.sigma = float(model_cfg.sigma)
        self.memory_length = int(model_cfg.memory_length)
        self.device = model_cfg.device
        self.drift = create_drift_network(model_cfg).to(self.device)
        self.time_sampling = model_cfg.time_sampling

    def forward(self, x, t, x_mem, t_mem, t2): # TODO eventually self.drift should accept these arguments
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
        return self.drift(inpu)
    

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
                
                # drift = self.forward(x, t, x_mem, t_mem, t2)
                x2_pred = self.forward(x, t, x_mem, t_mem, t2)
                drift = (x2_pred - x) / (t2 - t)
                x_new = x + stepsize * drift + torch.sqrt(stepsize) * torch.sqrt(sigma) * torch.randn_like(x)
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
        taut = self.sigma*(t-t1)*(t2-t)/(t2-t1)**2
        mt = (t2-t)/(t2-t1) * x1 + (t-t1)/(t2-t1) * x2
        x =  mt + torch.sqrt(taut) * torch.randn_like(x1)
        
        # # predict velocity
        # x_mem, t_mem = get_memory(data, times, idx_prev, self.memory_length)
        # pred_vel = self.forward(x, t, x_mem, t_mem, t2)
        
        # cond_vel = (x2-x)/(t2-t1+1e-8)

        # loss = torch.sum((pred_vel-cond_vel)**2)/batch_size 
        
        # predict next point (as in TFM paper) and calculate corresponding drift only during generation
        x_mem, t_mem = get_memory(data, times, idx_prev, self.memory_length)
        pred_x2 = self.forward(x, t, x_mem, t_mem, t2)
        
        loss = torch.mean((pred_x2 - x2)**2)

        return loss
    
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