import torch, torch.nn as nn
from typing import Optional, Dict, Any
from tgm.utils.network import create_drift_network, create_uncertainty_network, create_observation_network
from tgm.utils.memory import get_memory
from omegaconf import DictConfig
import pandas as pd
import os

class NoiseModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.device = model_cfg.device
        self.net = create_uncertainty_network(model_cfg).to(self.device)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0) # output should be positive
        self.memory_switch =model_cfg.memory_switch

    def forward(self, x, t, x_mem, t_mem, t2):
        batchsize = x.shape[0]
        single_vals = torch.cat([x, t, t2], dim=-1)
        if self.memory_switch == "on":
            memory_flat = torch.cat([x_mem, t_mem], dim=-1).reshape(batchsize, -1)
            inpu = torch.cat([single_vals, memory_flat], dim=-1)
        else:
            inpu = torch.cat([single_vals], dim=-1)
        return self.softplus(self.net(inpu))
    
class tnextModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.device = model_cfg.device
        self.net = create_observation_network(model_cfg).to(self.device)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0) # output should be positive

    def forward(self, t, x_mem, t_mem):
        batchsize = t.shape[0]
        single_vals = torch.cat([t], dim=-1)
        memory_flat = torch.cat([x_mem, t_mem], dim=-1).reshape(batchsize, -1)
        inpu = torch.cat([single_vals, memory_flat], dim=-1)
        return self.softplus(self.net(inpu))

class DriftDiffusionModel(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.rho = float(model_cfg.rho)
        self.memory_length = int(model_cfg.memory_length)
        self.device = model_cfg.device
        self.drift = create_drift_network(model_cfg).to(self.device)
        self.time_sampling = model_cfg.time_sampling
        self.noise = NoiseModel(model_cfg).to(self.device)
        self.tnext = tnextModel(model_cfg).to(self.device)
        #self.sigma_base = float(model_cfg.sigma)
        self.sigma_base = float(10.) #tfm paper
        self.mu = float(model_cfg.drift)
        self.memory_switch = model_cfg.memory_switch

    def forward(self, x, t, x_mem, t_mem, t2): # TODO eventually self.drift should accept these arguments
        """
        x of shape [batchsize, state_dim]
        t, t2 of shape [batchsize, 1]
        x_mem of shape [batchsize, memory_length, state_dim]
        t_mem of shape [batchsize, memory_length, 1]
        """    
        batchsize = x.shape[0]
        single_vals = torch.cat([x, t, t2], dim=-1)
        # single_vals = torch.cat([t2, t, x], dim=-1)
        # memory_flat = torch.cat([x_mem.reshape(batchsize, -1), t_mem.reshape(batchsize, -1)], dim=-1)
        if self.memory_switch == "on":
            memory_flat = torch.cat([x_mem, t_mem], dim=-1).reshape(batchsize, -1)
            inpu = torch.cat([single_vals, memory_flat], dim=-1)
        else:
            inpu = torch.cat([single_vals], dim=-1)
        return self.drift(inpu)
    
    def _loss_target(self, batch_size, pred_ut, ut):
        # predict drift
        loss = torch.sum((pred_ut-ut)**2)/batch_size
        #print("Drift-loss", loss)

        return loss

    def _loss_noise (self, batch_size, pred_noise, pred_x, xnext, scale):
        loss = torch.sum((pred_noise-scale*torch.abs(pred_x.detach()-xnext))**2)/batch_size
        
        #print("Noise: ", torch.sum(pred_noise**2)/batch_size)
        #print("TP-loss: ", scale, " * ", torch.sum((torch.abs(pred_x.detach()-xnext))**2)/batch_size, " = ", torch.sum((scale*torch.abs(pred_x.detach()-xnext))**2)/batch_size)
        #print("Noise-loss: ", loss)

        return loss
    
    def _loss_tp(self, batch_size, t, t2, pred_t):
        loss = torch.sum((pred_t - (t2 - t))**2)/batch_size

        return loss

    def loss(self, batch):
        data = batch["x"] # [batchsize, trajectory length, state_dim]
        times = batch["t"] # [batchsize, trajectory length]
        mask = None # batch["mask"] # TODO add nonuniform lengths later
 
        batch_size = data.shape[0]

        B, T, _ = data.shape
        x = data.squeeze(-1).detach().cpu().numpy()   # [B, T]
        t = times.detach().cpu().numpy()              # [B, T]

        # save data for consistency analyses
        os.makedirs("../output_traj", exist_ok=True)

        rows = []
        for i in range(B):
            for k in range(T):
                rows.append((i, float(t[i, k]), float(x[i, k])))

        df_long = pd.DataFrame(rows, columns=["traj_id", "t", "x"])
        df_long.to_csv("../output_traj/generated_data_long.csv", index=False)
        
        t, idx_prev = self._draw_t(data, times, mask)
        
        # draw x and calculate conditional velocities
        x1 = data[torch.arange(batch_size, device=self.device), idx_prev]
        x2 = data[torch.arange(batch_size, device=self.device), idx_prev + 1]
        t1 = times[torch.arange(batch_size, device=self.device), idx_prev].unsqueeze(1) 
        t2 = times[torch.arange(batch_size, device=self.device), idx_prev + 1].unsqueeze(1)

        x_mem, t_mem = get_memory(data, times, idx_prev, self.memory_length)

        mt = (t2-t)/(t2-t1) * x1 + (t-t1)/(t2-t1) * x2 # Mittelwert
       
        #pred_noise = self.sigma_base

        #taut = self.sigma_base*(t-t1)*(t2-t)/(t2-t1)+self.rho
        #x =  mt + torch.sqrt(taut) * torch.randn_like(x1) # same problem: wir bedingen x auf sigma
        x = mt + self.sigma_base * torch.randn_like(x1)

        #pred_t = self.tnext(t, x_mem, t_mem) # predicted next observation
        pred_t = t
        
        #Die Sachen kommen von uns und gehören hier eigentlich gar nicht hin:
        #prefactor = self.sigma_base*((t1+t2-2*t)/(t2-t1)-1)/(2*taut)
        #ut = (x2-x1)/(t2-t1) + prefactor * (x-mt)

        #Das ist für den Fall, das wir Drift vorhersagen wollen
        #pred_ut = self.forward(x, t, x_mem, t_mem, t2)
        #ut = (x2-x) / (t2-t+1e-4)
        #pred_x = x + ut * (t2-t)

        #scale = 1
        #loss_target = self._loss_target(batch_size, pred_ut, ut = ut)
        #loss_noise = self._loss_noise(batch_size, pred_noise = pred_noise, pred_x=pred_x, xnext = x2, scale=scale)
        #loss_tp = self._loss_tp(batch_size, t, t2, pred_t)

        #Wir probieren jetzt doch nochmal targetprediction genau wie im tfm code:
        tau = (t-t1)/(t2-t1)
        pred_noise = self.noise(x, t = t, x_mem = x_mem, t_mem = t_mem, t2 = t2)
        pred_x = self.forward(x, t = t, x_mem = x_mem, t_mem = t_mem, t2 = t2)
        
        h = (t2-t)*(t-t1)/(t2-t1) #rescale unsere skala auf [t1~0, t2~1] vgl TFM code 
        #loss_target = torch.mean((pred_x + torch.sqrt(h) * torch.sqrt(pred_noise.clone().detach()) * torch.randn_like(x1) - x2)**2)
        loss_target = torch.mean((pred_x - x2)**2)
        #loss_noise = torch.mean((pred_x.clone().detach() + torch.sqrt(h) * torch.sqrt(pred_noise) * torch.randn_like(x1) - x2)**2)
        loss_noise = torch.mean((pred_noise - torch.abs(pred_x.clone().detach() - x2))**2)
        delta = t2-t1
        ut = (pred_x.detach() - x1) / (delta + 1e-8)
        res = x2 - x1 - self.mu * delta   # [B,D]
        h2 = (res**2 / (delta + 1e-8)).mean()

        return loss_target, loss_noise, pred_noise, pred_x, x1, x2, delta, ut, h2

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
                
            #t2 = t_bridges[k+1].expand(no_samples, 1)
            
            for i in range(disc_steps):
                idx = k * disc_steps + i
                
                x = traj[:, idx, :]
                t = (t_bridges[k] + i * stepsize).expand(no_samples, 1)
                t2 = (t_bridges[k] + (i+1) * stepsize).expand(no_samples, 1)
                
                x2_pred = self.forward(x, t, x_mem, t_mem, t2)
                sigma = self.noise(x, t = t, x_mem = x_mem, t_mem = t_mem, t2 = t2)
                
                drift = (x2_pred - x) / (t2 - t)
                #drift = self.mu * x
                #sigma = self.sigma_base * x

                #x_new = x + stepsize * drift + torch.sqrt(stepsize) * sigma * torch.randn_like(x)
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