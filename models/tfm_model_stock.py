import torch, torch.nn as nn
import math
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
        """
        das tnextModel wird aktuell nicht genutzt. Da es aber im Paper besprochen wird,
        habe ich es schonmal angelegt, um es ggf später mit laufen zu lassen
        """
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
        self.drift = create_drift_network(model_cfg).to(self.device) #DriftModel
        self.time_sampling = model_cfg.time_sampling
        self.noise = NoiseModel(model_cfg).to(self.device) #NoiseModel
        self.tnext = tnextModel(model_cfg).to(self.device)
        self.mu = float(model_cfg.drift) #reale Drift der Trainingsdaten zum Vergleich / Testzwecken
        self.memory_switch = str(model_cfg.memory_switch) #Memory an / aus
        
        self.bridge_noise_mode = str(model_cfg.bridge_noise_mode) #bridge_noise_mode constant / ml / learned
        self.sigma_gt = float(model_cfg.sigma)
        self.sigma_base = float(model_cfg.sigma_tau) #sigma_base oder wie es im Paper heisst, einfach nur sigma ist eine konstante Zahl
        self.sigma_tau_param = nn.Parameter(torch.tensor(float(model_cfg.sigma_tau**2), device=self.device, dtype=torch.float32))
        self.sigma_tau_lr = float(model_cfg.sigma_tau_lr)
        self.trainable_parts = model_cfg.trainable_parts

    def forward(self, x, t, x_mem, t_mem, t2): # TODO eventually self.drift should accept these arguments
        """
        x of shape [batchsize, state_dim]
        t, t2 of shape [batchsize, 1]
        x_mem of shape [batchsize, memory_length, state_dim]
        t_mem of shape [batchsize, memory_length, 1]
        """    
        batchsize = x.shape[0]
        single_vals = torch.cat([x, t, t2], dim=-1)
        
        if self.memory_switch == "on":
            memory_flat = torch.cat([x_mem, t_mem], dim=-1).reshape(batchsize, -1)
            inpu = torch.cat([single_vals, memory_flat], dim=-1)
        else:
            inpu = torch.cat([single_vals], dim=-1)
        return self.drift(inpu)
    

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

        if self.bridge_noise_mode == "ml":
            t, idx1, idx2, idx3 = self._draw_t(data, times, mask)
            idx_prev = idx1
        
            # draw x and calculate conditional velocities
            x1 = data[torch.arange(batch_size, device=self.device), idx1]
            x2 = data[torch.arange(batch_size, device=self.device), idx2]
            x3 = data[torch.arange(batch_size, device=self.device), idx3]
            t1 = times[torch.arange(batch_size, device=self.device), idx1].unsqueeze(1) 
            t2 = times[torch.arange(batch_size, device=self.device), idx2].unsqueeze(1) 
            t3 = times[torch.arange(batch_size, device=self.device), idx3].unsqueeze(1)

            lt = (t3-t2)*(t2-t1)/(t3-t1)
            mt = (t3-t2)/(t3-t1)*x1 + (t2-t1)/(t3-t1)*x3
            bridge_noise = (x2-mt)**2/lt
            #bridge_noise = torch.sqrt(bridge_noise.mean())
            bridge_noise = torch.sqrt(bridge_noise)
        else:
            t, idx_prev = self._draw_t(data, times, mask)
            
            # draw x and calculate conditional velocities
            x1 = data[torch.arange(batch_size, device=self.device), idx_prev]
            x3 = data[torch.arange(batch_size, device=self.device), idx_prev + 1]
            t1 = times[torch.arange(batch_size, device=self.device), idx_prev].unsqueeze(1) 
            t3 = times[torch.arange(batch_size, device=self.device), idx_prev + 1].unsqueeze(1)

            if self.bridge_noise_mode == "constant":
                bridge_noise = torch.tensor(self.sigma_base, device=self.device, dtype=data.dtype)
            elif self.bridge_noise_mode == "learned":
                bridge_noise = torch.sqrt(self.sigma_tau_param.clamp(min=1e-8))
            else:
                raise ValueError(f"Unknown bridge_noise_mode: {self.bridge_noise_mode}")

        x_mem, t_mem = get_memory(data, times, idx_prev, self.memory_length)

        #print(f"t={t[1, 0]}: t_mem[1] = {t_mem[1, :, 0].cpu().numpy()}")
        #print(f"t={t[1, 0]}: x_mem[1] = {x_mem[1, :, 0].cpu().numpy()}")

        mt = (t3-t)/(t3-t1) * x1 + (t-t1)/(t3-t1) * x3 #Mittelwert
        tau_t = torch.sqrt((t3-t)*(t-t1)/(t3-t1).clamp_min(1e-3))*bridge_noise
        x = mt + tau_t * torch.randn_like(x1) #hierdrauf lernen unsere Netze

        #Das ist für den Fall, das wir Drift vorhersagen wollen
        if "drift" in self.trainable_parts:
            pred_ut = self.forward(x, t, x_mem, t_mem, t3)
        else: 
            pred_ut = self.mu
        ut = (x3-x) / (t3-t).clamp_min(1e-3) 
        #pred_x = x + pred_ut * (t3-t)

        if "uncertainty" in self.trainable_parts:
            pred_noise = self.noise(x, t = t, x_mem = x_mem, t_mem = t_mem, t2 = t3)
        else:
            pred_noise = (self.sigma_gt**2)
        # pred_x = self.forward(x, t = t, x_mem = x_mem, t_mem = t_mem, t2 = t3)

        #Losses für Targetprediction
        # dt = (t3 - t).clamp_min(1e-3) 
        # loss_target = torch.mean((pred_x - x3)**2)
        # loss_noise = torch.mean((pred_noise - torch.abs(pred_x.clone().detach() - x3)**2 / dt)**2)

        #Das ist für den Fall, das wir Drift vorhersagen wollen
        dt = (t3 - t)
        loss_target = torch.mean((pred_ut - ut)**2)
        loss_noise = torch.mean((pred_noise - torch.abs(pred_ut.clone().detach() - ut)**2 * dt)**2)

        #Losses SDE-Fall vgl TFM code
        #h = (t3-t)*(t-t1)/(t3-t1) #rescale unsere skala auf [t1~0, t3~1] vgl TFM code 
        #loss_target = torch.mean((pred_x + torch.sqrt(h) * torch.sqrt(pred_noise.clone().detach()) * torch.randn_like(x1) - x3)**2)
        #loss_noise = torch.mean((pred_x.clone().detach() + torch.sqrt(h) * torch.sqrt(pred_noise) * torch.randn_like(x1) - x3)**2)
        s = (t-t1)/(t3-t1)
        v_star = ((bridge_noise**2*2)+(0.09-bridge_noise**2)/(t3-t1)*(x-x1))/(s*0.09+bridge_noise**2*(1-s))
        test = (torch.abs(pred_ut.clone().detach() - v_star)**2)*(t3-t)

        return loss_target, loss_noise, pred_noise, pred_ut, bridge_noise, test

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

        if self.bridge_noise_mode == "ml":
            batch_idx = torch.arange(batch_size, device=self.device)
            idx_left = idx_prev_observation - 1          # [B]
            idx_right = idx_prev_observation + 2         # [B]
            # check ob die Indizes überhaupt valide sind
            left_valid = idx_left >= 0
            right_valid = idx_right < nb_timepoints
            # Distanztensoren default gefüllt mit infinity
            t_flat = t.squeeze(-1)                       # [B]
            big = torch.full_like(t_flat, float('inf'))
            dist_left = big.clone()
            dist_right = big.clone()
            # Distanzen nur dort berechnen, wo der Index gültig ist
            dist_left[left_valid] = torch.abs(
                t_flat[left_valid] - times[batch_idx[left_valid], idx_left[left_valid]]
            )
            dist_right[right_valid] = torch.abs(
                t_flat[right_valid] - times[batch_idx[right_valid], idx_right[right_valid]]
            )
            # Wenn beide gültig: wähle den mit der kleineren Distanz
            choose_left = dist_left <= dist_right
            candidate_idx3 = torch.where(choose_left, idx_left, idx_right)
            # Drei Indizes zusammen
            indices_three = torch.stack([
                idx_prev_observation,
                idx_prev_observation + 1,
                candidate_idx3
            ], dim=1)   # -> Form (B, 3)
            # Sortieren: klein, mittel, groß
            sorted_idx, _ = torch.sort(indices_three, dim=1)

            return t, sorted_idx[:,0], sorted_idx[:,1], sorted_idx[:,2]
        else:
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
                
                # x2_pred = self.forward(x, t, x_mem, t_mem, t2)
                # drift = (x2_pred - x) / (t2 - t)
                # sigma = self.noise(x, t, x_mem, t_mem, t2)
                
                if "drift" in self.trainable_parts:
                    drift = self.forward(x, t, x_mem, t_mem, t2)
                else:
                    drift = torch.full_like(x, self.mu) * x

                if "uncertainty" in self.trainable_parts:
                    sigma = self.noise(x, t, x_mem, t_mem, t2)
                else:
                    sigma = torch.full_like(x, self.sigma_gt**2) * x**2
                
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