import copy
import warnings
from typing import List, Optional
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from tgm.utils.metrics import mmd_metric, sinkhorn_dist
from tgm.train.callbacks import Callback
import math

class Trainer:
    def __init__(self, train_cfg: DictConfig, model, optimizer, train_loader, val_sub = None, val_full = None,
                 callbacks: Optional[List[Callback]] = None):
        self.cfg = train_cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_sub = val_sub
        self.val_full = val_full
        self._callbacks: List[Callback] = callbacks or []
        

    def train(self):
        
        global_step = 0
        
        self._cb("on_train_start", no_epochs=self.cfg.no_epochs)
        
        for i in range(self.cfg.no_epochs):
            
            epoch_loss = 0.0
            #epoch_sigma = 0.0
            
            for j, batch in enumerate(self.train_loader):
                global_step += 1
                loss = self._step(batch)
                #loss, sigma = self._step(batch)
                epoch_loss = (j / (j + 1)) * epoch_loss + (1 / (j + 1)) * loss
                #epoch_sigma = (j / (j + 1)) * epoch_sigma + (1 / (j + 1)) * sigma
                
                self._cb("on_step_end", step=global_step, loss=loss)

            #print("sigma:", epoch_sigma.mean())
            #print("n_train batches per epoch:", len(self.train_loader))
            self._cb("on_epoch_end", step=global_step, epoch = i, loss=epoch_loss)
    
            mmd, sinkhorn, trajectories, times = self._validate()
                
            self._cb("on_validation_end", step=global_step, mmd=mmd, sinkhorn = sinkhorn,
                     trajectories = trajectories, times = times) 
                
        self._cb("on_train_end")
                
        return

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        
        if self.cfg.regular_val_set_provided:
            # load validation set
            val_samples = self.val_full["x"]
            
            # generate as many sample paths
            x0 = torch.tensor(OmegaConf.to_container(self.cfg.x_start), dtype=torch.float32)
            #aemet:
            #x0 = val_samples[:, 0, :] # [N,1] statt cfg.x_start
            samples, times, _ = self.model.sample_unif(x0, self.cfg.no_bridges, self.cfg.t_start, self.cfg.t_end, self.cfg.stepsize, no_samples = val_samples.shape[0])
            
            i = 0
            x = samples[i]
            t = times[i]
            print(x.shape)
            print(t.shape)

            traj = x.detach().cpu().numpy()
            time = t.detach().cpu().numpy()

            # --- Output-Ordner ---
            out_dir = "debug_plots"
            os.makedirs(out_dir, exist_ok=True)

            # --- Dateiname (Epoch & Batch optional) ---
            current_time = datetime.now()
            print(current_time.strftime("%H_%M_%S"))
            fname = f"trajectory_no_{current_time.strftime("%H_%M_%S")}.png"
            path = os.path.join(out_dir, fname)

            # --- Plot ---
            plt.figure(figsize=(6, 4))
            plt.plot(time, traj, marker="o")
            plt.xlabel("t")
            plt.ylabel("x")
            plt.title(f"Validation trajectory")
            plt.grid(True)

            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()   # <<< wichtig bei Training!

            # subsample sample paths to get same time grid as validation paths
            trajectory_length = samples.shape[1]
            assert trajectory_length == round((self.cfg.t_end - self.cfg.t_start) / self.cfg.stepsize + 1)
            equidistant_steps = self.cfg.val_trajectory_length - 1
            assert int(trajectory_length - 1) % equidistant_steps == 0
            stepsize_subsampling = int((trajectory_length - 1) / equidistant_steps)
            samples_val_grid = samples[:, torch.arange(0, trajectory_length, stepsize_subsampling), :] #potentially just interpolate if grid cannot be aligned
            
            # compute metrics
            mmd = mmd_metric(samples_val_grid, val_samples)
            sinkhorn = sinkhorn_dist(samples_val_grid, val_samples)
        
        return mmd, sinkhorn, samples_val_grid, times[:, torch.arange(0, trajectory_length, stepsize_subsampling)]
        
        # Note:
        # complete trajectory mmd/sinkhorn works only for uniform timegrid
        # but mse is correlated an works for arbitray timegrids
        # so choose mse for stopping, when working with eicu data?!

    def _step(self, batch):
        
        self.model.train()
        self.optimizer.zero_grad()
        #loss , sigma = self.model.loss(batch)
        loss = self.model.loss(batch)
        if not torch.isfinite(loss):
            print("Non-finite loss, skipping step.")
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item() #, sigma
    
    def _cb(self, name: str, **kw):
        for cb in self._callbacks:
            fn = getattr(cb, name, None)
            if fn is None:
                continue
            try:
                fn(trainer=self, **kw)
            except Exception as e:
                warnings.warn(f"Callback {cb.__class__.__name__}.{name} failed: {e}")
