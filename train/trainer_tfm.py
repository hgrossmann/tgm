import copy
import warnings
from typing import List, Optional
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tgm.utils.metrics import mmd_metric, sinkhorn_dist
from tgm.train.callbacks import Callback
import math

class Trainer:
    def __init__(self, train_cfg: DictConfig, model, optimizer_target, optimizer_noise, train_loader, val_sub = None, val_full = None,
                 callbacks: Optional[List[Callback]] = None):
        self.cfg = train_cfg
        self.model = model
        self.optimizer_target = optimizer_target
        self.optimizer_noise = optimizer_noise
        self.train_loader = train_loader
        self.val_sub = val_sub
        self.val_full = val_full
        self._callbacks: List[Callback] = callbacks or []
        

    def train(self):
        
        global_step = 0
        
        self._cb("on_train_start", no_epochs=self.cfg.no_epochs)
        
        for i in range(self.cfg.no_epochs):
            
            epoch_loss = 0.0
            epoch_loss_target = 0.0
            epoch_loss_noise = 0.0

            epoch_noise = 0.0
            epoch_target = 0.0
            epoch_bn = 0.0
            epoch_test = 0.0
            
            for j, batch in enumerate(self.train_loader):
                global_step += 1
                #loss = self._step(batch)
                loss_target, loss_noise, noise, target, bridge_noise, test = self._step(batch)
                #if j%10 == 0:
                #    print("Batch", j)
                #    print("loss-target: ", loss_target)
                #    print("L-sigma: ", noise.mean(), "-", loss_target, "=", noise.mean()-loss_target)
                #    print("loss-noise: ", loss_noise)
                #    print(" ")
                loss = loss_target + loss_noise
                epoch_loss_target = (j / (j + 1)) * epoch_loss_target + (1 / (j + 1)) * loss_target
                epoch_loss_noise = (j / (j + 1)) * epoch_loss_noise + (1 / (j + 1)) * loss_noise

                epoch_noise = (j / (j + 1)) * epoch_noise + (1 / (j + 1)) * noise.mean().detach()
                epoch_target = (j / (j + 1)) * epoch_target + (1 / (j + 1)) * target.mean().detach()
                epoch_bn = (j / (j + 1)) * epoch_bn + (1 / (j + 1)) * bridge_noise
                epoch_test = (j / (j + 1)) * epoch_test + (1 / (j + 1)) * test.mean().detach()

                epoch_loss = epoch_loss_target + epoch_loss_noise
                #epoch_loss = (j / (j + 1)) * epoch_loss + (1 / (j + 1)) * loss
                
                self._cb(
                    "on_step_end",
                    step=global_step,
                    total_loss=loss,
                    drift_loss=loss_target,
                    noise_loss=loss_noise,
                    noise_mean=noise.mean().detach().item(),
                    drift_mean=target.mean().detach().item(),
                    bridge_noise=float(bridge_noise),
                    test_mean=test.mean().detach().item(),
                )

            print(" ")
            print("Epoch noise: ", epoch_noise)
            print("Epoch Drift: ", epoch_target)
            print("Epoch Bridge noise: ", epoch_bn)
            print("Lossrest: ", epoch_test)
            print(" ")

            self._cb(
                "on_epoch_end",
                step=global_step,
                epoch=i,
                total_loss=float(epoch_loss),
                drift_loss=float(epoch_loss_target),
                noise_loss=float(epoch_loss_noise),
                noise_mean=epoch_noise.detach().item() if torch.is_tensor(epoch_noise) else float(epoch_noise),
                drift_mean=epoch_target.detach().item() if torch.is_tensor(epoch_target) else float(epoch_target),
                bridge_noise=float(epoch_bn),
                test_mean=epoch_test.detach().item() if torch.is_tensor(epoch_test) else float(epoch_test),
            )
    
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
        self.optimizer_target.zero_grad()

        #loss_target = self.model.loss(batch)
        loss_target, loss_noise, cond_noise, xt, bridge_noise, test = self.model.loss(batch)
        
        if self.model.bridge_noise_mode == "learned":
            if self.model.sigma_tau_param.grad is not None:
                self.model.sigma_tau_param.grad.zero_()

        if "drift" in self.model.trainable_parts:
            if not torch.isfinite(loss_target):
                print("Non-finite target loss, skipping step.")
            else:
                loss_target.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(list(self.model.drift.parameters()) + list(self.model.tnext.parameters()), 1.0)
                self.optimizer_target.step()

        self.optimizer_noise.zero_grad()
        if "uncertainty" in self.model.trainable_parts:
            if not torch.isfinite(loss_noise):
                print("Non-finite noise loss, skipping step.")
            else:
                loss_noise.backward()
                torch.nn.utils.clip_grad_norm_(self.model.noise.parameters(), 1.0)
                self.optimizer_noise.step()

        if self.model.bridge_noise_mode == "learned":
            with torch.no_grad():
                self.model.sigma_tau_param -= (self.model.sigma_tau_lr * self.model.sigma_tau_param.grad)
                self.model.sigma_tau_param.clamp_(min=1e-5)
                self.model.sigma_tau_param.grad = None

        return loss_target.item(), loss_noise.item(), cond_noise, xt, bridge_noise.item(), test
    
    def _cb(self, name: str, **kw):
        for cb in self._callbacks:
            fn = getattr(cb, name, None)
            if fn is None:
                continue
            try:
                fn(trainer=self, **kw)
            except Exception as e:
                warnings.warn(f"Callback {cb.__class__.__name__}.{name} failed: {e}")
