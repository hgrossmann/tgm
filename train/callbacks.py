import os, time
from typing import Any, Dict, List, Optional
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from datetime import datetime


class Callback:
    def on_train_start(self, trainer, **kwargs): ...
    def on_train_end(self, trainer, **kwargs): ...
    def on_epoch_start(self, trainer, **kwargs): ...
    def on_epoch_end(self, trainer, **kwargs): ...
    def on_step_end(self, trainer, **kwargs): ...
    def on_validation_end(self, trainer, **kwargs): ...


class WandbCallback(Callback):
    def __init__(self, project: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.project = project
        self.run_name = run_name
        self.config = config

    def on_train_start(self, trainer, **kwargs):
        if wandb.run is None:
            init_kwargs = {
                "project": self.project,
                "name": self.run_name,
                "settings": wandb.Settings(
                    _service_wait=300,
                    x_disable_stats=True,
                ),
            }
            if self.config is not None:
                init_kwargs["config"] = self.config

            wandb.init(**init_kwargs)

    def on_step_end(self, trainer, step: int, **kwargs):
        if wandb.run is None:
            return

        wandb.log({
            "train/total_loss": kwargs["total_loss"],
            "train/drift_loss": kwargs["drift_loss"],
            "train/noise_loss": kwargs["noise_loss"],

            "noise/pred_mean": kwargs["noise_mean"],
            "noise/pred_mean_0": kwargs["noise_0_mean"],
            "noise/pred_mean_1": kwargs["noise_1_mean"],
            "noise/bridge_noise": kwargs["bridge_noise"],

            "drift/pred_mean": kwargs["drift_mean"],
            "drift/pred_mean_0": kwargs["drift_0_mean"],
            "drift/pred_mean_1": kwargs["drift_1_mean"],

            "debug/Lossrest_mean": kwargs["test_mean"],
        }, step=step)

    def on_epoch_end(self, trainer, step: int, epoch: int, **kwargs):
        if wandb.run is None:
            return

        wandb.log({
            "epoch/total_loss": kwargs["total_loss"],
            "epoch/drift_loss": kwargs["drift_loss"],
            "epoch/noise_loss": kwargs["noise_loss"],

            "epoch/noise_mean": kwargs["noise_mean"],
            "epoch/noise_0_mean": kwargs["noise_0_mean"],
            "epoch/noise_1_mean": kwargs["noise_1_mean"],
            "epoch/bridge_noise": kwargs["bridge_noise"],
            "epoch/drift_mean": kwargs["drift_mean"],
            "epoch/drift_0_mean": kwargs["drift_0_mean"],
            "epoch/drift_1_mean": kwargs["drift_1_mean"],
            "epoch/Lossrest_mean": kwargs["test_mean"],
        }, step=step)

    def on_validation_end(self, trainer, **kwargs):
        if wandb.run is None:
            return
        if "step" not in kwargs:
            return

        wandb.log({
            "val/mmd": float(kwargs["mmd"]),
            "val/sinkhorn": float(kwargs["sinkhorn"]),
        }, step=kwargs["step"])

    def on_train_end(self, trainer, **kwargs):
        if wandb.run is not None:
            wandb.finish()
        
        
class PrintCallback(Callback):  

    def _is_main(self):
        return int(os.environ.get("RANK", "0")) == 0

    def on_train_start(self, trainer, no_epochs: int, **kwargs):
        if not self._is_main(): return
        print(f"Training for {no_epochs} epochs...")
        self._t0 = time.time()
        self.no_epochs = no_epochs

    def on_epoch_end(self, trainer, epoch: int, total_loss: float, **kwargs):
        if not self._is_main(): return
        print(f"Epoch {epoch}: loss {total_loss:.4f}", end="")
            
    def on_validation_end(self, trainer, mmd, sinkhorn, **kwargs):
        if not self._is_main(): return
        print(f" mmd {mmd:.4f}, sinkhorn {sinkhorn: .4f}")
        
    def on_train_end(self, trainer, **kwargs):
        if not self._is_main(): return
        print("... training finished.")
        if hasattr(self, "_t0"):
            duration = time.time() - self._t0
            t_epoch_avg = duration / self.no_epochs
            print(f"Average time per epoch (incl. validation loop) {t_epoch_avg:.2f}s.")
        
        
class PlottingCallback(Callback):
    def __init__(self, no_traj_to_plot = 100, plot_all_dims = True, dim_to_plot = 0):
        self.no_traj_to_plot = no_traj_to_plot
        self.dim_to_plot = dim_to_plot
        self.plot_all_dims = plot_all_dims
    
    def _is_main(self):
        return int(os.environ.get("RANK", "0")) == 0
    
    def on_validation_end(self, trainer, trajectories, times, **kwargs):
        if not self._is_main(): return
        times_cpu = times.detach().cpu().numpy()
        trajectories_cpu = trajectories.detach().cpu().numpy()
        
        if self.plot_all_dims:
            dims = trajectories_cpu.shape[-1]
            fig, axes = plt.subplots(1, dims, figsize=(8*dims, 5))
            axes = [axes] if dims == 1 else axes
            for i in range(self.no_traj_to_plot):
                for j in range(dims):
                    axes[j].plot(times_cpu[i,:], trajectories_cpu[i,:,j])
        else:
            fig, ax = plt.subplots()
            for i in range(self.no_traj_to_plot):
                ax.plot(times_cpu[i,:], trajectories_cpu[i,:,self.dim_to_plot])
                
        plt.show()
        
class SavingCallback(Callback):
    def __init__(self, cfg, no_traj_to_plot = 100, plot_all_dims = True, dim_to_plot = 0):
        self.no_traj_to_plot = no_traj_to_plot
        self.dim_to_plot = dim_to_plot
        self.plot_all_dims = plot_all_dims
        self.best_sinkhorn = float("inf")
        self.best_mmd = float("inf")
        self.path = self._output_path(cfg)
        with open(os.path.join(self.path, "config_dump.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        
    def _is_main(self):
        return int(os.environ.get("RANK", "0")) == 0
        
    def on_train_start(self, trainer, **kwargs):
        if not self._is_main(): return
        self.loss_list = []
        self.sinkhorn_list = []
        self.mmd_list = []
        
    def on_epoch_end(self, trainer, total_loss: float, **kwargs):
        if not self._is_main(): return
        self.loss_list.append(total_loss)
        
    def on_validation_end(self, trainer, sinkhorn: float, mmd: float, trajectories, times, **kwargs):
        if not self._is_main(): return
        self.mmd_list.append(mmd)
        self.sinkhorn_list.append(sinkhorn)
        
        if sinkhorn < self.best_sinkhorn:
            self.best_sinkhorn = sinkhorn
            torch.save(trainer.model.state_dict(), os.path.join(self.path, "best_state_dict_sinkhorn.pt"))
            
            fig = self._trajectory_plot(trajectories, times)
            fig.savefig(os.path.join(self.path, f"trajectories_sinkhorn{self.best_sinkhorn:.4f}.png"), dpi=300)
            plt.close(fig)
            
        if mmd < self.best_mmd:
            self.best_mmd = mmd
            torch.save(trainer.model.state_dict(), os.path.join(self.path, "best_state_dict_mmd.pt"))
            
            fig = self._trajectory_plot(trajectories, times)
            fig.savefig(os.path.join(self.path, f"trajectories_mmd{self.best_mmd:.4f}.png"), dpi=300)
            plt.close(fig)
            
        
    def on_train_end(self, trainer, **kwargs):
        if not self._is_main(): return
        np.savetxt(os.path.join(self.path, "mmd_list.txt"), self.mmd_list)
        np.savetxt(os.path.join(self.path, "sink_list.txt"), self.sinkhorn_list)
        np.savetxt(os.path.join(self.path, "loss_list.txt"), self.loss_list)
        
        with open(os.path.join(self.path, "best_state_dict.pt"), "w") as f:
            f.write(f"{self.best_sinkhorn}")
            
        plt.figure()
        plt.plot(self.loss_list)
        plt.savefig(os.path.join(self.path, "loss.png"))
        plt.close
        plt.figure()
        plt.plot(self.mmd_list)
        plt.savefig(os.path.join(self.path, "mmd.png"))
        plt.close
        plt.figure()
        plt.plot(self.sinkhorn_list)
        plt.savefig(os.path.join(self.path, "sinkhorn.png"))
        plt.close
        
    def _trajectory_plot(self, trajectories, times):
        times_cpu = times.detach().cpu().numpy()
        trajectories_cpu = trajectories.detach().cpu().numpy()
        
        if self.plot_all_dims:
            dims = trajectories_cpu.shape[-1]
            fig, axes = plt.subplots(1, dims, figsize=(8*dims, 5))
            axes = [axes] if dims == 1 else axes
            for i in range(self.no_traj_to_plot):
                for j in range(dims):
                    axes[j].plot(times_cpu[i,:], trajectories_cpu[i,:,j])
        else:
            fig, ax = plt.subplots()
            for i in range(self.no_traj_to_plot):
                ax.plot(times_cpu[i,:], trajectories_cpu[i,:,self.dim_to_plot])
        
        return fig
    
    def _output_path(self, cfg):
        save_dir = cfg.train.save_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = f"{cfg.data.dataset}_{cfg.model.model_name}_sig{cfg.model.sigma}_rho{cfg.model.rho}_lr{cfg.train.lr}_sub{cfg.data.T_sub}_{cfg.train.manual_seed}_{timestamp}"
        output_path = os.path.join(save_dir, short_id)
        os.makedirs(output_path, exist_ok=True)
        return output_path