import _pathfix
import hydra
import torch, torch.nn as nn
from omegaconf import DictConfig
import os
import matplotlib.pyplot as plt
from pathlib import Path
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict
import math

from tgm.utils.seed_all import _seed_all
from tgm.utils.plot_trajectories import _trajectory_plot_all_dims
from tgm.utils.output_path import _output_path
from tgm.utils.stock_data import StockDataset, _subsample
from tgm.train.trainer import Trainer
from tgm.train.callbacks import WandbCallback, PrintCallback, PlottingCallback, SavingCallback

from tgm.models.drift_diff_model import DriftDiffusionModel
from tgm.models.jump_model import JumpModel
from tgm.models.jump_model_uncoupled import JumpModelUncoupled
from tgm.models.jump_model_full_cov import JumpModelFullCov
from tgm.models.tfm_model import TfmModel

MODEL_REGISTRY = {
    "DriftDiffusionModel": DriftDiffusionModel,
    "JumpModel": JumpModel,
    "JumpModelUncoupled": JumpModelUncoupled,
    "JumpModelFullCov": JumpModelFullCov,
    "TfmModel": TfmModel
}

@hydra.main(config_path="../conf", config_name="config_spiral", version_base=None)
def main(cfg: DictConfig):
    _seed_all(cfg.train.manual_seed)
    
    # qick hack to automatically choose no_bridges dependent on T_sub
    # if cfg.data.T_sub != 101:
    #     cfg.train.no_bridges = cfg.data.T_sub 
    # else:
    #     cfg.train.no_bridges = 100
    
    # hardcoded, since handling irrationals in yaml is awkward
    cfg.train.t_end = 2 * math.pi
    cfg.train.stepsize = 2 * math.pi * 1e-3
    
    # create datasets and dataloader
    train_x = make_corkscrew_dataset(cfg.data.size_train, cfg.data.n_steps+1, noise_std=cfg.data.noise_std, device=cfg.train.device)
    val_x = make_corkscrew_dataset(cfg.data.size_val, cfg.data.n_steps+1, noise_std=cfg.data.noise_std, device=cfg.train.device)
    test_x = make_corkscrew_dataset(cfg.data.size_test, cfg.data.n_steps+1, noise_std=cfg.data.noise_std, device=cfg.train.device)
    t = np.linspace(0, 2 * math.pi, cfg.data.n_steps+1)
    
    t = torch.tensor(t, dtype=torch.float32, device=cfg.train.device)
    train_full = {"x": train_x, "t": t.unsqueeze(0).expand(train_x.shape[0], -1)}
    val_full = {"x": val_x, "t": t.unsqueeze(0).expand(val_x.shape[0], -1)}
    train = _subsample(train_x, t, cfg.data.T_sub, cfg.data.fix_min_max, cfg.train.manual_seed)
    train_ds = ToyDataset(train["x"], train["t"])
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_sub = None
    
    # plot and save test set
    # fig = _trajectory_plot_all_dims(test_x, t.unsqueeze(0).expand(test_x.shape[0], -1))
    # output_path = _output_path(cfg)
    # fig.savefig(os.path.join(output_path, "trajectories_test_set.png"), dpi=300)
    # plt.show(fig)
    
    model = DriftDiffusionModel(cfg.model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        
    # callbacks = [PrintCallback(), PlottingCallback(), SavingCallback(cfg)]
    callbacks = [PrintCallback(), SavingCallback(cfg), 
                 WandbCallback(project = os.getenv("WANDB_PROJECT", "TGM-debug"), run_name = os.getenv("WANDB_RUN_NAME", "test_run"))]
    
    trainer = Trainer(cfg.train, model, optimizer, train_loader, val_sub, val_full, callbacks)
    
    trainer.train()
    
    
class ToyDataset(Dataset):
    def __init__(self, x: torch.Tensor, t: torch.Tensor):
        self.x, self.t = x, t  # x:[N,T,D], t:[N,T]
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[i], "t": self.t[i]}

def make_corkscrew_dataset(B: int, N: int, noise_std: float = 0.01, device="cpu"):
    """
    Generate B noisy corkscrew (circle) trajectories in 2D.

    Args:
        B (int): number of trajectories (batch size)
        N (int): number of time steps per trajectory
        noise_std (float): standard deviation of Gaussian noise
        device (str): "cpu" or "cuda"

    Returns:
        torch.Tensor of shape [B, N, 2]
    """
    # Time points from 0 to 2π
    t = torch.linspace(0, 2 * torch.pi, N, device=device)

    # Base deterministic trajectory on the unit circle
    x = torch.cos(t)
    y = torch.sin(t)
    base_traj = torch.stack([x, y], dim=-1)        # [N, 2]

    # Repeat for all B trajectories
    trajs = base_traj.unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]

    # Add Gaussian noise
    noise = noise_std * torch.randn_like(trajs)
    return trajs + noise


if __name__ == "__main__":
    main()