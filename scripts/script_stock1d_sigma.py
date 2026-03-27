import _pathfix
import hydra
import torch, torch.nn as nn
import numpy as np
from omegaconf import DictConfig
import os
import matplotlib.pyplot as plt

from tgm.utils.seed_all import _seed_all
from tgm.utils.plot_trajectories import _trajectory_plot_all_dims
from tgm.utils.output_path import _output_path
from tgm.utils.stock_data import make_loaders
from tgm.train.trainer import Trainer
from tgm.train.callbacks import WandbCallback, PrintCallback, PlottingCallback, SavingCallback

from tgm.models.sigma_model import DriftDiffusionModel, SigmaModel

MODEL_REGISTRY = {
    "DriftDiffusionModel": DriftDiffusionModel,
    "SigmaModel": SigmaModel
}

@hydra.main(config_path="../conf", config_name="config_stock_1d", version_base=None)
def main(cfg: DictConfig):
    _seed_all(cfg.train.manual_seed)
    
    # qick hack to automatically choose no_bridges dependent on T_sub
    # if cfg.data.T_sub != 101:
    #     cfg.train.no_bridges = cfg.data.T_sub 
    # else:
    #     cfg.train.no_bridges = 100
    
    train_loader, val_sub, val_full, test_full = make_loaders(cfg.data, cfg.train)
    
    # plot and save test set
    # fig = _trajectory_plot_all_dims(test_full["x"], test_full["t"])
    # output_path = _output_path(cfg)
    # fig.savefig(os.path.join(output_path, "trajectories_test_set.png"), dpi=300)
    # plt.show(fig)
    
    model_class = MODEL_REGISTRY[cfg.model.model_name]
    model = model_class(cfg.model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        
    # callbacks = [PrintCallback(), PlottingCallback(), SavingCallback(cfg), 
    #              WandbCallback(project = os.getenv("WANDB_PROJECT", "TGM-debug"), run_name = os.getenv("WANDB_RUN_NAME", "test_run"))]
    callbacks = [PrintCallback(), SavingCallback(cfg), 
                 WandbCallback(project = os.getenv("WANDB_PROJECT", "TGM-debug"), run_name = os.getenv("WANDB_RUN_NAME", "test_run"))]
    
    trainer = Trainer(cfg.train, model, optimizer, train_loader, val_sub, val_full, callbacks)
    
    trainer.train()
    
    
if __name__ == "__main__":
    main()