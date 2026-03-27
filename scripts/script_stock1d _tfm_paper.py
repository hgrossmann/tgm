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
#from tgm.utils.aemet_data import make_loaders
from tgm.train.trainer_tfm import Trainer
from tgm.train.callbacks import WandbCallback, PrintCallback, PlottingCallback, SavingCallback

from tgm.models.tfm_paper_model import DriftDiffusionModel, NoiseModel, tnextModel

MODEL_REGISTRY = {
    "DriftDiffusionModel": DriftDiffusionModel,
    "NoiseModel": NoiseModel,
    "tnextModel": tnextModel
}

#@hydra.main(config_path="../conf", config_name="config_stock_1d", version_base=None)
@hydra.main(config_path="../conf", config_name="config_bm_1d", version_base=None)
#@hydra.main(config_path="../conf", config_name="config_aemet", version_base=None)
def main(cfg: DictConfig):
    _seed_all(cfg.train.manual_seed)
    
    # qick hack to automatically choose no_bridges dependent on T_sub
    # if cfg.data.T_sub != 101:
    #     cfg.train.no_bridges = cfg.data.T_sub 
    # else:
    #     cfg.train.no_bridges = 100
    
    train_loader, val_sub, val_full, test_full = make_loaders(cfg.data, cfg.train)
    #train_loader, val_full = make_loaders(cfg.data, cfg.train)
    #val_sub = None
    #test_full = None
    
    # plot and save test set
    # fig = _trajectory_plot_all_dims(test_full["x"], test_full["t"])
    # output_path = _output_path(cfg)
    # fig.savefig(os.path.join(output_path, "trajectories_test_set.png"), dpi=300)
    # plt.show(fig)
    
    model_class = MODEL_REGISTRY[cfg.model.model_name]
    model = model_class(cfg.model)
    
    optimizer_target = torch.optim.Adam(
    list(model.drift.parameters()) + list(model.tnext.parameters()),
    lr=cfg.train.lr
)
    optimizer_noise = torch.optim.Adam(model.noise.parameters(), lr=cfg.train.lr)

        
    # callbacks = [PrintCallback(), PlottingCallback(), SavingCallback(cfg), 
    #              WandbCallback(project = os.getenv("WANDB_PROJECT", "TGM-debug"), run_name = os.getenv("WANDB_RUN_NAME", "test_run"))]
    callbacks = [PrintCallback(), SavingCallback(cfg), 
                 WandbCallback(project = os.getenv("WANDB_PROJECT", "TGM-debug"), run_name = os.getenv("WANDB_RUN_NAME", "test_run"))]
    
    trainer = Trainer(cfg.train, model, optimizer_target, optimizer_noise, train_loader, val_sub, val_full, callbacks)
    
    trainer.train()

    save_path = "../checkpoints/model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg
    }, save_path)
    
    
if __name__ == "__main__":
    main()