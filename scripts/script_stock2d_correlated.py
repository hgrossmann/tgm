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

@hydra.main(config_path="../conf", config_name="config_stock_2d_correlated", version_base=None)
def main(cfg: DictConfig):
    _seed_all(cfg.train.manual_seed)
    
    # qick hack to automatically choose no_bridges dependent on T_sub
    # if cfg.data.T_sub != 101:
    #     cfg.train.no_bridges = cfg.data.T_sub 
    # else:
    #     cfg.train.no_bridges = 100
    
    train_loader, val_sub, val_full, test_full = duplicate_data(cfg.data, cfg.train)
    
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
    
    
def duplicate_data(data_cfg: DictConfig, train_cfg: DictConfig):
    """
    Main entrypoint used by training:
      - ensures saved data under cfg.save_dir,
      - loads splits,
      - subsamples train/val on the fly,
      - returns:
          train_loader: yields {"x":[B,T_sub,D], "t":[B,T_sub]}
          val_sub:      {"x":[N_val, T_sub, D], "t":[N_val, T_sub]}
          val_full:     {"x":[N_val, T, D], "t":[N_val, T]}
          test_full:    {"x":[N_test, T, D], "t":[N_test, T]}
    """
    _seed_all(train_cfg.manual_seed)

    root = Path(to_absolute_path("../data/stock"))
    train_x = torch.load(root / "train_data.pt", map_location=train_cfg.device, weights_only=True)
    train_t = torch.load(root / "train_times.pt", map_location=train_cfg.device, weights_only=True)
    val_x   = torch.load(root / "val_data.pt",   map_location=train_cfg.device, weights_only=True)
    val_t   = torch.load(root / "val_times.pt",  map_location=train_cfg.device, weights_only=True)
    test_x  = torch.load(root / "test_data.pt",  map_location=train_cfg.device, weights_only=True)
    test_t  = torch.load(root / "test_times.pt", map_location=train_cfg.device, weights_only=True)
    
    train_x = train_x.expand(-1,-1,2)
    val_x = val_x.expand(-1,-1,2)
    test_x = test_x.expand(-1,-1,2)

    train = _subsample(train_x, train_t, data_cfg.T_sub, data_cfg.fix_min_max, train_cfg.manual_seed)
    val   = _subsample(val_x, val_t, data_cfg.T_sub, data_cfg.fix_min_max, train_cfg.manual_seed)
    
    train_ds = StockDataset(train["x"], train["t"])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    val_sub  = {"x": val["x"], "t": val["t"]}
    
    val_full = {"x": val_x, "t": val_t.unsqueeze(0).expand(val_x.shape[0], -1)}
    test_full = {"x": test_x, "t": test_t.unsqueeze(0).expand(test_x.shape[0], -1)}

    return train_loader, val_sub, val_full, test_full


if __name__ == "__main__":
    main()