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

from tgm.models.tfm_model import DriftDiffusionModel, NoiseModel, tnextModel

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

    #kleiner Workaround für aemet data:
    #train_loader, val_full = make_loaders(cfg.data, cfg.train)
    #val_sub = None
    #test_full = None
    
    model_class = MODEL_REGISTRY[cfg.model.model_name]
    model = model_class(cfg.model)
    
    optimizer_target = torch.optim.Adam(
    list(model.drift.parameters()) + list(model.tnext.parameters()),
    lr=cfg.train.lr
) # theoretisch koennen wir hier noch ein tnext modell rein geben. Aktuell lauft das nicht
    
    optimizer_noise = torch.optim.Adam(model.noise.parameters(), lr=cfg.train.lr)

    callbacks = [PrintCallback(), SavingCallback(cfg), 
                 WandbCallback(project = "TrajectoryFlowMatching", 
                               run_name = os.getenv("WANDB_RUN_NAME", "test_run"),
                               config = {
                                    "no_epochs": cfg.train.no_epochs,
                                    "trainable parts": cfg.model.trainable_parts,
                                    
                                    # memory
                                    "memory/switch": cfg.model.memory_switch,
                                    "memory/length": cfg.model.memory_length,                                   
                                    
                                    # brigde noise setting
                                    "bridgenoise/Modus": cfg.model.bridge_noise_mode,
                                    "bridgenoise/sigma_tau":cfg.model.sigma_tau,
                                    
                                    # time steps / spacing
                                    "time/stepsize training":cfg.data.T_sub,
                                    "time/spacing":cfg.data.time_spacing,

                                    # mixture setup
                                    "mixture/enabled": cfg.data.mixture.enabled,

                                    "mixture_disabled/drift": cfg.model.drift,
                                    "mixture_disabled/sigma": cfg.model.sigma,

                                    "mixture_enabled/p_bm1": cfg.data.mixture.p_bm1,
                                    "mixture_enabled/bm1/drift": cfg.data.mixture.bm1.drift,
                                    "mixture_enabled/bm1/volatility": cfg.data.mixture.bm1.volatility,
                                    "mixture_enabled/bm1/S0": cfg.data.mixture.bm1.S0,
                                    "mixture_enabled/bm2/drift": cfg.data.mixture.bm2.drift,
                                    "mixture_enabled/bm2/volatility": cfg.data.mixture.bm2.volatility,
                                    "mixture_enabled/bm2/S0": cfg.data.mixture.bm2.S0,
                                })]
    
    trainer = Trainer(cfg.train, model, optimizer_target, optimizer_noise, train_loader, val_sub, val_full, callbacks)
    
    trainer.train()

    # speichere das modell um es spaeter wieder aufrufen zu koennen
    save_path = "../checkpoints/model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg
    }, save_path)
    
    
if __name__ == "__main__":
    main()