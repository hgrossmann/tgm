import torch
import torch.nn as nn
from omegaconf import DictConfig

def create_drift_network(cfg: DictConfig) -> nn.Module:
    #input_dim = (cfg.memory_length + 1) * (cfg.data_dim + 1) + 1
    input_dim = 2 + (cfg.memory_length) * (cfg.data_dim + 1)
    width = cfg.drift_mlp_width
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, cfg.data_dim),
    )

def create_sigma_network(cfg: DictConfig) -> nn.Module:
    #input_dim für sigma model script:
    input_dim = (cfg.memory_length + 1) * (cfg.data_dim + 1) + 1
    
    width = cfg.sigma_width
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, cfg.data_dim),
    )

def create_uncertainty_network(cfg: DictConfig) -> nn.Module:
    #input_dim für tfm paper script:
    input_dim = 2 + (cfg.memory_length) * (cfg.data_dim + 1)
    #input_dim = (cfg.memory_length + 1) * (cfg.data_dim + 1) + 1

    width = cfg.sigma_width
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, cfg.data_dim),
    )

def create_observation_network(cfg: DictConfig) -> nn.Module:
    input_dim = 1 + (cfg.memory_length) * (cfg.data_dim + 1)
    width = cfg.sigma_width
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, cfg.data_dim),
    )

def create_jump_network(cfg: DictConfig) -> nn.Module:
    input_dim = (cfg.memory_length + 1) * (cfg.data_dim + 1) + 1
    output_dim = cfg.data_dim + 2
    width = cfg.jump_mlp_width
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, output_dim),
    )

def create_jump_uncoupled(cfg: DictConfig) -> nn.Module:
    input_dim = (cfg.memory_length + 1) * (cfg.data_dim + 1) + 1
    output_dim = 3*cfg.data_dim
    width = cfg.jump_mlp_width
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, output_dim),
    )