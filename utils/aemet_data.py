from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split

import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from typing import Dict
import pandas as pd

class AemetDataset(Dataset):
    def __init__(self, x: torch.Tensor, t: torch.Tensor):
        # x: [N, T, D], t: [N, T]
        self.x, self.t = x, t

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[i], "t": self.t[i]}
    
def _seed_all(seed: int):
    torch.manual_seed(seed)
    # numpy seed brauchst du hier nicht zwingend, aber schadet nicht
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def _subsample(x: torch.Tensor, t_1d: torch.Tensor, T_sub: int, fix_min_max: bool, seed: int) -> dict:
    """
    Wie in deinem Dummydata-Script:
      x:   [N, T, D]
      t_1d:[T]
    Returns:
      {"x":[N, T_sub, D], "t":[N, T_sub], "idx":[N, T_sub]}
    """
    N, T, D = x.shape
    assert T_sub <= T
    _seed_all(seed)
    device = x.device

    if fix_min_max:
        if T_sub < 2:
            raise ValueError("T_sub must be >= 2 when fix_min_max=True.")
        idx_interior = torch.rand(N, T - 2, device=device).argsort(dim=1)[:, : T_sub - 2] + 1
        idx, _ = torch.sort(idx_interior, dim=1)
        idx_min = torch.zeros(N, 1, dtype=torch.long, device=device)
        idx_max = torch.full((N, 1), fill_value=T - 1, dtype=torch.long, device=device)
        idx = torch.cat((idx_min, idx, idx_max), dim=1)
    else:
        idx = torch.rand(N, T, device=device).argsort(dim=1)[:, :T_sub]
        idx, _ = torch.sort(idx, dim=1)

    row = torch.arange(N, device=device).unsqueeze(1).expand(N, T_sub)
    x_sub = x[row, idx, :]  # [N, T_sub, D]

    # expand t to [N, T] then gather
    t_sub = t_1d.unsqueeze(0).expand(N, T).gather(1, idx)  # [N, T_sub]

    return {"x": x_sub, "t": t_sub, "idx": idx}

def make_loaders(data_cfg: DictConfig, train_cfg: DictConfig):
    # --- CSV laden ---
    data_raw = pd.read_csv("C:/Users/Startklar/Documents/Uni/MA/functional_flow_matching/data/aemet.csv", index_col=0)
    x = torch.tensor(data_raw.values, dtype=torch.float32)  # [N, T] typisch

    if x.ndim == 2:
        x = x.unsqueeze(-1)  # -> [N, T, 1]
    
    N, T, D = x.shape

    # --- Zeitgitter t:[T] ---
    t_start = float(getattr(data_cfg, "t_start", 0.0))
    t_end   = float(getattr(data_cfg, "t_end", 1.0))
    t_1d = torch.linspace(t_start, t_end, T, dtype=torch.float32)  # [T]

    # --- Split ---
    _seed_all(int(train_cfg.manual_seed))
    perm = torch.randperm(N)
    #train_fraction = float(getattr(data_cfg, "train_fraction", 0.8))
    #n_train = int(train_fraction * N)
    n_train = int(56)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_x = x[train_idx]   # [N_train, T, D]
    val_x   = x[val_idx]     # [N_val,   T, D]

    #Normalizing data fuer (hoffentlich) mehr Stabilitaet
    mu = train_x.mean()
    std = train_x.std() + 1e-6
    train_x = (train_x - mu) / std
    val_x   = (val_x   - mu) / std

    # --- Train subsampling (optional) ---
    T_sub = int(getattr(data_cfg, "T_sub", T))
    fix_min_max = bool(getattr(data_cfg, "fix_min_max", False))
    train = _subsample(train_x, t_1d, T_sub, fix_min_max, seed=int(train_cfg.manual_seed))

    train_ds = AemetDataset(train["x"], train["t"])
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # --- Val FULL (wichtig für deinen Trainer) ---
    # t wird wie im Dummycode auf [N_val, T] expanded
    val_full = {
        "x": val_x,
        "t": t_1d.unsqueeze(0).expand(val_x.shape[0], -1),
    }


    return train_loader, val_full