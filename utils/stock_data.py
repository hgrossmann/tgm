from pathlib import Path
from hydra.utils import to_absolute_path
from typing import Dict, Any, Tuple, Optional
import json, hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf, ListConfig

from tgm.utils.stock_model import STOCK_MODELS  # dict: name -> class  # noqa
# each classes generate_paths() returns spot_paths, dt 
# where stock_paths have dimension: [nb_paths, dimension, nb_time_points]
# where nb_time_points = nb_steps+1


# ----------------------------
# Data Creation
# ----------------------------

def create_dataset(cfg: DictConfig):

    root = Path(to_absolute_path(cfg.save_dir))
    root.mkdir(parents=True, exist_ok=True)
    
    # Generate three splits with deterministic seeds from cfg
    for split, size, seed in (("train", cfg.size_train, cfg.seed_train), 
                        ("val", cfg.size_val, cfg.seed_val), 
                        ("test", cfg.size_test, cfg.seed_test)):
        x, t, z = _gen_once(cfg, size, seed)
        p_data, p_times, p_labels, _ = _store_paths(root, split)
        torch.save(x, p_data)
        torch.save(t, p_times)
        torch.save(z, p_labels)

    # write minimal manifest
    _, _, _, manifest_p = _store_paths(root, "train")
    manifest_p.write_text(json.dumps(_to_plain(_manifest_payload(cfg)), indent=2))

    return


def _gen_once(cfg: DictConfig, size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    returns x:[N,T,D], t:[T], z:[N], 0 = BM1, 1 = BM2
    """
    mixture_cfg = getattr(cfg, "mixture", None)

    if mixture_cfg is not None and mixture_cfg.enabled:
        return _gen_mixture_once(cfg, size, seed)

    x, t = _gen_single_once(cfg, size, seed)
    z = torch.zeros(size, dtype=torch.long)
    return x, t, z


def _gen_single_once(cfg: DictConfig, size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns x:[N,T,D] CPU float32 and t:[T] CPU float32 built from the returned dt.
    N = nb of paths, T = nb of time points, D = dimension of observations
    """
    _seed_all(seed)
    Model = STOCK_MODELS[cfg.model_name]
    model_kwargs = _model_kwargs(cfg)
    model_kwargs["nb_paths"] = size
    model = Model(**model_kwargs)
    paths, dt = model.generate_paths()  # paths: [N, D, T], dt: float
    paths = np.asarray(paths, dtype=np.float32)  # [N, D, T]
    x = torch.from_numpy(np.swapaxes(paths, 1, 2)).contiguous()  # -> [N, T, D]
    T = x.shape[1]
    # t = torch.linspace(0.0, T - 1, T, dtype=torch.float32) # previous implementation, actually wrong!
    t = torch.linspace(0.0, (T - 1) * float(dt), T, dtype=torch.float32)
    return x, t

def _gen_mixture_once(cfg: DictConfig, size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a mixture of two Brownian motions.

    cfg.mixture.p_bm1 gives the fraction of paths from BM1.
    The rest comes from BM2.

    Returns: x:[N, T, D] t:[T]
    """
    p_bm1 = float(cfg.mixture.p_bm1)

    if not (0.0 <= p_bm1 <= 1.0):
        raise ValueError("cfg.mixture.p_bm1 must be between 0 and 1.")

    n1 = int(round(size * p_bm1))
    n2 = size - n1

    # Copy cfg so we can override drift/volatility separately
    cfg1 = cfg.copy()
    cfg2 = cfg.copy()

    # BM1 params
    cfg1.drift = cfg.mixture.bm1.drift
    cfg1.volatility = cfg.mixture.bm1.volatility
    cfg1.S0 = cfg.mixture.bm1.S0
    # BM2 params
    cfg2.drift = cfg.mixture.bm2.drift
    cfg2.volatility = cfg.mixture.bm2.volatility
    cfg2.S0 = cfg.mixture.bm2.S0

    # Generate both groups with different deterministic seeds
    x1, t1 = _gen_single_once(cfg1, n1, seed + 11)
    x2, t2 = _gen_single_once(cfg2, n2, seed + 23)

    if not torch.allclose(t1, t2):
        raise ValueError("BM1 and BM2 produced different time grids.")

    # Concatenate: first BM1, then BM2
    x = torch.cat([x1, x2], dim=0)

    # Labels: 0 = BM1, 1 = BM2
    z = torch.cat([torch.zeros(n1, dtype=torch.long),torch.ones(n2, dtype=torch.long)], dim=0)

    # Shuffle so batches contain a random mix of BM1/BM2
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 37)
    perm = torch.randperm(size, generator=g).cpu()

    x = x[perm]
    z = z[perm]

    return x, t1, z


def _manifest_payload(cfg: DictConfig) -> Dict[str, Any]:
    gen = {
        "model_name": cfg.model_name,
        "drift": cfg.drift,
        "volatility": cfg.volatility,
        "mean": getattr(cfg, "mean", None),
        "speed": getattr(cfg, "speed", None),
        "correlation": getattr(cfg, "correlation", None),

        "n_series": cfg.n_series,
        "n_steps": cfg.n_steps,
        "size_train": cfg.size_train,
        "size_val": cfg.size_val,
        "size_test": cfg.size_test,

        "S0": cfg.S0,
        "maturity": cfg.maturity,
        "sine_coeff": getattr(cfg, "sine_coeff", None),
        "scheme": getattr(cfg, "scheme", "euler"),
        "return_vol": getattr(cfg, "return_vol", False),
        "v0": getattr(cfg, "v0", None),

        "T_sub": cfg.T_sub,
        "time_spacing": cfg.time_spacing,

        "mixture": getattr(cfg, "mixture", None),

        "seeds": {
            "train": cfg.seed_train,
            "val": cfg.seed_val,
            "test": cfg.seed_test,
        },
    }
    return {"hash": _stable_hash(gen), "gen": gen}


# ----------------------------
# Data Loading
# ----------------------------

class StockDataset(Dataset):
    def __init__(self, x: torch.Tensor, t: torch.Tensor, z: Optional[torch.Tensor] = None):
        self.x, self.t, self.z = x, t, z  # x:[N,T,D], t:[N,T], z:[]
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        out = {"x": self.x[i], "t": self.t[i]}
        if self.z is not None: out["z"] = self.z[i]
        return out

def make_loaders(data_cfg: DictConfig, train_cfg: DictConfig):
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
    if not dataset_exists(data_cfg):
        create_dataset(data_cfg)
        
    _seed_all(train_cfg.manual_seed)

    root = Path(to_absolute_path(data_cfg.save_dir))
    train_x = torch.load(root / "train_data.pt", map_location=train_cfg.device, weights_only=True)
    train_t = torch.load(root / "train_times.pt", map_location=train_cfg.device, weights_only=True)
    val_x   = torch.load(root / "val_data.pt",   map_location=train_cfg.device, weights_only=True)
    val_t   = torch.load(root / "val_times.pt",  map_location=train_cfg.device, weights_only=True)
    test_x  = torch.load(root / "test_data.pt",  map_location=train_cfg.device, weights_only=True)
    test_t  = torch.load(root / "test_times.pt", map_location=train_cfg.device, weights_only=True)

    train_z = torch.load(root / "train_labels.pt", map_location=train_cfg.device, weights_only=True)
    val_z   = torch.load(root / "val_labels.pt", map_location=train_cfg.device, weights_only=True)
    test_z  = torch.load(root / "test_labels.pt", map_location=train_cfg.device, weights_only=True)

    train = _subsample(train_x, train_t, data_cfg.T_sub, data_cfg.fix_min_max, train_cfg.manual_seed, data_cfg.time_spacing)
    val   = _subsample(val_x, val_t, data_cfg.T_sub, data_cfg.fix_min_max, train_cfg.manual_seed, data_cfg.time_spacing)
    
    train_ds = StockDataset(train["x"], train["t"], train_z)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    val_sub  = {"x": val["x"], "t": val["t"], "z": val_z}
    
    val_full = {"x": val_x, "t": val_t.unsqueeze(0).expand(val_x.shape[0], -1), "z": val_z}
    test_full = {"x": test_x, "t": test_t.unsqueeze(0).expand(test_x.shape[0], -1), "z": test_z}

    return train_loader, val_sub, val_full, test_full


def dataset_exists(cfg: DictConfig) -> bool:
    """
    Check if {train,val,test}_{data,times}.pt exists and manifest hash fits.
    """
    root = Path(to_absolute_path(cfg.save_dir))
    _, _, _, manifest_p = _store_paths(root, "train")

    desired = _manifest_payload(cfg)
    need_regen = True
    if manifest_p.exists():
        try:
            current = json.loads(manifest_p.read_text())
            if current.get("hash") == desired["hash"]:
                # verify files exist
                ok = True
                for split in ("train", "val", "test"):
                    p_data, p_times, p_labels, _ = _store_paths(root, split)
                    ok = ok and p_data.exists() and p_times.exists() and p_labels.exists()
                need_regen = not ok
            else:
                need_regen = True
        except Exception:
            need_regen = True

    return not need_regen


def _subsample(x: torch.Tensor, t: torch.Tensor, T_sub: int, fix_min_max: bool, seed: int, time_spacing: str) -> dict:
    """
    Per-trajectory subsampling.
    Inputs:
      x: [N, T, D], t: [T]
    Returns:
      {"x": [N, T_sub, D], "t": [N, T_sub], "idx": [N, T_sub]}
    """
    N, T, D = x.shape
    assert T_sub <= T
    _seed_all(seed)
    device = x.device

    if time_spacing not in {"regular", "irregular"}:
        raise ValueError("time_spacing must be either 'regular' or 'irregular'.")

    if time_spacing == "irregular":
        if fix_min_max:
            if T_sub < 2:
                raise ValueError("T_sub must be >= 2 when fix_min_max=True.")
            idx_interior = torch.rand(N, T - 2, device=device).argsort(dim=1)[:, : T_sub - 2] + 1  # [N, T_sub-2]
            idx, _ = torch.sort(idx_interior, dim=1)
            idx_min = torch.zeros(N, 1, dtype=torch.long, device=device)
            idx_max = torch.full((N, 1), fill_value=T-1, dtype=torch.long, device=device)
            idx = torch.cat((idx_min, idx, idx_max), dim=1)
        else:
            idx = torch.rand(N, T, device=device).argsort(dim=1)[:, :T_sub]  # [N, T_sub]
            idx, _ = torch.sort(idx, dim=1)
    else: 
        if fix_min_max:
            if T_sub < 2:
                raise ValueError("T_sub must be >= 2 when fix_min_max=True.")
            idx_1d = torch.linspace(0, T - 1, T_sub, device=device).round().long()
        else:
            idx_1d = ((torch.arange(T_sub, device=device, dtype=torch.float32) + 0.5)* T / T_sub- 0.5).round().long()
            idx_1d = idx_1d.clamp(0, T - 1)
        idx = idx_1d.unsqueeze(0).expand(N, T_sub)

    # Gather x per series
    row = torch.arange(N, device=device).unsqueeze(1).expand(N, T_sub)      # [N, T_sub]
    x_sub = x[row, idx, :]                                                  # [N, T_sub, D]

    # Gather t per series -> first expand t to [N, T], then gather
    t_sub = t.unsqueeze(0).expand(N, T).gather(1, idx)                   # [N, T_sub]

    return {"x": x_sub, "t": t_sub, "idx": idx}


# ----------------------------
# Tiny helpers
# ----------------------------
def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _stable_hash(d: Dict[str, Any]) -> str:
    # minimal, stable hash of generation-relevant config
    s = json.dumps(_to_plain(d), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()

def _to_plain(obj: Any) -> Any:
    """Recursively turn Hydra containers into plain Python types.
    Works even if `obj` is already plain dict/list."""
    if isinstance(obj, (DictConfig, ListConfig)):
        # This handles any nested DictConfig/ListConfig too
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]  # tuples → lists for JSON
    # Convert sets (if any) to sorted lists for stability
    if isinstance(obj, set):
        return sorted(_to_plain(v) for v in obj)
    return obj  # leave JSON-serializable scalars as-is

def _store_paths(root: Path, split: str) -> Tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / f"{split}_data.pt",
        root / f"{split}_times.pt",
        root / f"{split}_labels.pt",
        root / "manifest.json",
    )

def _model_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    # Only the kwargs the stock models expect; ignore the rest.
    return {
        "drift": np.asarray(_to_plain(cfg.drift), dtype=np.float32),
        "volatility": np.asarray(_to_plain(cfg.volatility), dtype=np.float32),
        "mean": getattr(cfg, "mean", None),
        "speed": getattr(cfg, "speed", None),
        "correlation": getattr(cfg, "correlation", None),
        "nb_paths": cfg.n_series,
        "nb_steps": cfg.n_steps,
        "S0": np.asarray(_to_plain(cfg.S0), dtype=np.float32),
        "maturity": cfg.maturity,
        "sine_coeff": getattr(cfg, "sine_coeff", None),
        "scheme": getattr(cfg, "scheme", "euler"),
        "return_vol": getattr(cfg, "return_vol", False),
        "v0": getattr(cfg, "v0", None),
    }