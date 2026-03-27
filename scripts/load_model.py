import _pathfix
import torch
import pandas as pd
import os
import hydra
from omegaconf import DictConfig

from tgm.models.tfm_paper_model import DriftDiffusionModel, NoiseModel, tnextModel
from tgm.utils.memory import get_memory

MODEL_REGISTRY = {
    "DriftDiffusionModel": DriftDiffusionModel,
    "NoiseModel": NoiseModel,
    "tnextModel": tnextModel
}

@hydra.main(config_path="../conf", config_name="config_bm_1d", version_base=None)

def main(cfg: DictConfig):   
    ckpt = torch.load("../checkpoints/model.pt", map_location="cpu")
    model_class = MODEL_REGISTRY[cfg.model.model_name]
    model = model_class(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # wichtig: Inferenzmodus (Dropout/BatchNorm korrekt)

    device = "cpu"
    model.to(device)

    n = 500
    x0 = torch.ones(n, 1)  # Startpunkte bei 0

    traj = generate_trajectory(model, x0, 100, 0.01)

    os.makedirs("../output_traj", exist_ok=True)
    traj_np = traj.squeeze(-1).numpy()   # entfernt letzte Dimension
    T = traj_np.shape[0]
    dt = 0.01
    times = (torch.arange(T) * dt).numpy()    # [T]
    columns = [f"traj_{i}" for i in range(n)]
    df = pd.DataFrame(traj_np, columns=columns)
    df.insert(0, "t", times) # <-- t als erste Spalte
    df.to_csv("../output_traj/generated_traj.csv", index=False)

@torch.no_grad()
def generate_trajectory(model, x0, n_steps, dt, device="cpu"):
    model.eval()
    model.to(device)

    if isinstance(x0, torch.Tensor):
        x = x0.to(device)
    else:
        x = torch.tensor(x0, dtype=torch.float32, device=device)
    # falls 1D gegeben wurde:
    if x.dim() == 1:
        x = x.unsqueeze(1)

    B, D = x.shape
    dt_t = torch.tensor(dt, dtype=torch.float32, device=device)

    # history lists
    x_hist = [x]  # each: [B, D]
    t0 = torch.zeros((B, 1), dtype=torch.float32, device=device)
    t_hist = [t0]  # each: [B, 1]

    traj_out = [x.cpu()]

    for k in range(n_steps):
        t  = torch.full((B, 1), k * dt,      dtype=torch.float32, device=device)  # [B,1]
        t2 = torch.full((B, 1), (k + 1) * dt, dtype=torch.float32, device=device) # [B,1]
        
        # Build sequence tensors expected by get_memory:
        data_seq = torch.stack(x_hist, dim=1)  # [B, L, D]
        times_seq = torch.cat(t_hist, dim=1)    # [B, L]

        idx_last = torch.full((B,), k, dtype=torch.long, device=device)  # [B]
        x_mem, t_mem = get_memory(data_seq, times_seq, idx_last, model.memory_length, device=device)   
        
        single_vals = torch.cat([x, t, t2], dim=-1)

        if model.memory_switch == "on":
            memory_flat = torch.cat([x_mem, t_mem], dim=-1).reshape(B, -1)
            inpu = torch.cat([single_vals, memory_flat], dim=-1)
        else:
            inpu = torch.cat([single_vals], dim=-1)

        pred_x = model.drift(inpu)
        mu = (pred_x-x)/(t2-t)
        sigma = model.noise(x, t, x_mem, t_mem, t2)

        x = x + mu * dt_t + torch.sqrt(sigma) * torch.sqrt(dt_t) * torch.randn_like(x)

        # Append to histories (for next iteration)
        x_hist.append(x)
        t_hist.append(t2)

        traj_out.append(x.cpu())

    return torch.stack(traj_out, dim=0)  # [T, B, D]

if __name__ == "__main__":
    main()
    