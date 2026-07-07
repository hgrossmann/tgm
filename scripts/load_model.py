import _pathfix
import torch
import pandas as pd
import os
import hydra
from omegaconf import DictConfig

from tgm.models.tfm_model import DriftDiffusionModel, NoiseModel, tnextModel
from tgm.utils.memory import get_memory

MODEL_REGISTRY = {
    "DriftDiffusionModel": DriftDiffusionModel,
    "NoiseModel": NoiseModel,
    "tnextModel": tnextModel
}

#@hydra.main(config_path="../conf", config_name="config_bm_1d", version_base=None)
@hydra.main(config_path="../conf", config_name="config_stock_1d", version_base=None)

def main(cfg: DictConfig):   
    ckpt = torch.load("../checkpoints/model.pt", map_location="cpu")
    model_class = MODEL_REGISTRY[cfg.model.model_name]
    model = model_class(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # wichtig: Inferenzmodus (Dropout/BatchNorm korrekt)

    device = "cpu"
    model.to(device)

    n = 500
    n_steps = 100
    dt = 0.01#n_steps * dt muss 1 ergeben!

    use_true_history = False   # <- Switch: False = alles wie bisher, True = erst echte BM-Schritte
    n_true_history_steps = 5  # <- händisch wählen: wie viele erste Schritte echt generiert werden

    # Startpunkte bauen
    if cfg.data.mixture.enabled:
        n_bm1 = int(round(cfg.data.mixture.p_bm1 * n))
        n_bm2 = n - n_bm1

        x0_bm1 = torch.full((n_bm1, 1), cfg.data.mixture.bm1.S0)
        x0_bm2 = torch.full((n_bm2, 1), cfg.data.mixture.bm2.S0)
        x0 = torch.cat([x0_bm1, x0_bm2], dim=0)

        # merkt sich: 0 = bm1, 1 = bm2
        component_id = torch.cat([torch.zeros(n_bm1, dtype=torch.long),torch.ones(n_bm2, dtype=torch.long)], dim=0)
    else:
        x0 = torch.full((n, 1), cfg.data.S0)
        # nur eine Komponente
        component_id = torch.zeros(n, dtype=torch.long)

    x0 = x0.to(device)
    component_id = component_id.to(device)

    # ------------------------------------------------------------
    # Optional: echte Brownian-Historie erzeugen
    # ------------------------------------------------------------
    history_x = None
    history_t = None

    if (use_true_history and model.memory_switch == "on"):
        # [N, n_true_history_steps+1, 1]
        history_x = torch.zeros((n, n_true_history_steps + 1, 1), device=device)
        history_t = torch.zeros((n, n_true_history_steps + 1, 1), device=device)

        # Startwerte
        history_x[:, 0, :] = x0
        assert torch.allclose(history_x[:, 0, :], x0)

        # echte BM-Schritte
        for k in range(n_true_history_steps):
            current_x = history_x[:, k, :]
            t_k = k * dt

            # bm1 Parameter
            mask_bm1 = (component_id == 0)
            if mask_bm1.any():
                mu1 = cfg.data.mixture.bm1.drift
                sigma1 = cfg.data.mixture.bm1.volatility
                noise = torch.randn_like(current_x[mask_bm1])

                next_x = (current_x[mask_bm1]+ mu1 * dt + sigma1 * torch.sqrt(torch.tensor(dt)) * noise)
                history_x[mask_bm1, k + 1, :] = next_x

            # bm2 Parameter
            mask_bm2 = (component_id == 1)
            if mask_bm2.any():
                mu2 = cfg.data.mixture.bm2.drift
                sigma2 = cfg.data.mixture.bm2.volatility
                noise = torch.randn_like(current_x[mask_bm2])

                next_x = (current_x[mask_bm2] + mu2 * dt + sigma2 * torch.sqrt(torch.tensor(dt)) * noise)
                history_x[mask_bm2, k + 1, :] = next_x

            history_t[:, k + 1, :] = (k + 1) * dt

    traj = generate_trajectory(model, x0, n_steps, dt, device=device, history_x=history_x, history_t=history_t, T_sub=cfg.data.T_sub)

    print("x0 unique:", torch.unique(x0[:, 0]))
    print("traj[0] unique:", torch.unique(traj[0, :, 0]))
    print("traj shape:", traj.shape)

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
def generate_trajectory(model, x0, n_steps, dt, device="cpu", history_x=None, history_t=None, T_sub=None):
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
    if history_x is not None and history_t is not None:
        history_x = history_x.to(device)
        history_t = history_t.to(device)

        # Output MUSS mit t=0 starten
        traj_out = [history_x[:, j, :].cpu() for j in range(history_x.shape[1])]

        # Memory enthält komplette echte History
        x_hist = [history_x[:, j, :] for j in range(history_x.shape[1])]
        t_hist = [history_t[:, j, :] for j in range(history_t.shape[1])]

        # Netz startet beim letzten echten Punkt
        x = history_x[:, -1, :]

        # wenn history_x Länge 11 hat, sind 10 Schritte schon erledigt
        start_step = history_x.shape[1] - 1

    else:
        x_hist = [x]  # each: [B, D]
        t0 = torch.zeros((B, 1), dtype=torch.float32, device=device)
        t_hist = [t0]  # each: [B, 1]

        start_step = 0
        traj_out = [x.cpu()]

    for k in range(start_step, n_steps):
        t  = torch.full((B, 1), k * dt,       dtype=torch.float32, device=device)
        t2 = torch.full((B, 1), (k + 1) * dt, dtype=torch.float32, device=device)
        
        # Build sequence tensors expected by get_memory:
        data_seq_full = torch.stack(x_hist, dim=1)   # alle generierten Punkte, Abstand dt
        times_seq_full = torch.cat(t_hist, dim=1)

        # feste Trainingszeitpunkte auf Generierungsgrid
        train_idx = torch.round(torch.linspace(0, n_steps, T_sub, device=device)).long()

        # nur Trainingsgrid-Punkte strikt vor aktuellem k
        valid_idx = train_idx[train_idx < k]

        # Sonderfall am Anfang: wenn noch kein Gridpunkt < k existiert, nimm x0
        if len(valid_idx) == 0:
            valid_idx = train_idx[:1]   # also [0]

        data_seq = data_seq_full[:, valid_idx, :]
        times_seq = times_seq_full[:, valid_idx]

        idx_last = torch.full((B,),data_seq.shape[1] - 1,dtype=torch.long,device=device)

        x_mem, t_mem = get_memory(data_seq,times_seq,idx_last,model.memory_length,device=device)
        single_vals = torch.cat([x, t, t2], dim=-1)

        print(f"k={k}: t_mem[1] = {t_mem[1, :, 0].cpu().numpy()}")
        print(f"k={k}: x_mem[1] = {x_mem[1, :, 0].cpu().numpy()}")

        if model.memory_switch == "on":
            memory_flat = torch.cat([x_mem, t_mem], dim=-1).reshape(B, -1)
            inpu = torch.cat([single_vals, memory_flat], dim=-1)
        else:
            inpu = torch.cat([single_vals], dim=-1)

        # pred_x = model.drift(inpu)
        # mu = (pred_x-x)/(t2-t)
        # sigma = model.noise(x, t, x_mem, t_mem, t2)
        if "drift" in model.trainable_parts:
            mu = model.drift(inpu)
        else:
            #mu = torch.full_like(x, model.mu)
            mu = torch.full_like(x, model.mu)*x
        if "uncertainty" in model.trainable_parts:
            sigma = model.noise(x, t, x_mem, t_mem, t2)
        else:
            #sigma = torch.full_like(x, model.sigma_gt**2)
            sigma = torch.full_like(x, model.sigma_gt**2)*x**2
            
        x = x + mu * dt_t + torch.sqrt(sigma) * torch.sqrt(dt_t) * torch.randn_like(x)

        # Append to histories (for next iteration)
        x_hist.append(x)
        t_hist.append(t2)

        traj_out.append(x.cpu())

    return torch.stack(traj_out, dim=0)  # [T, B, D]

if __name__ == "__main__":
    main()
    