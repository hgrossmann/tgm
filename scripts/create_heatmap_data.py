import _pathfix
import torch
import pandas as pd
import os
import hydra
from omegaconf import DictConfig

from tgm.models.tfm_model import DriftDiffusionModel, NoiseModel, tnextModel


MODEL_REGISTRY = {
    "DriftDiffusionModel": DriftDiffusionModel,
    "NoiseModel": NoiseModel,
    "tnextModel": tnextModel,
}


def make_tx_grid(x_lower, x_upper, dt=0.01, dx=0.01, device="cpu"):
    # t: 0, dt, ..., 1-dt
    # t2: dt, 2dt, ..., 1
    t_vals = torch.arange(0.0, 1.0, dt, device=device)
    t2_vals = t_vals + dt

    # numerische Sicherheit: letzter Wert exakt höchstens 1
    t2_vals = torch.clamp(t2_vals, max=1.0)

    x_vals = torch.arange(x_lower, x_upper + 0.5 * dx, dx, device=device)

    tt, xx = torch.meshgrid(t_vals, x_vals, indexing="ij")
    tt2, _ = torch.meshgrid(t2_vals, x_vals, indexing="ij")

    t = tt.reshape(-1, 1)
    x = xx.reshape(-1, 1)
    t2 = tt2.reshape(-1, 1)

    return x, t, t2


@torch.no_grad()
def generate_heatmap_data(
    model,
    part,
    x_lower,
    x_upper,
    dt=0.01,
    dx=0.01,
    device="cpu",
    batch_size=10000,
):
    model.eval()
    model.to(device)

    if getattr(model, "memory_switch", "off") == "on":
        raise ValueError("Heatmap-Grid geht aktuell nicht für Modelle mit Memory.")

    if part not in model.trainable_parts:
        raise ValueError(
            f"part='{part}' ist kein trainable part. "
            f"Verfügbare trainable_parts: {model.trainable_parts}"
        )

    x_all, t_all, t2_all = make_tx_grid(
        x_lower=x_lower,
        x_upper=x_upper,
        dt=dt,
        dx=dx,
        device=device,
    )

    outputs = []

    for start in range(0, x_all.shape[0], batch_size):
        end = start + batch_size

        x = x_all[start:end]
        t = t_all[start:end]
        t2 = t2_all[start:end]

        single_vals = torch.cat([x, t, t2], dim=-1)

        if part == "drift":
            out = model.drift(single_vals)

        elif part == "uncertainty":
            out = model.noise(
                x=x,
                t=t,
                x_mem=None,
                t_mem=None,
                t2=t2,
            )

        else:
            raise ValueError(f"Unbekannter part: {part}")

        outputs.append(out.detach().cpu())

    y_all = torch.cat(outputs, dim=0)

    df = pd.DataFrame({
        "t": t_all.detach().cpu().squeeze().numpy(),
        "x": x_all.detach().cpu().squeeze().numpy(),
        part: y_all.squeeze().numpy(),
    })

    return df


@hydra.main(config_path="../conf", config_name="config_bm_1d", version_base=None)
def main(cfg: DictConfig):
    device = "cpu"

    ckpt = torch.load("../checkpoints/model.pt", map_location=device)

    model_class = MODEL_REGISTRY[cfg.model.model_name]
    model = model_class(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    parts = ["drift", "uncertainty"]

    x_lower = -2.5
    x_upper = 3.5
    dt = 0.01
    dx = 0.01

    os.makedirs("../output_traj", exist_ok=True)

    if "drift" in parts:
        df_drift = generate_heatmap_data(
            model=model,
            part="drift",
            x_lower=x_lower,
            x_upper=x_upper,
            dt=dt,
            dx=dx,
            device=device,
        )
        df_drift.to_csv("../output_traj/heatmap_data_drift.csv", index=False)

    if "uncertainty" in parts:
        df_unc = generate_heatmap_data(
            model=model,
            part="uncertainty",
            x_lower=x_lower,
            x_upper=x_upper,
            dt=dt,
            dx=dx,
            device=device,
        )
        df_unc.to_csv("../output_traj/heatmap_data_uncertainty.csv", index=False)


if __name__ == "__main__":
    main()