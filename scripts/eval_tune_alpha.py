# load trained sde and jump models (with saved state dicts) from disk 
# (on specified hyperparameter grid), combine into markov superposition model,
# generate paths and compare with eval set for different values of alpha.
# Summarized results (mean/std mmd/sinkhorn) and results for every seed saved 
# in two csv files ("summary" and "raw").

import _pathfix
import torch
from tgm.utils.eval_utils import evaluate_alpha_grid, score_fn


if __name__ == "__main__":
    parent_dir = "/home/datastore/pfohl/tgm_runs/runs_stock1d"
    dataset = "BlackScholes1d"

    # Two model classes to superpose
    model_a_name = "DriftDiffusionModel"
    model_b_name = "JumpModel"
    jump_api = "jump"

    sigma_list = [3, 1, 0.3, 0.1, 0.03, 0.01]
    rho_list = [0.001, 0.0001]
    t_sub_list = [50]
    seeds = [0, 1, 2, 3, 4]

    # Alpha sweep
    alphas = [i/10 for i in range(11)]          # 0.0, 0.1, ..., 1.0

    # Which checkpoint type for each side
    variant_a = "mmd"       # or "sinkhorn"
    variant_b = "mmd"       # or "sinkhorn"

    val_x   = torch.load("../data/stock/val_data.pt",   map_location="cuda", weights_only=True)
    val_t   = torch.load("../data/stock/val_times.pt",  map_location="cuda", weights_only=True)
    val_set = {"x": val_x, "t": val_t.unsqueeze(0).expand(val_x.shape[0], -1)}

    lr = 1e-5
    map_location = "cuda"
    output_dir = "/homes/numerik/pfohl/Projects/Trajectory Generator Matching/tgm/eval_output"

    raw_df, summary_df = evaluate_alpha_grid(
        parent_dir=parent_dir,
        dataset=dataset,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        sigma_list=sigma_list,
        rho_list=rho_list,
        t_sub_list=t_sub_list,
        seeds=seeds,
        alphas=alphas,
        val_set=val_set,
        variant_a=variant_a,
        variant_b=variant_b,
        lr=lr,
        map_location=map_location,
        output_dir=output_dir,
        score_fn=score_fn,
        jump_api = jump_api
    )
    print(summary_df.head())
