# load selected models (with saved state dicts) from disk, generate paths and
# compare with specified test set. Summarized results (mean/std mmd/sinkhorn)
# and results for every seed saved in two csv files ("summary" and "raw").

# version for evaluation of markov superposition model

import _pathfix
import torch
from tgm.utils.eval_utils import score_fn, evaluate_on_test

if __name__ == "__main__":
    
    # load test set
    test_x   = torch.load("../data/stock/test_data.pt",   map_location="cuda", weights_only=True)
    test_t   = torch.load("../data/stock/test_times.pt",  map_location="cuda", weights_only=True)
    test_set = {"x": test_x, "t": test_t.unsqueeze(0).expand(test_x.shape[0], -1)}
    
    # specify models to load
    parent_dir = "/home/datastore/pfohl/tgm_runs/runs_stock1d" # source folder
    dataset = "BlackScholes1d"
    model_a_name = "DriftDiffusionModel"
    model_b_name = "JumpModel"
    jump_api = "jump" # needs to correspond to chosen jump model, see forward method of class superposition model

    selections = [
        {"variant_a": "mmd", "variant_b": "mmd", "sigma": 0.03, "rho": 0.001, "t_sub": 101, "alpha": 0.9},
        {"variant_a": "mmd", "variant_b": "mmd", "sigma": 0.03, "rho": 0.0001, "t_sub": 50, "alpha": 0.7},
        {"variant_a": "mmd", "variant_b": "mmd", "sigma": 0.1, "rho": 0.001, "t_sub": 25, "alpha": 0.8},
        {"variant_a": "mmd", "variant_b": "mmd", "sigma": 0.3, "rho": 0.001, "t_sub": 10, "alpha": 0.9}
    ]
    
    # Seeds to evaluate (intersection of available seeds on both sides is used)
    seeds = [0, 1, 2, 3, 4]

    raw_df, summary_df = evaluate_on_test(
        parent_dir=parent_dir,
        dataset=dataset,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        selections=selections,
        test_set=test_set,
        seeds=seeds,
        lr=1e-5,
        map_location="cuda",
        output_dir="/homes/numerik/pfohl/Projects/Trajectory Generator Matching/tgm/eval_output",
        score_fn=score_fn,
        summary_name="test_results_MS_50_summary.csv",
        raw_name="test_results_MS_50_raw.csv",
        jump_api = jump_api
    )
    print(summary_df.head())