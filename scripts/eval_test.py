# load selected models (with saved state dicts) from disk, generate paths and
# compare with specified test set. Summarized results (mean/std mmd/sinkhorn)
# and results for every seed saved in two csv files ("summary" and "raw").

import _pathfix
import torch
from tgm.utils.eval_utils import score_fn, evaluate_single_on_test


if __name__ == "__main__":
    
    # load test set
    test_x   = torch.load("../data/stock2d/test_data.pt",   map_location="cuda", weights_only=True)
    test_t   = torch.load("../data/stock2d/test_times.pt",  map_location="cuda", weights_only=True)
    test_set = {"x": test_x, "t": test_t.unsqueeze(0).expand(test_x.shape[0], -1)}

    # specify models to load
    parent_dir = "/home/datastore/pfohl/tgm_runs/runs_stock2d" # source folder
    dataset = "BlackScholes2d" # required for full source path
    model_name = "DriftDiffusionModel"
    selections = [
        {"variant": "mmd", "sigma": 0.3, "rho": 0.0001, "t_sub": 25},
        {"variant": "mmd", "sigma": 0.3, "rho": 0.0001, "t_sub": 10}
    ]
    
    # specify output names and location
    output_dir = "/homes/numerik/pfohl/Projects/Trajectory Generator Matching/tgm/eval_output"
    summary_name = "test_results_sde_summary.csv" 
    raw_name = "test_results_sde_raw.csv"

    # Seeds to evaluate (intersection of available seeds on both sides is used)
    seeds = [0, 1, 2, 3, 4]
    lr = 1e-5
    
    raw_df, summary_df = evaluate_single_on_test(
        parent_dir=parent_dir,
        dataset=dataset,
        model_name=model_name,
        selections=selections,
        test_set=test_set,
        seeds=seeds,
        lr=lr,
        score_fn=score_fn,
        output_dir=output_dir,
        summary_name=summary_name,
        raw_name=raw_name
    )
