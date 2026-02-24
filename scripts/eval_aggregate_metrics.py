# script to aggregate metrics (mean and std of mmd/sinkhorn) for a chosen grid
# of hyperparameters from the output saved during these runs (i.e. the script
# assumes all runs have already been executed, does not start them itself - on
# a single machine that would take forever for every reasonably large grid)

# How to use: fill your paths and grids, then run this script.

import _pathfix
from tgm.utils.eval_utils import aggregate_metrics

if __name__ == "__main__":
    # parent_dir = "/home/datastore/pfohl/tgm_runs/runs_stock1d"
    # dataset = "BlackScholes1d"
    # model_list = ["JumpModel", "DriftDiffusionModel"]
    # sigma_list = [10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003]
    # rho_list = [0.001, 0.0001]
    # t_sub_list = [10, 25, 101]
    # seeds = [0, 1, 2, 3, 4]
    # lr = 1e-5
    # output_csv = "/homes/numerik/pfohl/Projects/Trajectory Generator Matching/tgm/eval_output/results_tgm.csv"

    # df = aggregate_metrics(parent_dir, dataset, model_list, sigma_list, rho_list, t_sub_list, seeds, lr, output_csv)
    # print(df.head())
    
    parent_dir = "/home/datastore/pfohl/tgm_runs/runs_stock2d"
    dataset = "BlackScholes2d"
    model_list = ["JumpModel", "DriftDiffusionModel", "JumpModelFullCov", "JumpModelUncoupled", "TfmModel"]
    sigma_list = [10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003]
    rho_list = [0.001, 0.0001]
    t_sub_list = [10, 25, 50, 101]
    seeds = [0, 1, 2, 3, 4]
    lr = 1e-5
    output_csv = "/homes/numerik/pfohl/Projects/Trajectory Generator Matching/tgm/eval_output/results_2d.csv"

    df = aggregate_metrics(parent_dir, dataset, model_list, sigma_list, rho_list, t_sub_list, seeds, lr, output_csv)
    print(df.head())
    