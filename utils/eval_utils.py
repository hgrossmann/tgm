# import _pathfix
import os
import glob
from typing import Any, Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from datetime import datetime

from tgm.eval.metrics import mmd_metric, sinkhorn_dist

# ---- Your model registry ----
from tgm.models.drift_diff_model import DriftDiffusionModel
from tgm.models.jump_model import JumpModel
from tgm.models.jump_model_uncoupled import JumpModelUncoupled
from tgm.models.jump_model_full_cov import JumpModelFullCov
from tgm.models.tfm_model import TfmModel
from tgm.models.superposition_model import MarkovSuperpositionModel

MODEL_REGISTRY = {
    "DriftDiffusionModel": DriftDiffusionModel,
    "JumpModel": JumpModel,
    "JumpModelUncoupled": JumpModelUncoupled,
    "JumpModelFullCov": JumpModelFullCov,
    "TfmModel": TfmModel
}

# ----------------- PATHS & CONFIG -----------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def default_output_dir(parent_dir: str, dataset: str) -> str:
    out = os.path.join(parent_dir, "eval_outputs", dataset)
    ensure_dir(out)
    return out

def resolve_experiment_folder(parent_dir: str, dataset: str, model: str,
                              sigma: Any, rho: Any, lr: Any, t_sub: Any, seed: int) -> str:
    """Find the most recent folder matching the parameter configuration and seed."""
    prefix = f"{dataset}_{model}_sig{sigma}_rho{rho}_lr{lr}_sub{t_sub}_{seed}"
    candidates = glob.glob(os.path.join(parent_dir, prefix + "*"))
    if not candidates:
        raise FileNotFoundError(f"No folder found for prefix {prefix}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def load_run_config(run_folder: str) -> DictConfig:
    """Load Hydra config from config_dump.yaml inside the run folder."""
    cfg_path = os.path.join(run_folder, "config_dump.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config_dump.yaml in {run_folder}")
    return OmegaConf.load(cfg_path)

def instantiate_model_from_cfg(model_name: str, model_cfg: DictConfig):
    """Build model instance using your registry and cfg.model."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}' in MODEL_REGISTRY.")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(model_cfg)

def _safe_load_state_dict(model: torch.nn.Module, ckpt_path: str, map_location: str = "cpu") -> bool:
    """Non-strict load to avoid crashing on small key diffs."""
    if not os.path.exists(ckpt_path):
        return False
    state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] {os.path.basename(ckpt_path)}: missing={len(missing)}, unexpected={len(unexpected)}")
    return True

# ----------------- METRIC AGGREGATION (Phase 1) -----------------

def get_min_metric_values(folder: str) -> Tuple[float, float]:
    """Load mmd_list.txt and sink_list.txt and return their minima."""
    mmd_path = os.path.join(folder, "mmd_list.txt")
    sink_path = os.path.join(folder, "sink_list.txt")
    mmd_vals = np.loadtxt(mmd_path)
    sink_vals = np.loadtxt(sink_path)
    return float(np.min(mmd_vals)), float(np.min(sink_vals))

def aggregate_metrics(parent_dir: str, dataset: str, model_list: List[str],
                      sigma_list: List[Any], rho_list: List[Any], t_sub_list: List[Any],
                      seeds: List[int], lr: Any = 1e-5, output_csv: Optional[str] = None) -> pd.DataFrame:
    """Aggregate metrics across models & seeds and save results to CSV."""
    results = []
    for model in model_list:
        for sigma in sigma_list:
            for rho in rho_list:
                for t_sub in t_sub_list:
                    mmd_mins, sink_mins = [], []
                    for seed in seeds:
                        try:
                            folder = resolve_experiment_folder(parent_dir, dataset, model, sigma, rho, lr, t_sub, seed)
                            min_mmd, min_sink = get_min_metric_values(folder)
                            mmd_mins.append(min_mmd)
                            sink_mins.append(min_sink)
                        except Exception as e:
                            print(f"Skipping (model={model}, sigma={sigma}, rho={rho}, t_sub={t_sub}, seed={seed}): {e}")
                    if mmd_mins and sink_mins:
                        results.append({
                            "dataset": dataset,
                            "model": model,
                            "sigma": sigma,
                            "rho": rho,
                            "t_sub": t_sub,
                            "mmd_mean": float(np.mean(mmd_mins)),
                            "mmd_std": float(np.std(mmd_mins)),
                            "sink_mean": float(np.mean(sink_mins)),
                            "sink_std": float(np.std(sink_mins)),
                            "n_seeds": int(len(mmd_mins))
                        })
    df = pd.DataFrame(results)
    if output_csv is None:
        output_csv = os.path.join(default_output_dir(parent_dir, dataset), "results.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved aggregation to {output_csv}")
    return df

# ----------------- TRAINED MODEL LOADING -----------------

def load_trained_models(parent_dir: str,
                        dataset: str,
                        model_name: str,
                        sigma: Any, rho: Any, t_sub: Any,
                        seeds: List[int],
                        lr: Any = 1e-3,
                        map_location: str = "cpu") -> Dict[int, Dict[str, torch.nn.Module]]:
    """
    For one (dataset, model_name, sigma, rho, t_sub), load MMD/Sinkhorn models per seed.
    Returns: {seed: {"mmd": model?, "sinkhorn": model?}}
    """
    out: Dict[int, Dict[str, torch.nn.Module]] = {}
    for seed in seeds:
        try:
            run_folder = resolve_experiment_folder(parent_dir, dataset, model_name, sigma, rho, lr, t_sub, seed)
            cfg = load_run_config(run_folder)
            model_cfg = cfg.model
        except Exception as e:
            print(f"[skip] ({model_name}, s={sigma}, r={rho}, sub={t_sub}, seed={seed}): {e}")
            continue

        models_for_seed: Dict[str, torch.nn.Module] = {}

        # MMD
        try:
            mmd_ckpt = os.path.join(run_folder, "best_state_dict_mmd.pt")
            mdl = instantiate_model_from_cfg(model_name, model_cfg)
            if _safe_load_state_dict(mdl, mmd_ckpt, map_location):
                models_for_seed["mmd"] = mdl
        except Exception as e:
            print(f"[warn] MMD load failed (seed={seed}): {e}")

        # Sinkhorn
        try:
            sink_ckpt = os.path.join(run_folder, "best_state_dict_sinkhorn.pt")
            mdl = instantiate_model_from_cfg(model_name, model_cfg)
            if _safe_load_state_dict(mdl, sink_ckpt, map_location):
                models_for_seed["sinkhorn"] = mdl
        except Exception as e:
            print(f"[warn] Sinkhorn load failed (seed={seed}): {e}")

        if models_for_seed:
            out[seed] = models_for_seed

    return out

def load_trained_models_grid(parent_dir: str,
                             dataset: str,
                             model_names: List[str],
                             sigma_list: List[Any],
                             rho_list: List[Any],
                             t_sub_list: List[Any],
                             seeds: List[int],
                             lr: Any = 1e-3,
                             map_location: str = "cpu") -> Dict[str, Dict[Tuple[Any, Any, Any], Dict[int, Dict[str, torch.nn.Module]]]]:
    """
    Iterate over (model_name, sigma, rho, t_sub) and load per-seed trained models.
    Returns:
      results[model_name][(sigma, rho, t_sub)][seed]["mmd"/"sinkhorn"] -> model
    """
    results: Dict[str, Dict[Tuple[Any, Any, Any], Dict[int, Dict[str, torch.nn.Module]]]] = {}
    for model_name in model_names:
        per_model: Dict[Tuple[Any, Any, Any], Dict[int, Dict[str, torch.nn.Module]]] = {}
        for sigma in sigma_list:
            for rho in rho_list:
                for t_sub in t_sub_list:
                    loaded = load_trained_models(parent_dir, dataset, model_name,
                                                 sigma, rho, t_sub, seeds, lr, map_location)
                    if loaded:
                        per_model[(sigma, rho, t_sub)] = loaded
        if per_model:
            results[model_name] = per_model
    return results

# ----------------- SIMPLE SCORING STUB -----------------

def test_score(test_set: Any, model: torch.nn.Module, **kwargs) -> float:
    """
    Stub: evaluate a single model on a dataset and return a scalar score.
    Replace with your real evaluation loop.
    """
    print(f"[stub] test_score: {model.__class__.__name__}, size={len(test_set) if hasattr(test_set,'__len__') else 'NA'}")
    return float("nan")

# ----------------- PHASE 2: Alpha tuning on VAL -----------------

def evaluate_alpha_grid(parent_dir: str,
                        dataset: str,
                        model_a_name: str,
                        model_b_name: str,
                        sigma_list: List[Any],
                        rho_list: List[Any],
                        t_sub_list: List[Any],
                        seeds: List[int],
                        alphas: List[float],
                        val_set: Any,
                        variant_a: str = "mmd",
                        variant_b: str = "mmd",
                        lr: Any = 1e-3,
                        map_location: str = "cpu",
                        output_dir: Optional[str] = None,
                        score_fn: Optional[Callable[[Any, torch.nn.Module], float]] = None,
                        jump_api: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep (sigma, rho, T_sub, seed, alpha) on validation set.
    Writes:
      - alpha_tuning_raw.csv    (one row per seed/alpha)
      - alpha_tuning_summary.csv (grouped mean/std over seeds per alpha)
    """
    if output_dir is None:
        output_dir = default_output_dir(parent_dir, dataset)
    ensure_dir(output_dir)

    if score_fn is None:
        raise ValueError("No score function.")

    total_triplets = len(sigma_list) * len(rho_list) * len(t_sub_list)
    total_alphas = len(alphas)
    print(f"[alpha grid] total triplets={total_triplets}, alphas per triplet={total_alphas}")
    triplet_idx = 0

    raw_rows = []
    for sigma in sigma_list:
        for rho in rho_list:
            for t_sub in t_sub_list:
                triplet_idx += 1
                print(f"[{triplet_idx}/{total_triplets}] σ={sigma}, ρ={rho}, T_sub={t_sub}")
                
                a_by_seed = load_trained_models(parent_dir, dataset, model_a_name, sigma, rho, t_sub, seeds, lr, map_location)
                b_by_seed = load_trained_models(parent_dir, dataset, model_b_name, sigma, rho, t_sub, seeds, lr, map_location)

                common = sorted(
                    s for s in set(a_by_seed.keys()).intersection(b_by_seed.keys())
                    if (variant_a in a_by_seed[s]) and (variant_b in b_by_seed[s])
                )
                if not common:
                    print(f"[info] No overlapping seeds for {model_a_name}/{model_b_name} at (sigma={sigma}, rho={rho}, sub={t_sub})")
                    continue

                for alpha in alphas:
                    for seed in common:
                        mdl_a = a_by_seed[seed][variant_a]
                        mdl_b = b_by_seed[seed][variant_b]
                        superposed = MarkovSuperpositionModel(mdl_a, mdl_b, alpha=alpha, jump_api=jump_api)
                        score = score_fn(
                            val_set, superposed,
                            split="val",
                            dataset=dataset,
                            model_a=model_a_name, model_b=model_b_name,
                            variant_a=variant_a, variant_b=variant_b,
                            sigma=sigma, rho=rho, t_sub=t_sub,
                            alpha=alpha, seed=seed,
                            output_dir=output_dir
                        )
                        raw_rows.append({
                            "dataset": dataset,
                            "model_a": model_a_name,
                            "model_b": model_b_name,
                            "variant_a": variant_a,
                            "variant_b": variant_b,
                            "sigma": sigma,
                            "rho": rho,
                            "t_sub": t_sub,
                            "alpha": float(alpha),
                            "seed": int(seed),
                            "score": float(score)
                        })

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = os.path.join(output_dir, "alpha_tuning_raw.csv")
    raw_df.to_csv(raw_csv, index=False)
    print(f"Saved alpha tuning raw to {raw_csv}")

    if raw_df.empty:
        return raw_df, raw_df

    group_cols = ["dataset", "model_a", "model_b", "variant_a", "variant_b", "sigma", "rho", "t_sub", "alpha"]
    summary = (raw_df
               .groupby(group_cols)["score"]
               .agg(score_mean="mean", score_std="std", n_seeds="count")
               .reset_index()
               .sort_values(["score_mean", "score_std", "n_seeds"], ascending=[True, True, False]))
    sum_csv = os.path.join(output_dir, "alpha_tuning_summary.csv")
    summary.to_csv(sum_csv, index=False)
    print(f"Saved alpha tuning summary to {sum_csv}")
    return raw_df, summary

def select_best_from_val(summary_csv: str) -> Dict[str, Any]:
    """Load alpha_tuning_summary.csv and return best row (lowest mean, tie by std then n_seeds)."""
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError("alpha_tuning_summary.csv is empty.")
    df_sorted = df.sort_values(["score_mean", "score_std", "n_seeds"], ascending=[True, True, False])
    best = df_sorted.iloc[0].to_dict()
    print("[best val] sigma={sigma}, rho={rho}, t_sub={t_sub}, alpha={alpha}, score={score_mean}±{score_std}, n={n_seeds}".format(**best))
    return best

# ----------------- PHASE 3: Final TEST evaluation -----------------

def evaluate_on_test(parent_dir: str,
                     dataset: str,
                     model_a_name: str,
                     model_b_name: str,
                     selections: Optional[List[Dict[str, Any]]],   # list of dicts or None to auto-pick
                     test_set: Any,
                     seeds: List[int],
                     lr: Any = 1e-3,
                     map_location: str = "cpu",
                     output_dir: Optional[str] = None,
                     score_fn: Optional[Callable[[Any, torch.nn.Module], float]] = None,
                     summary_name: str = "test_results_summary.csv",
                     raw_name: str = "test_results_raw.csv",
                     jump_api: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate chosen (sigma, rho, T_sub, alpha) on the TEST set.
    If selections is None, auto-pick best from alpha_tuning_summary.csv in output_dir.
    Each selection dict must include: variant_a, variant_b, sigma, rho, t_sub, alpha.
    Writes raw per-seed rows and grouped summary.
    """
    if output_dir is None:
        output_dir = default_output_dir(parent_dir, dataset)
    ensure_dir(output_dir)

    if selections is None:
        summary_csv = os.path.join(output_dir, "alpha_tuning_summary.csv")
        sel = select_best_from_val(summary_csv)
        selections = [sel]

    if score_fn is None:
        raise ValueError("No score function defined.")

    raw_rows = []
    for sel in selections:
        sigma = sel["sigma"]; rho = sel["rho"]; t_sub = sel["t_sub"]; alpha = sel["alpha"]
        variant_a = sel.get("variant_a", "mmd"); variant_b = sel.get("variant_b", "mmd")

        a_by_seed = load_trained_models(parent_dir, dataset, model_a_name, sigma, rho, t_sub, seeds, lr, map_location)
        b_by_seed = load_trained_models(parent_dir, dataset, model_b_name, sigma, rho, t_sub, seeds, lr, map_location)

        common = sorted(
            s for s in set(a_by_seed.keys()).intersection(b_by_seed.keys())
            if (variant_a in a_by_seed[s]) and (variant_b in b_by_seed[s])
        )
        if not common:
            print(f"[info] No overlapping seeds on TEST for (sigma={sigma}, rho={rho}, sub={t_sub})")
            continue

        for seed in common:
            mdl_a = a_by_seed[seed][variant_a]
            mdl_b = b_by_seed[seed][variant_b]
            superposed = MarkovSuperpositionModel(mdl_a, mdl_b, alpha=alpha, jump_api=jump_api)
            score = score_fn(
                test_set, superposed,
                split="test",
                dataset=dataset,
                model_a=model_a_name, model_b=model_b_name,
                variant_a=variant_a, variant_b=variant_b,
                sigma=sigma, rho=rho, t_sub=t_sub,
                alpha=alpha, seed=seed,
                output_dir=output_dir
            )
            raw_rows.append({
                "dataset": dataset,
                "model_a": model_a_name,
                "model_b": model_b_name,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "sigma": sigma,
                "rho": rho,
                "t_sub": t_sub,
                "alpha": float(alpha),
                "seed": int(seed),
                "score": float(score)
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = os.path.join(output_dir, raw_name)
    raw_df.to_csv(raw_csv, index=False)
    print(f"Saved TEST raw to {raw_csv}")

    if raw_df.empty:
        return raw_df, raw_df

    group_cols = ["dataset", "model_a", "model_b", "variant_a", "variant_b", "sigma", "rho", "t_sub", "alpha"]
    summary = (raw_df
               .groupby(group_cols)["score"]
               .agg(score_mean="mean", score_std="std", n_seeds="count")
               .reset_index()
               .sort_values(["score_mean", "score_std", "n_seeds"], ascending=[True, True, False]))
    sum_csv = os.path.join(output_dir, summary_name)
    summary.to_csv(sum_csv, index=False)
    print(f"Saved TEST summary to {sum_csv}")
    return raw_df, summary

def evaluate_single_on_test(parent_dir: str,
                            dataset: str,
                            model_name: str,
                            selections: List[Dict[str, Any]],   # each: {"variant": "mmd"/"sinkhorn", "sigma": ..., "rho": ..., "t_sub": ...}
                            test_set: Any,
                            seeds: List[int],
                            lr: Any = 1e-3,
                            map_location: str = "cpu",
                            output_dir: Optional[str] = None,
                            score_fn: Optional[Callable[[Any, torch.nn.Module], float]] = None,
                            summary_name: str = "single_test_results_summary.csv",
                            raw_name: str = "single_test_results_raw.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate ONE trained model class (no superposition) on TEST for given selections.
    Each selection dict must include: {"variant": "mmd" or "sinkhorn", "sigma": ..., "rho": ..., "t_sub": ...}
    Writes raw per-seed rows and grouped summary.
    """
    if output_dir is None:
        output_dir = default_output_dir(parent_dir, dataset)
    ensure_dir(output_dir)

    if score_fn is None:
        raise ValueError("No score function defined.")

    raw_rows = []
    for sel in selections:
        variant = sel["variant"]
        sigma = sel["sigma"]
        rho = sel["rho"]
        t_sub = sel["t_sub"]

        by_seed = load_trained_models(parent_dir, dataset, model_name, sigma, rho, t_sub, seeds, lr, map_location)
        available = sorted(s for s in by_seed.keys() if variant in by_seed[s])
        if not available:
            print(f"[info] No seeds on TEST for single model {model_name} at (sigma={sigma}, rho={rho}, sub={t_sub})")
            continue

        for seed in available:
            mdl = by_seed[seed][variant]
            score = score_fn(
                test_set, mdl,
                split="test",
                dataset=dataset,
                model_a=model_name,
                model_b=None,              # <- single-model
                variant_a=variant,
                variant_b=None,            # <- single-model
                sigma=sigma, rho=rho, t_sub=t_sub,
                alpha=None,                # <- single-model
                seed=seed,
                output_dir=output_dir
            )
            raw_rows.append({
                "dataset": dataset,
                "model_a": model_name,
                "model_b": "",             # keep CSV consistent
                "variant_a": variant,
                "variant_b": "",
                "sigma": sigma,
                "rho": rho,
                "t_sub": t_sub,
                "alpha": "",               # no alpha in single-model
                "seed": int(seed),
                "score": float(score)
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = os.path.join(output_dir, raw_name)
    raw_df.to_csv(raw_csv, index=False)
    print(f"Saved single-model TEST raw to {raw_csv}")

    if raw_df.empty:
        return raw_df, raw_df

    group_cols = ["dataset", "model_a", "variant_a", "sigma", "rho", "t_sub"]
    summary = (raw_df
               .groupby(group_cols)["score"]
               .agg(score_mean="mean", score_std="std", n_seeds="count")
               .reset_index()
               .sort_values(["score_mean", "score_std", "n_seeds"], ascending=[True, True, False]))
    sum_csv = os.path.join(output_dir, summary_name)
    summary.to_csv(sum_csv, index=False)
    print(f"Saved single-model TEST summary to {sum_csv}")
    return raw_df, summary


@torch.inference_mode()
def score_fn(data, model, **ctx):
    split     = ctx["split"]          # "val" or "test"
    t_sub     = ctx["t_sub"]
    sigma     = ctx["sigma"]
    rho       = ctx["rho"]
    alpha     = ctx["alpha"]
    seed      = ctx["seed"]
    dataset_name = ctx["dataset"]
    model_a_name = ctx["model_a"]
    model_b_name = ctx.get("model_b", None)
    variant_a = ctx["variant_a"]
    variant_b = ctx.get("model_b", None)

    _seed_all(seed)

    data_x = data["x"] # no_samples, traj_length, datadim
    data_t = data["t"] # no_samples, traj_length
    
    # Build your sampling hyperparams (depends on T_sub etc.)
    if t_sub != 101:
        no_bridges = t_sub 
    else:
        no_bridges = 100
        
    x0 = data_x[0, 0, :]
    t_start = data_t[0, 0].item()
    t_end = data_t[0, -1].item()
    no_samples = data_x.shape[0]
    stepsize = 0.001 # hardcoded, who cares anymore
    
    traj, times, _ = model.sample_unif(x0, no_bridges, t_start, t_end, stepsize, no_samples)

    # compute score from traj 
    sample_trajectory_length = traj.shape[1]
    test_trajectory_length = data_x.shape[1]
    assert sample_trajectory_length == round((t_end - t_start) / stepsize + 1)
    equidistant_steps = test_trajectory_length - 1
    assert (sample_trajectory_length - 1) % equidistant_steps == 0 # not ideal check
    stepsize_subsampling = int((sample_trajectory_length - 1) / equidistant_steps)
    
    samples_val_grid = traj[:, torch.arange(0, sample_trajectory_length, stepsize_subsampling), :] #potentially just interpolate if grid cannot be aligned
    times_val_grid = times[:, torch.arange(0, sample_trajectory_length, stepsize_subsampling)]
    distance = mmd_metric if variant_a=="mmd" else sinkhorn_dist
    score = distance(samples_val_grid, data_x)

    # save figure of trajectories    
    fig = _trajectory_plot_all_dims(samples_val_grid, times_val_grid)
    save_dir = "./test_plots"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_b = model_b_name if model_b_name not in (None, "") else "single"
    safe_alpha = alpha if alpha is not None else "NA"
    short_id = f"{dataset_name}_{model_a_name}_{safe_model_b}_sig{sigma}_rho{rho}_a{safe_alpha}_sub{t_sub}_{seed}_{timestamp}_{variant_a}"
    output_path = os.path.join(save_dir, short_id)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, f"trajectories_{split}_generated.png"), dpi=300)
    plt.close()

    return score

def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def _trajectory_plot_all_dims(trajectories, times, no_traj_to_plot = 100):
    times_cpu = times.detach().cpu().numpy()
    trajectories_cpu = trajectories.detach().cpu().numpy()
    
    dims = trajectories_cpu.shape[-1]
    fig, axes = plt.subplots(1, dims, figsize=(8*dims, 5))
    axes = [axes] if dims == 1 else axes
    for i in range(no_traj_to_plot):
        for j in range(dims):
            axes[j].plot(times_cpu[i,:], trajectories_cpu[i,:,j])

    return fig
