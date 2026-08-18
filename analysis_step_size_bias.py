"""
2D step-size bias vs. n_cells, aggregated over multiple independent runs.

Bias definition, reused as-is from comp1_addpositions_hilbert.py:

    bias_cm = mean(decoded L2 step) - mean(true L2 step)

"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sim_params_seq import params_id, n_reps, n_mincells, n_windows, mean_peak_Hz, n_osc_cycles, gamma_mod_depth, x_speed_m_per_sec

GAMMA_MOD_DEPTH = gamma_mod_depth
MODES = ["smooth"]
DEFAULT_N_CELLS = [100, 200, 400, 600, 800, 1000, 1400, 2000]
SUMMARY_DIR = "bias_summaries"


def _base_name(mode, n_cells):
    """
    File stem shared by all runs of a (mode, n_cells)

    Must match `file_name` built in sim_sequence_percell_rates.py exactly.
    """
    tag = "smooth" if mode == "smooth" else "speedmod1.0"
    return (f"percellrates_sim_seq_mode-{tag}_paramsid_{params_id}"
            f"_{n_reps}reps_{int(n_mincells)}mincells_{n_cells}cells_{n_windows}steps"
            f"_{GAMMA_MOD_DEPTH}gammamod_{mean_peak_Hz}meanHz")


_RUN_RE = re.compile(r"_run(\d+)_npformat\.npy$")


def find_run_files(mode, n_cells, restrict_runs=None):
    """All per-run npformat files for this (mode, n_cells), sorted by run index."""
    base = _base_name(mode, n_cells)
    paths = glob.glob(f"data/{base}_run*_npformat.npy")

    def run_idx(p):
        m = _RUN_RE.search(p)
        return int(m.group(1)) if m else -1

    paths = sorted(paths, key=run_idx)
    if restrict_runs is not None:
        paths = [p for p in paths if run_idx(p) in restrict_runs]
    return paths


def load_file(path):
    x_true, y_true, x_mean, y_mean = np.load(path, allow_pickle=True)[:4]
    return x_true, y_true, x_mean, y_mean


def load_step_spikes(npformat_path):
    sidecar = npformat_path.replace("_npformat.npy", "_stepdata.npy")
    if not os.path.exists(sidecar):
        return None
    step_spikes = np.load(sidecar, allow_pickle=True)[0]
    return step_spikes


def step_size_bias(x_true, y_true, x_mean, y_mean):
    """Returns (true_mean, decoded_mean, bias_cm, bias_pct, decoded_sem_cm)."""
    true_step = np.sqrt(np.diff(x_true, axis=1) ** 2 + np.diff(y_true, axis=1) ** 2)
    decoded_step = np.sqrt(np.diff(x_mean, axis=1) ** 2 + np.diff(y_mean, axis=1) ** 2)

    true_mean = np.nanmean(true_step)
    decoded_mean = np.nanmean(decoded_step)
    n_finite = np.sum(np.isfinite(decoded_step))
    decoded_sem = np.nanstd(decoded_step) / np.sqrt(n_finite)

    bias_cm = decoded_mean - true_mean
    bias_pct = 100 * bias_cm / true_mean
    return true_mean, decoded_mean, bias_cm, bias_pct, decoded_sem


def replay_speed_mps():
    return float(np.sqrt(2) * x_speed_m_per_sec)


def save_results(results, out_dir=SUMMARY_DIR):
    speed_mps = replay_speed_mps()
    os.makedirs(out_dir, exist_ok=True)

    payload = {"speed_mps": speed_mps, "params_id": params_id}
    for mode, r in results.items():
        for key in ("n_cells", "mean_pct", "std_pct", "sem_pct", "n_runs", "mean_spikes"):
            payload[f"{mode}__{key}"] = np.asarray(r[key])

    out_path = os.path.join(out_dir, f"stepsize_bias_id{params_id}_speed{speed_mps:.2f}mps.npz")
    np.savez(out_path, **payload)
    print(f"saved summary -> {out_path}")
    return out_path


def run(n_cells_list, restrict_runs=None):
    """
    {mode: {'n_cells', 'mean_pct', 'std_pct', 'sem_pct', 'n_runs'} lists}.

    mean/std/sem are taken OVER the per-run bias values (one bias per run file).
    """
    results = {mode: {"n_cells": [], "mean_pct": [], "std_pct": [], "sem_pct": [], "n_runs": [],
                       "mean_spikes": []}
               for mode in MODES}

    for mode in MODES:
        print(f"\nmode={mode}")
        print(f"{'n_cells':>8s} {'n_runs':>7s} {'mean[%]':>10s} {'std[%]':>9s} {'sem[%]':>9s} "
              f"{'spk/step':>9s}")
        for n_cells in n_cells_list:
            paths = find_run_files(mode, n_cells, restrict_runs)
            if not paths:
                print(f"{n_cells:8d}  -- no _run*.npy files found, skipped")
                continue

            per_run_pct = []
            per_run_spikes = []
            for p in paths:
                x_true, y_true, x_mean, y_mean = load_file(p)
                _, _, _, bias_pct, _ = step_size_bias(x_true, y_true, x_mean, y_mean)
                per_run_pct.append(bias_pct)

                step_spikes = load_step_spikes(p)
                if step_spikes is not None:
                    per_run_spikes.append(float(np.nanmean(step_spikes)))
            per_run_pct = np.asarray(per_run_pct, dtype=float)

            n_runs = per_run_pct.size
            mean_pct = float(np.mean(per_run_pct))
            # ddof=1: the spread is an estimate of run-to-run variability, not a
            # population parameter. Undefined for a single run.
            std_pct = float(np.std(per_run_pct, ddof=1)) if n_runs > 1 else np.nan
            sem_pct = std_pct / np.sqrt(n_runs) if n_runs > 1 else np.nan
            mean_spikes = float(np.mean(per_run_spikes)) if per_run_spikes else np.nan

            r = results[mode]
            r["n_cells"].append(n_cells)
            r["mean_pct"].append(mean_pct)
            r["std_pct"].append(std_pct)
            r["sem_pct"].append(sem_pct)
            r["n_runs"].append(n_runs)
            r["mean_spikes"].append(mean_spikes)
            print(f"{n_cells:8d} {n_runs:7d} {mean_pct:10.3f} {std_pct:9.3f} {sem_pct:9.3f} "
                  f"{mean_spikes:9.2f}")
    return results


def plot(results, out_path, error="std"):
    err_key = "std_pct" if error == "std" else "sem_pct"
    label_suffix = "std over runs" if error == "std" else "SEM over runs"

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for mode in MODES:
        r = results[mode]
        if not r["n_cells"]:
            continue
        ax.errorbar(r["n_cells"], r["mean_pct"], yerr=r[err_key],
            marker="o", capsize=3, color="tab:orange",
            label=f"{mode} (mean $\\pm$ {label_suffix})")

    ax.axhline(0, color="k", lw=0.75, ls="--")
    ax.set_xlabel("Number of place cells")
    ax.set_ylabel("Step-size bias [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.ticklabel_format(axis="x", style="plain")

    # secondary top x-axis: mean spikes/step
    n_cells_arr = np.asarray(results[MODES[0]]["n_cells"])
    spikes_arr = np.asarray(results[MODES[0]]["mean_spikes"])
    finite = np.isfinite(spikes_arr)
    n_cells_arr, spikes_arr = n_cells_arr[finite], spikes_arr[finite]
    order = np.argsort(n_cells_arr)
    n_cells_arr, spikes_arr = n_cells_arr[order], spikes_arr[order]
    if n_cells_arr.size >= 2:
        secax = ax.secondary_xaxis(
            "top",
            functions=(lambda n: np.interp(n, n_cells_arr, spikes_arr),
                       lambda s: np.interp(s, spikes_arr, n_cells_arr)))
        secax.set_xlabel("Mean spikes / 25 ms step")
        secax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="upper"))
        secax.ticklabel_format(axis="x", style="plain")

    fig.suptitle(f"2D step-size bias with event speed {replay_speed_mps():.2f} m/s",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_cells", nargs="*", type=int, default=DEFAULT_N_CELLS)
    ap.add_argument("--error", choices=["std", "sem"], default="std",
                    help="error band over runs: 'std' (run-to-run spread, default) "
                         "or 'sem' (uncertainty of the mean-over-runs)")
    ap.add_argument("--runs", type=int, nargs="*", default=None,
                    help="restrict to these run indices; default = all _run*.npy found")
    args = ap.parse_args()

    restrict = set(args.runs) if args.runs else None
    results = run(args.n_cells, restrict_runs=restrict)
    save_results(results)

    os.makedirs("figs", exist_ok=True)
    plot(results, f"figs/stepsize_bias_vs_ncells_id{params_id}_speed{replay_speed_mps():.2f}mps.png",
         error=args.error)
