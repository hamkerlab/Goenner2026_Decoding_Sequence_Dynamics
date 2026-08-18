import os
import argparse
import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from sim_params_seq import (
    params_id, n_reps, n_mincells, n_windows, mean_peak_Hz, n_osc_cycles, gamma_mod_depth
)
from analysis_spearman_phase import (
    load_mode_data as load_spearman_data, plot_panel, add_panel_letter, YLIMS
)

GAMMA_MOD_DEPTH = gamma_mod_depth
MODE_TITLES = {"speedmod": "Phase-locked movement", "smooth": None}

COL_BLUE2 = "#0072B2"
COL_GREEN = "#2B9F78"


def plot_spearman_panel(ax, phase_deg, y, ylabel, title, y_ref=None, ymin=None, ymax=None):
    plot_panel(ax, phase_deg, y, ylabel, None, y_ref, ymin, ymax, fontsize=12)


# --------------------------------------------------------------------------
# Polar-panel data loading (Hilbert phase, 
# comp1_addpositions_hilbert.py)
# --------------------------------------------------------------------------
def load_polar_raw(mode, n_cells, old_sim=False):
    if not old_sim:
        tag = "smooth" if mode == "smooth" else "speedmod1.0"
        base = (f"data/percellrates_sim_seq_mode-{tag}_paramsid_{params_id}"
                f"_{n_reps}reps_{int(n_mincells)}mincells_{n_cells}cells_{n_windows}steps"
                f"_{GAMMA_MOD_DEPTH}gammamod_{mean_peak_Hz}meanHz")
    elif mode == "smooth":
        base = (f"data/sim_seq_paramsid_{params_id}_{n_reps}reps_{int(n_mincells)}mincells"
                f"_{n_cells}cells_{n_windows}steps_{GAMMA_MOD_DEPTH}moddepth")
    else:
        base = (f"data/speedmod1.0_sim_seq_paramsid_{params_id}_{n_reps}reps_{int(n_mincells)}mincells"
                f"_{n_cells}cells_{n_windows}steps_{GAMMA_MOD_DEPTH}moddepth")

    npd = np.load(base + "_npformat.npy", allow_pickle=True)
    x_true, y_true, x_mean, y_mean, tot_spikes = npd[0], npd[1], npd[2], npd[3], npd[4]

    window_dim = 1
    true_xdiff = np.diff(x_true, axis=window_dim)
    true_ydiff = np.diff(y_true, axis=window_dim)
    L2_diff_perframe = np.sqrt(np.diff(x_mean, axis=window_dim) ** 2
                                + np.diff(y_mean, axis=window_dim) ** 2)

    hilb = hilbert(tot_spikes - tot_spikes.mean())
    hilb_phase = np.arctan2(hilb.imag, hilb.real)

    return dict(tot_spikes=tot_spikes, true_xdiff=true_xdiff, true_ydiff=true_ydiff,
                L2_diff_perframe=L2_diff_perframe, hilb_phase=hilb_phase)


def compute_polar_series(d):

    phase_vals = np.arange(-np.pi, np.pi, np.pi / 18.0)
    n_bins = len(phase_vals)
    centres = phase_vals + np.pi / 36.0

    phase_bins = np.clip(np.digitize(d["hilb_phase"][:, :-1], phase_vals) - 1, 0, n_bins - 1)

    mean_L2diff_i = np.full(n_bins, np.nan)
    true_step_i = np.full(n_bins, np.nan)
    spikes_i = np.full(n_bins, np.nan)

    for i in range(n_bins):
        idx = np.nonzero(phase_bins == i)
        if len(idx[0]) > 0:
            mean_L2diff_i[i] = np.nanmean(d["L2_diff_perframe"][idx])
            txm = np.nanmean(d["true_xdiff"][idx])
            tym = np.nanmean(d["true_ydiff"][idx])
            true_step_i[i] = np.sqrt(txm ** 2 + tym ** 2)
            spikes_i[i] = np.nanmean(d["tot_spikes"][idx])

    spikes_i = spikes_i / np.nanmax(spikes_i) * 15

    pad = np.concatenate([mean_L2diff_i[-1:], mean_L2diff_i, mean_L2diff_i[:1]])
    smoothed = np.array([np.nanmean(pad[i:i + 3]) for i in range(n_bins)])

    return dict(phase_vals=centres, mean_L2diff_smoothed=smoothed,
                true_step=true_step_i, spikes=spikes_i)


def _close(theta, r):
    return np.append(theta, theta[0] + 2 * np.pi), np.append(r, r[0])


def plot_polar_panel(ax, s, title):
    th = s["phase_vals"]
    for r, col in ((s["mean_L2diff_smoothed"], COL_BLUE2),
                   (s["true_step"], "k"),
                   (s["spikes"], COL_GREEN)):
        t_c, r_c = _close(th, r)
        ax.plot(t_c, r_c, "-", color=col, linewidth=3)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(["0", "", "", "$\\pi$/2", "", "", "$\\pi$", "", "", "3$\\pi$/2", "", ""], fontsize=12)
    ax.tick_params(axis="x", pad=6)
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.8) 
    ax.yaxis.grid()
    ax.set_title(title, fontweight="bold")


# --------------------------------------------------------------------------
# Combined figure
# --------------------------------------------------------------------------
def run_all(mode, n_cells, ref_line=False, ymin=None, ymax=None, old_sim=False):
    spearman_data = load_spearman_data(mode, n_cells, ref_line, old_sim)
    polar_raw = load_polar_raw(mode, n_cells, old_sim)
    polar_series = compute_polar_series(polar_raw)

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2], hspace=0.55, wspace=0.6)
    
    # --- panel a: polar, column 0, spans all three rows ---
    ax_polar = fig.add_subplot(gs[0:3, 0], projection="polar")
    plot_polar_panel(ax_polar, polar_series, MODE_TITLES[mode])
    pos = ax_polar.get_position()
    fig_w, fig_h = fig.get_size_inches()
    new_h = (pos.width * fig_w) / fig_h
    ax_polar.set_position([pos.x0, pos.y0 + (pos.height - new_h) / 2, pos.width, new_h])
    add_panel_letter(ax_polar, "a", x=-0.15, y=1.2)

    legend_handles = [
        Line2D([0], [0], color=COL_BLUE2, lw=2),
        Line2D([0], [0], color="k", lw=2),
        Line2D([0], [0], color=COL_GREEN, lw=2),
    ]
    ax_polar.legend(legend_handles, ["2D step size", "True step size", "Population rate"],
                     loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=14, frameon=True)

    # --- panel b: 2D step size vs phase, column 1, row 0 ---
    y2min, y2max = YLIMS[("step2d", mode)]
    ax_b = fig.add_subplot(gs[0, 1])
    plot_spearman_panel(ax_b, spearman_data["phase_diff"], spearman_data["step2d"],
                         "2D step size [cm]", MODE_TITLES[mode], spearman_data["ref_2d"], y2min, y2max)
    ax_b.set_xlabel("")
    add_panel_letter(ax_b, "b", x=-0.2, y=1.05)

    # --- panel c: dx vs phase, column 1, row 1 ---
    ax_c = fig.add_subplot(gs[1, 1])
    plot_spearman_panel(ax_c, spearman_data["phase_diff"], spearman_data["dx"], "dx [cm]",
                         MODE_TITLES[mode], spearman_data["ref_dx"], ymin, ymax)
    ax_c.set_xlabel("")
    add_panel_letter(ax_c, "c", x=-0.2, y=1.05)

    # --- panel d: dy vs phase, column 1, row 2 ---
    ax_d = fig.add_subplot(gs[2, 1])
    plot_spearman_panel(ax_d, spearman_data["phase_diff"], spearman_data["dy"], "dy [cm]",
                         MODE_TITLES[mode], spearman_data["ref_dy"], ymin, ymax)
    add_panel_letter(ax_d, "d", x=-0.2, y=1.05)

    os.makedirs("figs", exist_ok=True)
    fmt_tag = "_oldformat" if old_sim else ""
    out = f"figs/combined_polar_dxdy_{mode}_id{params_id}_{n_cells}cells{fmt_tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_cells", nargs="?", type=int, default=100)
    ap.add_argument("--mode", choices=["smooth", "speedmod"], default="smooth",
                     help="movement condition to plot (default: smooth)")
    ap.add_argument("--ref-line", action="store_true",
                     help="draw the red dashed loess of ground-truth kinematics on panels b/c/d")
    ap.add_argument("--ymin", type=float, default=None, help="lower y limit for panels c/d (default: auto)")
    ap.add_argument("--ymax", type=float, default=None, help="upper y limit for panels c/d (default: auto)")
    ap.add_argument("--old-sim", action="store_true", dest="old_sim")

    args = ap.parse_args()
    run_all(args.mode, args.n_cells, ref_line=args.ref_line, ymin=args.ymin, ymax=args.ymax,
            old_sim=args.old_sim)
