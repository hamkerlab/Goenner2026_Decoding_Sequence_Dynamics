import os
import argparse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sim_params_seq import params_id, n_reps, n_mincells, n_windows, n_osc_cycles, gamma_mod_depth


COL_MODES = ["speedmod", "smooth"]  # left column, right column
COL_TITLES = {"speedmod": "Phase-locked movement", "smooth": "Smooth movement"}

N_PLOT     = 1000       # Max dots DRAWN per panel, thinned at random for
                        # legibility only. Statistics (rho, p) and the lm/loess
                        # fits use ALL finite data
LM_BLUE    = "#3366FF"  # ggplot geom_smooth(method="lm") default line colour
MARKER_SIZE  = 6        # geom_point size
MARKER_ALPHA = 0.4      # geom_point(alpha = 0.4)
FIT_LW       = 2        # lm line width
REF_LW       = 2        # loess (linewidth = 2 in R)

YLIMS = {
    ("spikes", "speedmod"): (None, None),
    ("spikes", "smooth"):   (None, None),
    ("step2d", "speedmod"): (-2.0, 30.0),
    ("step2d", "smooth"):   (-2.0, 30.0),
    ("dx",     "speedmod"): (-11.0, 21.0),
    ("dx",     "smooth"):   (-11.0, 21.0),
    ("dy",     "speedmod"): (-11.0, 21.0),
    ("dy",     "smooth"):   (-11.0, 21.0),
}

FS_TICK         = 14   # x/y tick labels
FS_AXIS_LABEL   = 14   # xlabel/ylabel
FS_TITLE        = 16
FS_STAT         = 16   # rho/p annotation box
FS_PANEL_LETTER = 18


def fold_phase(phase_deg):
    phi = np.deg2rad(phase_deg)
    return np.where(phi > np.pi, 2 * np.pi - phi, phi)


def circmean_pair_deg(phase_deg_per_window):
    phi = np.deg2rad(phase_deg_per_window)
    c = np.cos(phi)
    s = np.sin(phi)
    c_pair = 0.5 * (c[:-1] + c[1:])
    s_pair = 0.5 * (s[:-1] + s[1:])
    return np.degrees(np.arctan2(s_pair, c_pair)) % 360.0


def load(mode, n_cells, old_sim=False):
    if not old_sim:
        from sim_params_seq import mean_peak_Hz
        tag = "smooth" if mode == "smooth" else "speedmod1.0"
        base = (f"data/percellrates_sim_seq_mode-{tag}_paramsid_{params_id}"
                f"_{n_reps}reps_{int(n_mincells)}mincells_{n_cells}cells_{n_windows}steps"
                f"_{gamma_mod_depth}gammamod_{mean_peak_Hz}meanHz")
        npd = np.load(base + "_npformat.npy", allow_pickle=True)

        x_true, y_true = npd[0], npd[1]
        x_mean, y_mean = npd[2], npd[3]
        sd = np.load(base + "_stepdata.npy", allow_pickle=True)
        return x_true, y_true, x_mean, y_mean, sd[0], sd[1], sd[1]

    if mode == "smooth":
        base = (f"data/sim_seq_paramsid_{params_id}_{n_reps}reps_{int(n_mincells)}mincells"
                f"_{n_cells}cells_{n_windows}steps_{gamma_mod_depth}moddepth")
    else:
        base = (f"data/speedmod1.0_sim_seq_paramsid_{params_id}_{n_reps}reps_{int(n_mincells)}mincells"
                f"_{n_cells}cells_{n_windows}steps_{gamma_mod_depth}moddepth")
    npd = np.load(base + "_npformat.npy", allow_pickle=True)

    x_true, y_true, x_mean, y_mean, tot_spikes = npd[0], npd[1], npd[2], npd[3], npd[4]

    phase_rad_per_window = (np.arange(n_windows) / float(n_windows)) * n_osc_cycles * 2 * np.pi
    phase_deg_per_window = np.degrees(phase_rad_per_window) % 360.0

    spikes = tot_spikes[:, :-1]
    phase_spikes = np.tile(phase_deg_per_window[:-1], (tot_spikes.shape[0], 1))

    phase_diff_vals = circmean_pair_deg(phase_deg_per_window)
    phase_diff = np.tile(phase_diff_vals, (tot_spikes.shape[0], 1))

    return x_true, y_true, x_mean, y_mean, spikes, phase_spikes, phase_diff


def plot_panel(ax, phase_deg, y, ylabel, title, y_ref=None, ymin=None, ymax=None, rng=None, fontsize=FS_STAT):
    x = fold_phase(phase_deg)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rho, p = stats.spearmanr(x, y)

    if rng is None:
        rng = np.random.default_rng(0)
    draw = (rng.choice(x.size, size=N_PLOT, replace=False)
            if x.size > N_PLOT else np.arange(x.size))
    ax.scatter(x[draw], y[draw], alpha=MARKER_ALPHA, s=MARKER_SIZE,
               color="black", edgecolors="none", zorder=1)
    sns.regplot(x=x, y=y, ax=ax, scatter=False, ci=95,
                line_kws=dict(color=LM_BLUE, linewidth=FIT_LW, zorder=3))

    if y_ref is not None:
        yr = y_ref[m]
        mr = np.isfinite(x) & np.isfinite(yr)
        sns.regplot(x=x[mr], y=yr[mr], ax=ax, scatter=False, lowess=True,
                    line_kws=dict(color="red", linestyle="--", linewidth=REF_LW, zorder=4))

    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(["0", "$\\pi/4$", "$\\pi/2$", "3$\\pi/4$", "$\\pi$"], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlabel("Absolute Phase [rad]", fontsize=FS_AXIS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL)
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=FS_TITLE)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.grid(True, color="0.85", linewidth=0.6)
    p_str = "< 0.001" if p < 1e-3 else f"= {p:.3f}"
    ax.text(
        0.05, 0.92,
        f"$\\rho = {rho:.2f},\\ p {p_str}$",
        transform=ax.transAxes, va="top", ha="left", fontsize=fontsize, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"),
    )


def add_panel_letter(ax, letter, x=-0.12, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=FS_PANEL_LETTER, fontweight="bold", va="top", ha="left")


def load_mode_data(mode, n_cells, ref_line, old_sim=False):
    x_true, y_true, x_mean, y_mean, spikes, phase_spikes, phase_diff = load(mode, n_cells, old_sim)
    dx = np.diff(x_mean, axis=1)
    dy = np.diff(y_mean, axis=1)
    step2d = np.sqrt(dx ** 2 + dy ** 2)

    phase_spikes_f = phase_spikes.ravel()
    phase_diff_f = phase_diff.ravel()
    spikes_f = spikes.ravel()
    dx_f, dy_f, step2d_f = dx.ravel(), dy.ravel(), step2d.ravel()

    ref_2d = ref_dx = ref_dy = None
    if ref_line:
        dxt, dyt = np.diff(x_true, axis=1), np.diff(y_true, axis=1)
        ref_2d = np.sqrt(dxt ** 2 + dyt ** 2).ravel()
        ref_dx, ref_dy = dxt.ravel(), dyt.ravel()

    return dict(phase_spikes=phase_spikes_f, phase_diff=phase_diff_f,
                spikes=spikes_f, dx=dx_f, dy=dy_f, step2d=step2d_f,
                ref_2d=ref_2d, ref_dx=ref_dx, ref_dy=ref_dy)


def run_all(n_cells, ref_line=False, old_sim=False):
    data = {mode: load_mode_data(mode, n_cells, ref_line, old_sim) for mode in COL_MODES}
    fmt_tag = "_oldformat" if old_sim else ""
    os.makedirs("figs", exist_ok=True)
    # seed for reproducible random thinning of scatter points in the plots
    rng = np.random.default_rng(0)
    
    # Fig. 1: spikes vs phase, 1 row x 2 cols (speedmod | smooth). Sanity check.
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4))
    for col, mode in enumerate(COL_MODES):
        d = data[mode]
        ymin, ymax = YLIMS[("spikes", mode)]
        plot_panel(axes1[col], d["phase_spikes"], d["spikes"], "Spikes / 25 ms step",
                   f"Spikes vs abs phase ({COL_TITLES[mode]})", None, ymin, ymax, rng=rng)
        add_panel_letter(axes1[col], "abcdefgh"[col])
    fig1.tight_layout()
    out1 = f"figs/phase_spikes_id{params_id}_{n_cells}cells{fmt_tag}.png"
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"saved {out1}")

    # Fig. 2: step kinematics, 3 rows (2D, dx, dy) x 2 cols (speedmod | smooth)
    row_specs = [
        ("step2d", "ref_2d", "2D step size [cm]", "2D step vs abs phase"),
        ("dx",     "ref_dx", "dx [cm]",           "dx vs abs phase"),
        ("dy",     "ref_dy", "dy [cm]",           "dy vs abs phase"),
    ]
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 10))
    letters2 = "abcdefghijklmn"
    li = 0
    for row, (key, refkey, ylabel, title) in enumerate(row_specs):
        for col, mode in enumerate(COL_MODES):
            d = data[mode]
            ymin, ymax = YLIMS[(key, mode)]
            # only the top row gets a title
            if row == 0:
                plot_panel(axes2[row, col], d["phase_diff"], d[key], ylabel,
                        f"{COL_TITLES[mode]}", d[refkey], ymin, ymax, rng=rng)
            else:
                plot_panel(axes2[row, col], d["phase_diff"], d[key], ylabel,
                        None, d[refkey], ymin, ymax, rng=rng)
            add_panel_letter(axes2[row, col], letters2[li])
            li += 1
    fig2.tight_layout()
    out2 = f"figs/phase_kinematics_id{params_id}_{n_cells}cells{fmt_tag}.png"
    fig2.savefig(out2, dpi=300)
    plt.close(fig2)
    print(f"saved {out2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_cells", nargs="?", type=int, default=100)
    ap.add_argument("--ref-line", action="store_true",
                    help="draw the red dashed loess of ground-truth kinematics")
    ap.add_argument("--old-sim", action="store_true", dest="old_sim",)

    args = ap.parse_args()
    run_all(args.n_cells, ref_line=args.ref_line, old_sim=args.old_sim)
