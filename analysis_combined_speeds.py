import argparse
import colorsys
import glob
import os
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _tab_orange_gradient():
    """
    Sequential colormap anchored on tab:orange: light tint -> tab:orange
    itself (at the midpoint) -> dark shade, same hue throughout.
    """
    h, s, v = colorsys.rgb_to_hsv(*mcolors.to_rgb("tab:orange"))
    light = colorsys.hsv_to_rgb(h, s * 0.35, min(1.0, v * 1.25))
    dark = colorsys.hsv_to_rgb(h, min(1.0, s * 1.3), v * 0.55)
    return mcolors.LinearSegmentedColormap.from_list("tab_orange_grad", [light, "tab:orange", dark])


def load_summary(path, mode="smooth"):
    """
    Read one per-speed summary .npz. Falls back to whichever mode is
    present if the requested one wasn't saved for that file.
    """
    with np.load(path) as npz:
        prefix = f"{mode}__"
        if f"{prefix}n_cells" not in npz.files:
            avail = sorted({k.split("__")[0] for k in npz.files if "__" in k})
            if not avail:
                raise ValueError(f"{path}: no per-mode summary data found")
            prefix = f"{avail[0]}__"
        d = {"speed_mps": float(npz["speed_mps"]), "params_id": str(npz["params_id"])}
        for key in ("n_cells", "mean_pct", "std_pct", "sem_pct", "n_runs", "mean_spikes"):
            d[key] = npz[f"{prefix}{key}"]
        return d


def load_all(summary_dir, params_id=None, mode="smooth"):
    paths = sorted(glob.glob(os.path.join(summary_dir, "stepsize_bias_id*_speed*mps.npz")))
    if params_id is not None:
        paths = [p for p in paths if f"_id{params_id}_" in os.path.basename(p)]
    if not paths:
        raise SystemExit(f"no summary .npz files found in {summary_dir!r} "
                          f"(params_id filter: {params_id})")

    summaries = [load_summary(p, mode=mode) for p in paths]
    ids_present = {s["params_id"] for s in summaries}
    if len(ids_present) > 1:
        raise SystemExit(f"summaries span multiple params_id {ids_present}; "
                          f"pass --params-id to pick one")
    summaries.sort(key=lambda s: s["speed_mps"])
    return summaries, next(iter(ids_present))


def plot_combined(summaries, params_id, out_path, error="std"):
    err_key = "std_pct" if error == "std" else "sem_pct"

    speeds = [s["speed_mps"] for s in summaries]
    cmap = _tab_orange_gradient()
    norm = mcolors.Normalize(vmin=min(speeds), vmax=max(speeds))

    n_panels = len(summaries)
    y_max = max(np.nanmax(s["mean_pct"] + np.nan_to_num(s[err_key])) for s in summaries) * 1.08
    y_min = min(np.nanmin(s["mean_pct"] - np.nan_to_num(s[err_key])) for s in summaries)
    y_min = min(y_min, 0) * 1.08

    fig, axes = plt.subplots(1, n_panels, figsize=(3.8 * n_panels, 3), sharey=True)
    axes = np.atleast_1d(axes)

    # build each panel's (n_cells, mean_spikes) curve and find
    # the spike range common to every panel, so the shared top-axis ticks we
    # pick below are guaranteed valid (no extrapolation/clamping) everywhere.
    panel_curves = []
    common_hi = np.inf
    for s in summaries:
        n_cells = s["n_cells"]
        finite = np.isfinite(s["mean_spikes"])
        if finite.sum() >= 2:
            nc, sp = n_cells[finite], s["mean_spikes"][finite]
            order = np.argsort(nc)
            nc, sp = nc[order], sp[order]
            xlim = (n_cells.min() * 0.8, n_cells.max() * 1.25)
            hi = np.interp(xlim[1], nc, sp)
            common_hi = min(common_hi, hi)
        else:
            nc = sp = None
        panel_curves.append((nc, sp))

    shared_ticks = []
    if np.isfinite(common_hi) and common_hi > 0:
        n_target = 5
        step = max(20, round(common_hi / (n_target - 1) / 20) * 20)
        top = int(np.ceil(common_hi / step) * step)
        shared_ticks = list(range(0, min(top, int(common_hi)) + 1, step))
        if 20 <= common_hi:
            shared_ticks = sorted(set(shared_ticks) | {20})
        if not shared_ticks:
            shared_ticks = [0]

    for panel_idx, (ax, s) in enumerate(zip(axes, summaries)):
        color = cmap(norm(s["speed_mps"]))
        n_cells, mean_pct, err = s["n_cells"], s["mean_pct"], s[err_key]
        err = np.nan_to_num(err)  # single-run points have no std/sem

        ax.plot(n_cells, mean_pct, color=color, lw=2.2, marker="o", ms=5, zorder=3)
        ax.fill_between(n_cells, mean_pct - err, mean_pct + err, color=color, alpha=0.25, zorder=2)
        ax.axhline(0, color="k", ls="--", lw=1, zorder=1)
        ax.set_xscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(n_cells.min() * 0.8, n_cells.max() * 1.25)

        # loc='left'/'center' titles share one y-coordinate; that's what
        # keeps the panel letter aligned with the speed label above it.
        ax.set_title(f"{s['speed_mps']:.2f} m/s", loc="center", fontweight="bold", pad=24)
        letter = ax.set_title(string.ascii_lowercase[panel_idx], loc="left",
                               fontweight="bold", fontsize=13, pad=24)
        letter.set_x(-0.1)  # loc='left' default sits at axes x=0; nudge left of it

        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.set_xlabel("Number of place cells")

        nc, sp = panel_curves[panel_idx]
        if nc is not None and shared_ticks:
            secax = ax.secondary_xaxis(
                "top",
                functions=(lambda n, nc=nc, sp=sp: np.interp(n, nc, sp),
                           lambda v, nc=nc, sp=sp: np.interp(v, sp, nc)))
            # Shared spike-value ticks are guaranteed within [0, common_hi]
            xlo_log, xhi_log = np.log10(ax.get_xlim())
            min_frac = 0.06
            # force 20 to be the first tick if it's in range, then fill in the rest
            priority = sorted(shared_ticks, key=lambda t: (t != 20, t))
            kept, kept_logpos = [], []
            for t in priority:
                pos = np.log10(np.interp(t, sp, nc))
                if all(abs(pos - kp) >= min_frac * (xhi_log - xlo_log) for kp in kept_logpos):
                    kept.append(t)
                    kept_logpos.append(pos)
            kept.sort()
            secax.set_xticks(kept)
            secax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            secax.tick_params(labelsize=8, pad=2)
            
            secax.set_xlabel("Mean spikes / step", fontsize=9,
                              color="black" if panel_idx == 0 else "none")

    axes[0].set_ylabel("Step-size bias [%]")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Replay speed [m/s]")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="bias_summaries",
                     help="directory of per-speed .npz summaries (default: %(default)s)")
    ap.add_argument("--params-id", default=None,
                     help="restrict to this params_id; required if the summary dir "
                          "mixes more than one")
    ap.add_argument("--mode", default="smooth")
    ap.add_argument("--error", choices=["std", "sem"], default="std",
                     help="error band per panel: 'std' (run-to-run spread, default) "
                          "or 'sem' (uncertainty of the mean-over-runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    summaries, params_id = load_all(args.summary_dir, params_id=args.params_id, mode=args.mode)
    out = args.out or f"figs/stepsize_bias_combined_id{params_id}.png"
    plot_combined(summaries, params_id, out, error=args.error)
