from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


MEASURES = ("step_2d_cm", "directed_1d_cm")

# Spike-count noise transforms. Adding a form here propagates to the per-step
# predictor columns, the lo/hi anchor deltas, the fitted curves and the plots.
NOISE_FORMS = {
    "inv_n": lambda n: 1.0 / n, # used for testing, not in the main paper
    "inv_sqrt_n": lambda n: 1.0 / np.sqrt(n),
}
METHODS = ("nonparam", *NOISE_FORMS)

# Plot colors per measure (colorblind-safe).
COLORS = {
    "step_2d_cm": "#D55E00",      # vermillion
    "directed_1d_cm": "#0072B2",  # blue
    "difference": "#7B2D8B",      # violet
    "c_less": "#D62728",          # red 
    "c_greater": "#1F77B4"        # blue
}


def _fold_phase_rad(phase_deg):
    """Degrees in [0, 360) -> radians folded into [0, pi]."""
    phi = np.deg2rad(phase_deg)
    return np.where(phi > np.pi, 2 * np.pi - phi, phi)


def _rho(x, y):
    # version-safe Spearman; returns nan if an input is constant in a resample
    r = stats.spearmanr(x, y)[0]
    return np.nan if r is None else r


def _event_row_groups(events: np.ndarray, *, sort: bool) -> list[np.ndarray]:
    """Row-index arrays grouped by event, for cluster resampling.

    `sort` mirrors pandas groupby: sort=True gives groups in sorted-event
    order, sort=False in first-appearance order. The order is irrelevant to
    the statistics but fixes which event each bootstrap draw selects, so it is
    kept per call site to reproduce existing results bit-for-bit.
    """
    pos = pd.Series(np.arange(len(events)))
    return [g.to_numpy() for _, g in pos.groupby(np.asarray(events), sort=sort)]


def _cluster_bootstrap(group_rows, n_boot, seed, fn) -> np.ndarray:
    """Resample whole events with replacement; stack fn(rows) into (n_boot, k)."""
    rng = np.random.default_rng(seed)
    n = len(group_rows)
    out = None
    for b in range(n_boot):
        rows = np.concatenate([group_rows[i] for i in rng.integers(0, n, n)])
        vals = np.asarray(fn(rows), dtype=float)
        if out is None:
            out = np.empty((n_boot, vals.size))
        out[b] = vals
    return out


def stepsize_2d_vs_1d_diff(
    df,
    *,
    event_col="event_uid",
    phase_col="phase",
    two_step_col="2Dsteps",
    signed_y_col="1Dsteps_y",
    n_boot=5000,
    seed=0,
):
    """
    Compare how folded phase relates to 2D step size vs. directed 1D step size.

    For each event we estimate a net direction from the signed 1D (y) steps and
    use it to re-sign every step in that event ("directed 1D"). We then compute
    two pooled Spearman correlations against folded phase:

        rho_2D           : folded phase vs. 2D step size (a non-negative magnitude)
        rho_directed_1D  : folded phase vs. directed 1D step (signed)

    Both correlations, and their difference, are bootstrapped by resampling whole
    events with replacement (cluster bootstrap) to obtain percentile CIs.
    """

    by_event = df.groupby(event_col)[signed_y_col]
    events = pd.DataFrame({
        "net_dy": by_event.sum(),
        "consistency": by_event.apply(lambda s: np.sign(s).mean()),  # computed, unused internally
        "n_steps": by_event.size(),
    })
    events["dir"] = np.sign(events["net_dy"])

    folded_phase = _fold_phase_rad(df[phase_col].to_numpy())
    step_direction = df[event_col].map(events["dir"]).to_numpy()
    directed_dy = step_direction * df[signed_y_col].to_numpy()
    step_2d = df[two_step_col].to_numpy()
    step_event = df[event_col].to_numpy()

    valid = np.isfinite(folded_phase) & np.isfinite(step_2d) & np.isfinite(directed_dy)
    folded_phase = folded_phase[valid]
    step_2d = step_2d[valid]
    directed_dy = directed_dy[valid]
    step_event = step_event[valid]

    event_groups = _event_row_groups(step_event, sort=True)

    def paired_rhos(rows):
        phase = folded_phase[rows]
        return _rho(phase, step_2d[rows]), _rho(phase, directed_dy[rows])

    all_rows = np.arange(folded_phase.size)
    rho_2d_hat, rho_1d_hat = paired_rhos(all_rows)

    def boot_fn(rows):
        r2, r1 = paired_rhos(rows)
        return r2, r1, r2 - r1

    boot = _cluster_bootstrap(event_groups, n_boot, seed, boot_fn)

    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    summary = pd.DataFrame(
        {
            "estimate": [rho_2d_hat, rho_1d_hat, rho_2d_hat - rho_1d_hat],
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        },
        index=["rho_2D", "rho_directed_1D", "difference"],
    )
    summary.attrs["p_diff_le_0"] = float(np.nanmean(boot[:, 2] <= 0))
    return summary, boot, events


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form simple linear regression -> (slope, intercept).

    Used in the bootstrap hot loop; a stats package would add per-call overhead
    (x n_boot x measures x forms) for analytic SEs we don't use (the bootstrap
    supplies the CI). This is exactly np.polyfit(x, y, 1) but ~10x cheaper.
    """
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return np.nan, np.nan
    slope = float(xc @ (y - y.mean()) / denom)
    return slope, float(y.mean() - slope * x.mean())


def _safe_mean(a: np.ndarray) -> float:
    return float(a.mean()) if a.size else np.nan


def _prepare_steps(
    df,
    *,
    event_col,
    spikes_col,
    two_step_col,
    signed_y_col,
    bin_size_cm,
    min_spikes,
) -> pd.DataFrame:

    net_dy = df.groupby(event_col)[signed_y_col].transform("sum").to_numpy()
    direction = np.sign(net_dy)  # same convention as stepsize_2d_vs_1d_diff

    spikes = df[spikes_col].to_numpy(dtype=float)
    pos = spikes > 0
    noise = {form: np.where(pos, f(spikes), np.nan) for form, f in NOISE_FORMS.items()}

    steps = pd.DataFrame({
        "event": df[event_col].to_numpy(),
        "spikes": spikes,
        **noise,
        "step_2d_cm": df[two_step_col].to_numpy(dtype=float) * bin_size_cm,
        "directed_1d_cm": direction * df[signed_y_col].to_numpy(dtype=float) * bin_size_cm,
    })
    cols = [*NOISE_FORMS, *MEASURES]
    ok = steps[cols].notna().all(axis=1).to_numpy() & (spikes >= min_spikes)
    return steps.loc[ok].reset_index(drop=True)


def _anchors(spikes: np.ndarray, n_lo_pct, n_hi_pct):
    """Spike-count percentiles and the per-form low-high predictor deltas."""
    n_lo, n_hi = np.percentile(spikes, [n_lo_pct, n_hi_pct])
    deltas = {form: f(n_lo) - f(n_hi) for form, f in NOISE_FORMS.items()}
    return (float(n_lo), float(n_hi)), deltas


def _estimates(arrays, rows, anchors, deltas, *, want_fits=False):
    """All inflation estimates for one (sub)sample of step rows.

    Returns {(measure, method): cm}. If want_fits, also returns
    {(measure, form): (slope, intercept)} for plotting the fitted curves.
    """
    n_lo, n_hi = anchors
    spikes = arrays["spikes"][rows]
    lo, hi = spikes <= n_lo, spikes >= n_hi

    out, fits = {}, {}
    for measure in MEASURES:
        y = arrays[measure][rows]
        out[(measure, "nonparam")] = _safe_mean(y[lo]) - _safe_mean(y[hi])
        for form in NOISE_FORMS:
            slope, intercept = _ols_slope_intercept(arrays[form][rows], y)
            out[(measure, form)] = slope * deltas[form]
            if want_fits:
                fits[(measure, form)] = (slope, intercept)
    for method in METHODS:
        out[("difference", method)] = (out[("step_2d_cm", method)]
                                       - out[("directed_1d_cm", method)])
    return (out, fits) if want_fits else out


def stepsize_noise_slope(
    df,
    *,
    event_col="event_uid",
    spikes_col="spikes",
    two_step_col="2Dsteps",
    signed_y_col="1Dsteps_y",
    bin_size_cm=2.0,
    min_spikes=1,
    n_lo_pct=10.0,
    n_hi_pct=90.0,
    n_boot=5000,
    seed=0,
):
    """
    Estimate low-N step-size inflation for 2D vs directed-1D and their difference.
    """
    steps = _prepare_steps(
        df, event_col=event_col, spikes_col=spikes_col, two_step_col=two_step_col,
        signed_y_col=signed_y_col, bin_size_cm=bin_size_cm, min_spikes=min_spikes,
    )
    arrays = {c: steps[c].to_numpy() for c in
              ("spikes", *NOISE_FORMS, *MEASURES)}
    all_rows = np.arange(len(steps))

    anchors, deltas = _anchors(arrays["spikes"], n_lo_pct, n_hi_pct)
    point, fits = _estimates(arrays, all_rows, anchors, deltas, want_fits=True)
    keys = list(point)

    group_rows = _event_row_groups(steps["event"].to_numpy(), sort=False)
    n_events = len(group_rows)

    def boot_fn(rows):
        est = _estimates(arrays, rows, anchors, deltas)
        return [est[k] for k in keys]

    boot = _cluster_bootstrap(group_rows, n_boot, seed, boot_fn)

    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    summary = pd.DataFrame(
        {"estimate_cm": [point[k] for k in keys], "ci_lo": ci_lo, "ci_hi": ci_hi},
        index=pd.MultiIndex.from_tuples(keys, names=["measure", "method"]),
    )
    diff_cols = {m: keys.index(("difference", m)) for m in METHODS}
    summary.attrs["p_diff_le_0"] = {
        m: float(np.nanmean(boot[:, j] <= 0)) for m, j in diff_cols.items()}
    summary.attrs["N_lo"], summary.attrs["N_hi"] = anchors
    summary.attrs["n_steps"], summary.attrs["n_events"] = len(steps), n_events
    summary.attrs["fits"] = fits
    return summary, boot, steps


def plot_stepsize_2d_vs_1d_diff(
    summary,
    boot,
    title=None,
    whisker_ymax=None,
    axes=None,
):

    import matplotlib.pyplot as plt

    boot = np.asarray(boot)
    b2, b1, bdiff = boot[:, 0], boot[:, 1], boot[:, 2]

    C2, C1, CDIFF = COLORS["step_2d_cm"], COLORS["directed_1d_cm"], COLORS["difference"]

    if axes is None:
        fig, (ax, ax_diff) = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    else:
        ax, ax_diff = axes
        fig = ax.figure

    lohi = np.nanpercentile(np.concatenate([b2, b1]), [0.5, 99.5])
    bins = np.linspace(min(0, lohi[0]), lohi[1], 70)

    measures = [
        (b2, C2, r"$\rho_{2D}$", "rho_2D"),
        (b1, C1, r"$\rho_{directed\ 1D}$", "rho_directed_1D"),
    ]

    for samples, color, label, row_key in measures:
        ax.hist(samples, bins=bins, color=color, alpha=0.35, edgecolor="none", label=label)

    ymax = whisker_ymax if whisker_ymax is not None else ax.get_ylim()[1]
    whisker_ys = [ymax * 0.78, ymax * 0.67]

    for (_, color, _, row_key), whisker_y in zip(measures, whisker_ys):
        ci_lo = summary.loc[row_key, "ci_lo"]
        ci_hi = summary.loc[row_key, "ci_hi"]

        ax.errorbar(
            x=(ci_lo + ci_hi) / 2,
            y=whisker_y,
            xerr=(ci_hi - ci_lo) / 2,
            fmt="none",
            color=color,
            lw=2.0,
            capsize=6,
            capthick=2.0,
        )

    ax.axvline(0, color="0.6", lw=0.8, ls="--")
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"Spearman $\rho$ (gamma phase vs. step size)")
    ax.set_ylabel("Bootstrap count")
    if axes is None:
        ax.set_title("a", loc="left", fontweight="bold")
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    lohi_diff = np.nanpercentile(bdiff, [0.5, 99.5])
    bins_diff = np.linspace(lohi_diff[0], lohi_diff[1], 60)

    ax_diff.hist(bdiff, bins=bins_diff, color=CDIFF, alpha=0.40, edgecolor="none",
                 label=r"$\rho_{2D} - \rho_{directed\ 1D}$")

    ci_lo_diff = summary.loc["difference", "ci_lo"]
    ci_hi_diff = summary.loc["difference", "ci_hi"]
    est_diff = summary.loc["difference", "estimate"]

    ax_diff.errorbar(
        x=(ci_lo_diff + ci_hi_diff) / 2,
        y=ymax * 0.78,
        xerr=(ci_hi_diff - ci_lo_diff) / 2,
        fmt="none",
        color=CDIFF,
        lw=2.0,
        capsize=6,
        capthick=2.0,
    )
    ax_diff.axvline(0, color="0.6", lw=0.8, ls="--", label="no difference")
    ax_diff.axvline(est_diff, color=CDIFF, lw=1.2, ls=":", alpha=0.8)

    ax_diff.set_ylim(0, ymax)
    ax_diff.set_xlabel(r"$\rho_{2D} - \rho_{directed\ 1D}$")
    if axes is None:
        ax_diff.set_title("b", loc="left", fontweight="bold")
    ax_diff.legend(frameon=True, fontsize=9, loc="upper left")

    if axes is None:
        if title:
            fig.suptitle(title, fontweight="bold", y=1.02)
        fig.tight_layout()
    return fig


def plot_stepsize_noise_slope_bootstrap(
    summary,
    boot,
    *,
    method="nonparam",
    title=None,
    whisker_ymax=None,
    add_1D=True,
):
    """
    Bootstrap distribution of spike inflation estimates.
    """
    import matplotlib.pyplot as plt

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    boot = np.asarray(boot)
    method_label = {
        "nonparam": "non-parametric (lo-hi mean)",
        "inv_n": r"$1/N$ fit",
        "inv_sqrt_n": r"$1/\sqrt{N}$ fit",
    }[method]

    measures = [(("step_2d_cm", method), COLORS["step_2d_cm"], "2D"),]
    if add_1D:
        measures += [(("directed_1d_cm", method), COLORS["directed_1d_cm"], "directed 1D")]

    cols = [boot[:, summary.index.get_loc(key)] for key, _, _ in measures]

    fig, ax = plt.subplots(figsize=(7, 4.2))

    lohi = np.nanpercentile(np.concatenate(cols), [0.5, 99.5])
    bins = np.linspace(min(0, lohi[0]), lohi[1], 70)
    for samples, (key, color, label) in zip(cols, measures):
        ax.hist(samples, bins=bins, color=color, alpha=0.35, edgecolor="none", label=label)

    ymax = whisker_ymax if whisker_ymax is not None else ax.get_ylim()[1]
    whisker_ys = [ymax * 0.78, ymax * 0.67]
    for (key, color, _), whisker_y in zip(measures, whisker_ys):
        ci_lo = summary.loc[key, "ci_lo"]
        ci_hi = summary.loc[key, "ci_hi"]
        ax.errorbar(
            x=(ci_lo + ci_hi) / 2,
            y=whisker_y,
            xerr=(ci_hi - ci_lo) / 2,
            fmt="none",
            color=color,
            lw=2.0,
            capsize=6,
            capthick=2.0,
        )

    ax.axvline(0, color="0.6", lw=0.8, ls="--")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Step-size inflation [cm] (Step size at low N - high N)")
    ax.set_ylabel("Bootstrap count")
    ax.set_title(f"{method_label}", loc="left", fontweight="bold")
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    if title:
        fig.suptitle(title, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def plot_stepsize_noise_slope(
    summary, 
    steps, 
    *, 
    n_bins=8, 
    fit_form="inv_n",
    n_boot=5000, 
    seed=0, 
    ci=95, 
    figsize=(7, 4.5),
    title=None,
    ax=None,
    add_1D=True,
    scatter: Optional[int] = None,
    ylim: Optional[tuple[float, float]] = None,
    xlim: Optional[tuple[float, float]] = None,
):
    """
    Binned mean step size vs spike count, with cluster-bootstrap CIs.

    scatter : number of individual (spike count, 2D step size) points to draw
        as a gray background cloud, or None for no cloud. Drawn without
        replacement from the full sample.
    ylim, xlim : (bottom, top) / (left, right) passed to set_ylim / set_xlim.
        The raw scatter spans a far wider range than the binned means in both
        axes, so without these it will squash the mean curve.
    """
    import matplotlib.pyplot as plt

    fits = summary.attrs["fits"]
    labels = {"step_2d_cm": "2D",}
    if add_1D:
        labels["directed_1d_cm"] = "directed 1D"
    fit_label = {"inv_n": r"$1/N$ fit", "inv_sqrt_n": r"$1/\sqrt{N}$ fit"}

    spikes = steps["spikes"].to_numpy()
    group_rows = _event_row_groups(steps["event"].to_numpy(), sort=False)

    # quantile bins in spike count, shared by the point estimate and bootstrap
    edges = np.unique(np.quantile(spikes, np.linspace(0, 1, n_bins + 1)))
    bin_id = np.clip(np.digitize(spikes, edges[1:-1]), 0, len(edges) - 2)
    n_bin = len(edges) - 1
    centers = np.array([spikes[bin_id == b].mean() for b in range(n_bin)])
    grid = np.linspace(2, 100, 200)

    def bin_means(y, rows):
        b = bin_id[rows]
        return np.array([y[rows][b == k].mean() if (b == k).any() else np.nan
                         for k in range(n_bin)])

    a = (100 - ci) / 2
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if scatter is not None and scatter > 0:
        y_raw = steps["step_2d_cm"].to_numpy()
        n_raw = len(y_raw)
        sel = (np.arange(n_raw) if scatter >= n_raw else
               np.random.default_rng(seed).choice(n_raw, scatter, replace=False))
        ax.scatter(spikes[sel], y_raw[sel], s=6, color="gray", alpha=0.15,
                   linewidths=0, zorder=0, rasterized=True)

    for measure, label in labels.items():
        y = steps[measure].to_numpy()
        means = bin_means(y, np.arange(len(steps)))
        boot = _cluster_bootstrap(group_rows, n_boot, seed,
                                  lambda rows, y=y: bin_means(y, rows))
        lo, hi = np.nanpercentile(boot, [a, 100 - a], axis=0)

        show_fit = fit_form is not None and measure == "step_2d_cm"
        ax.errorbar(centers, means, 
                    yerr=np.clip([means - lo, hi - means], 0, None),
                    fmt="o", linestyle="none",
                    color=COLORS[measure], ms=5, lw=1.3, capsize=3, label=label,
                    zorder=3)
        if show_fit:
            slope, intercept = fits[(measure, fit_form)]
            ax.plot(grid, intercept + slope * NOISE_FORMS[fit_form](grid),
                    color=COLORS[measure], lw=1.3, label=fit_label[fit_form],
                    zorder=4)

    ax.set_xlabel("Number of spikes per bin")
    ax.set_ylabel("Mean step size [cm]")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(alpha=0.3, ls="--")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if owns_fig:
        if title:
            fig.suptitle(title)
        fig.tight_layout()
    return fig


def stepsize_threshold_prob(
    steps,
    thresh_less,
    thresh_greater,
    *,
    measure="step_2d_cm",
    n_bins=12,
    bin_edges=None,
    event_col="event",
    spikes_col="spikes",
    n_boot=5000,
    seed=0,
    ci=95,
):
    """Per-spike-count-bin probability that a step falls below / above a threshold.

    Uncertainty is a cluster (whole-event) percentile bootstrap, matching
    `stepsize_noise_slope` and `plot_stepsize_noise_slope`. Percentiles are used
    rather than a bootstrap SD because p_less / p_greater are proportions bounded
    in [0, 1]: at large spike counts p_less approaches 0, where the bootstrap
    distribution is strongly skewed and a symmetric +/- SD band has no
    meaningful coverage.
    """
    if not (thresh_less < thresh_greater):
        raise ValueError(
            f"thresh_less must be < thresh_greater, got {thresh_less} and {thresh_greater}."
        )
    if not (0 < ci < 100):
        raise ValueError(f"ci must be in (0, 100), got {ci}.")
    for col in (spikes_col, measure, event_col):
        if col not in steps.columns:
            raise KeyError(f"{col!r} not in steps.columns: {list(steps.columns)}")

    spikes = steps[spikes_col].to_numpy(dtype=float)
    x = steps[measure].to_numpy(dtype=float)
    events = steps[event_col].to_numpy()

    ok = np.isfinite(spikes) & np.isfinite(x)
    spikes, x, events = spikes[ok], x[ok], events[ok]

    # Fixed bins from the full sample; resamples are assigned to these same bins.
    if bin_edges is None:
        bin_edges = np.unique(np.quantile(spikes, np.linspace(0, 1, n_bins + 1)))
    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bin = len(bin_edges) - 1
    if n_bin < 1:
        raise ValueError("bin_edges must define at least one bin.")
    bin_id = np.clip(np.digitize(spikes, bin_edges[1:-1]), 0, n_bin - 1)

    def per_bin_probs(rows):
        b, xr = bin_id[rows], x[rows]
        p_less = np.full(n_bin, np.nan)
        p_greater = np.full(n_bin, np.nan)
        for k in range(n_bin):
            m = b == k
            if m.any():
                xk = xr[m]
                p_less[k] = np.mean(xk < thresh_less)
                p_greater[k] = np.mean(xk > thresh_greater)
        return p_less, p_greater

    p_less_hat, p_greater_hat = per_bin_probs(np.arange(spikes.size))
    centers = np.array([
        spikes[bin_id == k].mean() if (bin_id == k).any() else np.nan
        for k in range(n_bin)
    ])
    counts = np.array([int((bin_id == k).sum()) for k in range(n_bin)])

    a = (100 - ci) / 2
    if n_boot:
        group_rows = _event_row_groups(events, sort=False)
        boot = _cluster_bootstrap(
            group_rows, n_boot, seed,
            lambda rows: np.concatenate(per_bin_probs(rows)),
        )
        lo, hi = np.nanpercentile(boot, [a, 100 - a], axis=0)
        p_less_lo, p_greater_lo = lo[:n_bin], lo[n_bin:]
        p_less_hi, p_greater_hi = hi[:n_bin], hi[n_bin:]
    else:
        boot = None
        nan_bins = np.full(n_bin, np.nan)
        p_less_lo = p_less_hi = p_greater_lo = p_greater_hi = nan_bins

    summary = pd.DataFrame({
        "spike_center": centers,
        "p_less": p_less_hat,
        "p_less_lo": p_less_lo,
        "p_less_hi": p_less_hi,
        "p_greater": p_greater_hat,
        "p_greater_lo": p_greater_lo,
        "p_greater_hi": p_greater_hi,
        "n_steps": counts,
    })
    summary.attrs["thresh_less"] = float(thresh_less)
    summary.attrs["thresh_greater"] = float(thresh_greater)
    summary.attrs["measure"] = measure
    summary.attrs["bin_edges"] = bin_edges
    summary.attrs["ci"] = float(ci)
    return summary, boot


def plot_stepsize_combined(
    summary_diff,
    boot_diff,
    summary_slope,
    steps_slope,
    *,
    whisker_ymax=None,
    fit_form="inv_n",
    n_bins=8,
    n_boot=5000,
    seed=0,
    ci=95,
    figsize=(15, 8.4),
):
    """
    a, b: plot_stepsize_2d_vs_1d_diff (row 1)
    c:    plot_stepsize_noise_slope (row 2, full width)
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_stepsize_2d_vs_1d_diff(
        summary_diff, boot_diff, whisker_ymax=whisker_ymax, axes=(ax_a, ax_b)
    )
    plot_stepsize_noise_slope(
        summary_slope, steps_slope, fit_form=fit_form, n_bins=n_bins,
        n_boot=n_boot, seed=seed, ci=ci, ax=ax_c, ylim=(0, 40), xlim=(0, 50), scatter=10_000
    )

    # Finalize the layout before reading axes positions, otherwise the
    # figure-fraction coordinates below would be computed pre-layout and
    # drift once tight_layout runs.
    fig.tight_layout()

    pad_x, pad_y = 0.045, 0.015
    pos_a, pos_b, pos_c = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
    # "a" and "c" share panel a's left edge (x0); "b" keeps its own column's
    # left edge, since it isn't meant to align with anything else.
    label_specs = [
        (pos_a.x0 - pad_x, pos_a.y1 + pad_y),
        (pos_b.x0 - pad_x, pos_b.y1 + pad_y),
        (pos_a.x0 - pad_x, pos_c.y1 + pad_y),
    ]
    for (x, y), letter in zip(label_specs, "abc"):
        fig.text(x, y, letter, fontsize=16, fontweight="bold", va="bottom", ha="right")

    return fig


def plot_stepsize_threshold_prob(
    summary,
    *,
    show_ci=True,
    title=None,
    ax=None,
):
    """Threshold-crossing probability vs spike count, with bootstrap CI bands.

    The band is the percentile interval stored in `summary` by
    `stepsize_threshold_prob` (level in `summary.attrs["ci"]`), so it is already
    inside [0, 1] by construction and needs no clipping.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
    else:
        fig = ax.figure

    tl = summary.attrs["thresh_less"]
    tg = summary.attrs["thresh_greater"]
    ci = summary.attrs.get("ci")

    order = np.argsort(summary["spike_center"].to_numpy())
    take = lambda col: summary[col].to_numpy()[order]
    x = take("spike_center")
    pl, pl_lo, pl_hi = take("p_less"), take("p_less_lo"), take("p_less_hi")
    pg, pg_lo, pg_hi = take("p_greater"), take("p_greater_lo"), take("p_greater_hi")

    ax.plot(x, pl, color=COLORS["c_less"], lw=2.2, label=f"Step size\n< {tl} cm")
    ax.plot(x, pg, color=COLORS["c_greater"], lw=2.2, label=f"Step size\n> {tg} cm")
    if show_ci and np.isfinite(pl_lo).any():
        band = f" ({ci:g}% CI)" if ci is not None else ""
        ax.fill_between(x, pl_lo, pl_hi, color=COLORS["c_less"],
                        alpha=0.25, edgecolor="none", label=f"_less{band}")
        ax.fill_between(x, pg_lo, pg_hi, color=COLORS["c_greater"],
                        alpha=0.25, edgecolor="none", label=f"_greater{band}")

    ax.set_xlabel("Number of spikes")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(frameon=True, fontsize=9, loc="lower right")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_spearman_grid(
    panels,
    plot_fn,
    ncols=3,
    figsize=None,
    sharey=False,
    label_offset=(-0.12, 1.08),
    label_fontsize=16,
    label_ha="left",
    save_path=None,
):
    import matplotlib.pyplot as plt
    import string

    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.5 * ncols, 4.5 * nrows)
 
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes = np.atleast_1d(axes).flatten()
 
    letters = string.ascii_lowercase
    li = 0
    for i, panel_kwargs in enumerate(panels):
        if panel_kwargs is None:          # deliberate gap in the grid
            axes[i].axis("off")
            continue
 
        kwargs = dict(panel_kwargs)       # copy, don't mutate caller's dict
        ylim = kwargs.pop("ylim", None)
 
        plot_fn(ax=axes[i], **kwargs)
        if ylim is not None:
            axes[i].set_ylim(*ylim)
 
        axes[i].text(
            *label_offset,
            letters[li],
            transform=axes[i].transAxes,
            fontsize=label_fontsize,
            fontweight="bold",
            va="top",
            ha=label_ha,
        )
        li += 1
 
    for j in range(n, len(axes)):         # unused trailing axes
        axes[j].axis("off")
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes
