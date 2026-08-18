"""
Combined simulation script  (REVISION: parameters adjusted to P&F data).

  python sim_sequence_percell_rates.py <n_cells> <mode>
    mode = 'smooth'    -> constant-velocity trajectory
    mode = 'speedmod'  -> oscillation-phase-locked speed modulation

What I changed in this revision:
  1. Binning matches P&F: 20 ms decoding WINDOW, 5 ms step (overlap 0.75), and the
     per-step spike count is taken over the window+step = 25 ms span, exactly as our
     data-analysis notebook counts spikes per decoding step.
  2. TODO: Generative rate: the Poisson mean is peak * dt * virtualPF. Previously it was 
     with an extra factor exp(-dt * virtualPF) 
  3. Zero-spike handling: a window is drawn ONCE; if it has zero spikes its decode is left NaN
     (like P&F Analysis). The old `while sum(n_spikes)==0: redraw` resampled the same phase 
     until non-zero.
  5. Rate SCALE: behavioral peak rates keep their shape (lognormal); a single
     replay_gain scalar multiplies every cell's rate (so it does NOT affect
     modulation depth or step-size statistics). replay_gain is now a hardcoded
     constant in sim_params_seq.py (params_id 15/16 only), originally obtained
     by calibrating so the mean spikes per 25 ms step matched target_spikes_per_step
     at n_cells=100. It is NOT re-calibrated at runtime -- if you change n_cells
     substantially, the realized spikes/step will drift from that target.
  6. n_osc_cycles is derived as f_gamma * T_sim. 
  7. All draws use one seeded Generator (rate_seed) so runs are reproducible.
  8. (Change undone, because Lorenz wanted increasing spike counts) 
     Number of cells can be set as a command-line argument (default 100). Gain should handle
     scaling to higher cell counts without hand-tuning the rate parameters to match P&F data.

NOTE: In contrast to P&F analysis the decoders are population vectors, not Bayesian decoders. 
"""

from pylab import *
import numpy as np
import scipy.stats as st
import pickle
import os
from sys import argv

from sim_params_seq import (
    params_id, n_grid_cm, DeltaT_window_sec, overlap, n_windows, n_reps,
    n_osc_cycles, sigma_code_cm, x_step_cm, y_step_cm, n_mincells,
    mean_peak_Hz, sigma_peak_Hz, rate_seed,
    pf_width_mode, pf_width_std_cm, pf_width_logn_mean, pf_width_logn_sigma,
    f_gamma_Hz, target_spikes_per_step, replay_max_rate_Hz, replay_gain, gamma_mod_depth
)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("n_cells", nargs="?", type=int, default=100)
_ap.add_argument("mode", nargs="?", default="smooth", choices=("smooth", "speedmod"))
_ap.add_argument("--run", type=int, default=None,
                 help="append _run<N> to output file names; None (default) -> no suffix")
_args = _ap.parse_args()
n_cells = _args.n_cells
mode    = _args.mode
print(f"mode={mode} | n_cells={n_cells} | params_id={params_id}")

# --- seed -------------------------------------------------------------------
# rate_seed is imported from sim_params_seq. Set it to None draw fresh random seed each time.
_seed = rate_seed
if _seed is None:
    _seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"[seed] rate_seed=None -> drew random seed = {_seed}")
else:
    print(f"[seed] using rate_seed = {_seed}")

rng   = np.random.default_rng(_seed)

# do explicit print-outs for debugging
VERBOSE = True

# ---------------------------------------------------------------------------
# per-cell BEHAVIORAL peak firing rates (shape from data; scale set by replay_gain)
if sigma_peak_Hz is None:
    if VERBOSE:
        print("Peak rates: exponential (sigma_peak_Hz=None)")
    max_rate_per_cell = rng.exponential(scale=mean_peak_Hz, size=n_cells)
else:
    max_rate_per_cell = rng.lognormal(mean=mean_peak_Hz, sigma=sigma_peak_Hz, size=n_cells)
    if VERBOSE:
        print(f"Peak rates: lognormal. median={np.median(max_rate_per_cell):.2f} Hz, "
              f"mean={max_rate_per_cell.mean():.2f} Hz")

# ---------------------------------------------------------------------------
# per-cell place-field widths  ->  sim-space sigma_code  (exp(-d^2 / sigma_code^2))
#   fitted_std is the std of a fit using exp(-d^2 / 2 sigma^2)  =>  sigma_code = sqrt(2)*std
# ---------------------------------------------------------------------------
if pf_width_mode == "single":
    sigma_code_per_cell = np.full(n_cells, pf_width_std_cm)
    if VERBOSE:
        print(f"PF width: single value. fitted_std={pf_width_std_cm:.2f} cm "
            f"-> sigma_code={pf_width_std_cm:.2f} cm")
elif pf_width_mode == "lognormal":
    if pf_width_logn_mean is None or pf_width_logn_sigma is None:
        raise ValueError(
            "pf_width_mode='lognormal' but pf_width_logn_mean/pf_width_logn_sigma are "
            "not set. Fit a lognormal to the open-field place-field widths and fill both "
            "params in sim_params_seq.py (id 18) before running."
        )
    fitted_std = rng.lognormal(mean=pf_width_logn_mean, sigma=pf_width_logn_sigma, size=n_cells)
    sigma_code_per_cell = fitted_std
    if VERBOSE:
        print(f"PF width: lognormal per-cell. fitted_std median={np.median(fitted_std):.2f} cm "
            f"-> sigma_code median={np.median(sigma_code_per_cell):.2f} cm")
else:
    raise ValueError(f"unknown pf_width_mode={pf_width_mode!r}")

speed_mod_depth = 1.0  # only used in 'speedmod' mode

# place-field centres (seeded)
mu_x = n_grid_cm * rng.random(n_cells)
mu_y = n_grid_cm * rng.random(n_cells)

# ---------------------------------------------------------------------------
# binning  (P&F: 20 ms window, 5 ms step  ->  n_increments = 4, count span = 25 ms)
# see sim_params_seq.py for the parameters that define this binning (DeltaT_window_sec, overlap, n_windows)
# ---------------------------------------------------------------------------
increment_fraction = 1 - overlap
increment_sec      = increment_fraction * DeltaT_window_sec
n_increments       = int(round(1.0 / increment_fraction))
frac_mincells      = n_mincells / float(n_cells)
span_increments    = n_increments + 1
x_start_cm = (n_grid_cm - (n_windows - 1 - n_increments) * x_step_cm) / 2
y_start_cm = x_start_cm  # square arena, same start for x and y


if VERBOSE:
    print(f"window={DeltaT_window_sec*1e3:.0f} ms | step={increment_sec*1e3:.0f} ms | "
        f"n_increments={n_increments} | count span={span_increments*increment_sec*1e3:.0f} ms")
    print(f"gamma cycles per count span (sim) = {span_increments*n_osc_cycles/float(n_windows):.3f} "
        f"| data target ~ f_gamma*0.025 = {f_gamma_Hz*0.025:.3f}")


def osc_modulation_at(phase):
    """Rate-modulation envelope; Michelson depth of this term == gamma_mod_depth."""
    baseline = 0.5 * (1 + frac_mincells)
    return baseline * (1 + gamma_mod_depth * cos(phase))


def run_sim(replay_gain, n_reps_run, seed):
    """One full set of repetitions. Returns per-window and per-step arrays.

    Per-step quantities (n_windows-1 steps) use the 25 ms span and the
    spike-count-weighted circular-mean oscillation phase over that span, mirroring
    the notebook's per-step spike count and circular-mean gamma phase.
    """
    rs       = np.random.default_rng(seed)
    rs_phase = np.random.default_rng(seed + 10000)
    n_burn   = n_increments          # throwaway front windows = ring depth
    x_true   = np.full((n_reps_run, n_windows), np.nan)
    y_true   = np.full((n_reps_run, n_windows), np.nan)
    x_mean   = np.full((n_reps_run, n_windows), np.nan)
    y_mean   = np.full((n_reps_run, n_windows), np.nan)
    win_spk  = np.full((n_reps_run, n_windows), np.nan)      # 20 ms window total
    std_arr  = np.zeros((n_reps_run, n_windows))
    step_spk = np.full((n_reps_run, n_windows - 1), np.nan)  # 25 ms span total
    step_phs = np.full((n_reps_run, n_windows - 1), np.nan)  # deg, [0,360)

    for i_reps in range(n_reps_run):
        ring      = np.zeros((n_cells, n_increments))        # per-cell, per-increment (decode)
        inc_total = []                                       # flat per-increment spike totals
        inc_phase = []                                       # flat per-increment phase (rad)

        off_x = 10 * (rs.random() - 0.5)
        off_y = 10 * (rs.random() - 0.5)
        # speedmod integrates forward: back the start up by the burn-in distance so
        # the FIRST KEPT window (i_wind=0) lands near x_start (smooth ignores x_old).
        x_old = x_start_cm + off_x - n_burn * x_step_cm
        y_old = y_start_cm + off_y - n_burn * y_step_cm
        phi0 = rs_phase.uniform(0.0, 2*np.pi)

        for i_wind in range(-n_burn, n_windows):            # i_wind < 0 = throwaway burn-in
            # Phase from real elapsed time: 2*pi * f_gamma * t, with t = i_wind * increment_sec.
            # The clock runs continuously through the burn-in; kept window 0 sits at t = 0.
            phase = 2 * pi * f_gamma_Hz * (i_wind * increment_sec) + phi0

            if mode == "smooth":
                x_t = x_start_cm + i_wind * x_step_cm + off_x
                y_t = y_start_cm + i_wind * y_step_cm + off_y
            else:  # speedmod
                sm  = 1.0 - speed_mod_depth * cos(phase)
                x_t = x_old + x_step_cm * sm
                y_t = y_old + y_step_cm * sm
                x_old, y_old = x_t, y_t
            if i_wind >= 0:                                  # record only the kept region
                x_true[i_reps, i_wind] = x_t
                y_true[i_reps, i_wind] = y_t

            virtualPF = exp(-((x_t - mu_x) ** 2 + (y_t - mu_y) ** 2) / sigma_code_per_cell ** 2)
            # Poisson MEAN per increment = peak * gain * mod * dt * PF
            # (no exp(-dt*PF): that decoder term does not belong in the generator)
            rate = (max_rate_per_cell * replay_gain * osc_modulation_at(phase)
                    * increment_sec * virtualPF)

            # ---- ALTERNATIVE rate anchor (discuss with co-authors) -------------
            # Keep the lognormal *shape* but cap the instantaneous rate at the measured
            # (rare) replay maximum instead of scaling behavioral peaks by replay_gain:
            #   rate = (np.minimum(max_rate_per_cell, replay_max_rate_Hz)
            #           * osc_modulation_at(phase) * increment_sec * virtualPF)
            # --------------------------------------------------------------------

            if i_wind == -n_burn:                            # fill once, inside the throwaway region
                for j in range(n_increments):
                    ns = rs.poisson(rate)
                    ring[:, j % n_increments] = ns
                    inc_total.append(ns.sum()); inc_phase.append(phase)
            else:                                            # advance by one increment
                ns = rs.poisson(rate)
                ring[:, i_wind % n_increments] = ns          # overwrite oldest
                inc_total.append(ns.sum()); inc_phase.append(phase)

            weights = ring.sum(1)
            wtot    = weights.sum()
            if i_wind >= 0:                                  # record only the kept region
                win_spk[i_reps, i_wind] = wtot
                if wtot > 0:                                 # else: skip (decode left NaN)
                    mx = np.average(mu_x, weights=weights)
                    my = np.average(mu_y, weights=weights)
                    x_mean[i_reps, i_wind] = mx
                    y_mean[i_reps, i_wind] = my
                    sx = sqrt(np.average((mu_x - mx) ** 2, weights=weights))
                    sy = sqrt(np.average((mu_y - my) ** 2, weights=weights))
                    std_arr[i_reps, i_wind] = sqrt(sx * sy)

        inc_total = np.asarray(inc_total, dtype=float)
        inc_phase = np.asarray(inc_phase, dtype=float)
        # window m increments = inc[m : m+n_increments]; step m span = inc[m : m+n_increments+1]
        for s in range(n_windows - 1):
            sl  = slice(s + n_burn, s + n_burn + span_increments)   # +n_burn: skip burn-in increments
            tot = inc_total[sl]
            step_spk[i_reps, s] = tot.sum()
            if tot.sum() > 0:                                # spike-weighted circular mean phase
                ang = np.arctan2((tot * np.sin(inc_phase[sl])).sum(),
                                 (tot * np.cos(inc_phase[sl])).sum())
                step_phs[i_reps, s] = np.rad2deg(ang) % 360.0

    return dict(x_true=x_true, y_true=y_true, x_mean=x_mean, y_mean=y_mean,
                win_spk=win_spk, std=std_arr, step_spk=step_spk, step_phs=step_phs)


# ---------------------------------------------------------------------------
# full run  (replay_gain is now a hardcoded constant from sim_params_seq.py,
# not calibrated at runtime -- see the comment next to replay_gain there for
# provenance and the n_cells caveat)
# ---------------------------------------------------------------------------
if VERBOSE:
    print(f"\nreplay_gain = {replay_gain:.3f} (hardcoded, params_id={params_id})")
    print(f"  => effective median peak rate = {np.median(max_rate_per_cell)*replay_gain:.1f} Hz "
        f"(behavioral median {np.median(max_rate_per_cell):.1f} Hz x {replay_gain:.2f})")
    print(f"  (for reference, measured replay max ~ {replay_max_rate_Hz:.0f} Hz, but rare)")

print(f"\nRunning full simulation ({n_reps} reps) ...")
res = run_sim(replay_gain, n_reps, seed=_seed + 1)
x_true_array_cm = res["x_true"]; y_true_array_cm = res["y_true"]
x_mean_array_cm = res["x_mean"]; y_mean_array_cm = res["y_mean"]
tot_spikes      = res["win_spk"]                       # kept name for downstream/save
step_spikes     = res["step_spk"]
step_phase      = res["step_phs"]
realized_step   = np.nanmean(step_spikes)
print(f"  realized mean spikes / 25 ms step = {realized_step:.2f} "
      f"(target {target_spikes_per_step:.1f}, observed 16-21)")

repdim, window_dim = 0, 1

# ---------------------------------------------------------------------------
# file name
# ---------------------------------------------------------------------------
mode_tag  = mode if mode == "smooth" else f"speedmod{speed_mod_depth}"
file_name = (f"percellrates_sim_seq_mode-{mode_tag}"
             f"_paramsid_{params_id}_{n_reps}reps_{int(n_mincells)}mincells"
             f"_{n_cells}cells_{n_windows}steps"
             f"_{gamma_mod_depth}gammamod_{mean_peak_Hz}meanHz")
if _args.run is not None:
    file_name = f"{file_name}_run{_args.run}"


tab_red = "#E57373"
fsize, fsize_small, linew, msize = 6, 4, 1.5, 3
n_plots_x, n_plots_y = 3, 2
figure(figsize=(8, 6), dpi=300)

# Plot 1 - sample decoded vs true trajectory (rep 0)
subplot(n_plots_x, n_plots_y, 1)
t = arange(0, 2 * pi, 0.01 * pi)
std_array_cm = res["std"][0]
R = np.zeros(n_windows)
for i_wind in range(n_windows):
    tot = tot_spikes[0, i_wind]
    if isfinite(tot) and tot > 1:
        t_conf_lo, _ = st.t.interval(0.95, tot - 1, 0, 1)
        R[i_wind] = abs(t_conf_lo * std_array_cm[i_wind] / sqrt(tot))
for i_wind in range(n_windows):
    if isfinite(x_mean_array_cm[0, i_wind]):
        plot(x_mean_array_cm[0, i_wind] + R[i_wind] * cos(t),
             y_mean_array_cm[0, i_wind] + R[i_wind] * sin(t),
             color=str(np.clip(tot_spikes[0, i_wind] / np.nanmax(tot_spikes), 0, 1)), lw=linew)
plot(x_true_array_cm[0, :], y_true_array_cm[0, :],
     "k|-" if mode == "smooth" else "k.-", markersize=5, lw=1)
plot(x_mean_array_cm[0, :], y_mean_array_cm[0, :], ".-", color=tab_red, markersize=2.5)
axis([0, n_grid_cm, 0, n_grid_cm])
xlabel("x [cm]", fontsize=fsize); ylabel("y [cm]", fontsize=fsize)
title(f"Sample, n={n_cells} cells ({mode})", fontsize=fsize)
ax = gca()
for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontsize(fsize)

# per-step displacement (decoded)
dx = diff(x_mean_array_cm, window_dim)
dy = diff(y_mean_array_cm, window_dim)
L2_step = sqrt(dx ** 2 + dy ** 2)

# Plot 2 - step size vs spikes/step (25 ms span)
subplot(n_plots_x, n_plots_y, 2)
plot(step_spikes.ravel(), L2_step.ravel(), "r.", markersize=msize)
xlabel("Spikes / 25 ms step", fontsize=fsize); ylabel("Estim. step size [cm]", fontsize=fsize)
ax = gca()
for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontsize(fsize)

# Plot 3 - spikes/step vs oscillation phase (binned)
subplot(n_plots_x, n_plots_y, 3)
def phase_binned_mean(phase_deg, vals, n_bins=36, bw=10):
    edges = np.arange(0, n_bins * bw + 1, bw)
    idx   = np.clip(np.digitize(phase_deg, edges) - 1, 0, n_bins - 1)
    out   = np.full(n_bins, np.nan)
    finite = np.isfinite(vals) & np.isfinite(phase_deg)
    for b in range(n_bins):
        m = finite & (idx == b)
        if m.any(): out[b] = vals[m].mean()
    return out, (edges[:-1] + bw / 2.0)
binned_spk, centres = phase_binned_mean(step_phase.ravel(), step_spikes.ravel())
bar(centres, binned_spk, width=10, color="0.7", edgecolor="white")
axhline(np.nanmean(binned_spk), color="k", lw=1, ls="--")
xlabel("Gamma phase [deg]", fontsize=fsize); ylabel("Spikes / step", fontsize=fsize)
xlim(0, 360)
ax = gca()
for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontsize(fsize)

# Plot 4 - step size vs oscillation phase (binned)
subplot(n_plots_x, n_plots_y, 4)
binned_step, _ = phase_binned_mean(step_phase.ravel(), L2_step.ravel())
bar(centres, binned_step, width=10, color="0.7", edgecolor="white")
axhline(np.nanmean(binned_step), color="k", lw=1, ls="--")
xlabel("Gamma phase [deg]", fontsize=fsize); ylabel("Step size [cm]", fontsize=fsize)
xlim(0, 360)
ax = gca()
for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontsize(fsize)

# Plot 5 - true vs estimated step size across windows
subplot(n_plots_x, n_plots_y, 5)
true_dx = mean(diff(x_true_array_cm, window_dim), repdim)
true_dy = mean(diff(y_true_array_cm, window_dim), repdim)
xax = x_start_cm + x_step_cm * arange(1, n_windows)
plot(xax, sqrt(true_dx ** 2 + true_dy ** 2), label="True step")
plot(xax, np.nanmean(L2_step, repdim), label="Mean estim. step")
plot(x_start_cm + x_step_cm * arange(n_windows),
     10 * np.nanmean(tot_spikes, repdim) / np.nanmax(tot_spikes), label="spikes (scaled)")
ylabel("Step size [cm]", fontsize=fsize); xlabel("x [cm]", fontsize=fsize)
legend(fontsize=fsize_small)
ax = gca()
for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontsize(fsize)

# Plot 6 - histogram of spikes/step
subplot(n_plots_x, n_plots_y, 6)
hist(step_spikes[np.isfinite(step_spikes)].ravel(), 20)
xlabel("Spikes / 25 ms step", fontsize=fsize); ylabel("Count", fontsize=fsize)
ax = gca()
for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontsize(fsize)

tight_layout()
os.makedirs("figs", exist_ok=True)
savefig("figs/plot_" + file_name + ".png", dpi=300)


# spikes vs gamma phase + modulation depth (should be the same as in the notebook)
def smooth_two_cycles(binned, sw=8, n_bins=36):
    tiled = np.tile(np.nan_to_num(binned, nan=np.nanmean(binned)), 4)
    sm    = np.convolve(tiled, np.ones(sw) / sw, mode="same")
    return sm[n_bins:-n_bins]

def michelson(curve):
    c = curve[np.isfinite(curve)]
    if c.size == 0 or (c.max() + c.min()) == 0:
        return np.nan
    return (c.max() - c.min()) / (c.max() + c.min())

mod_depth = michelson(binned_spk)
smoothed  = smooth_two_cycles(binned_spk)

figure(figsize=(6, 3.5), dpi=300)
xx = np.arange(72)
bar(xx, np.tile(binned_spk, 2), width=1.0, color="0.7", edgecolor="white")
plot(xx, smoothed, color="r", lw=3)
axhline(np.nanmean(binned_spk), color="k", lw=1, ls="--")
xlabel("Gamma phase (10 deg bins, two cycles)", fontsize=fsize)
ylabel("Mean spikes per step", fontsize=fsize)
title(f"Spikes vs gamma phase (n={n_cells}, {n_reps} reps, {mode})\n"
      f"Michelson modulation depth = {mod_depth*100:.2f}%  |  gamma_mod = {gamma_mod_depth}",
      fontsize=fsize)
ax = gca()
for lab in ax.get_xticklabels() + ax.get_yticklabels(): lab.set_fontsize(fsize)
tight_layout()
savefig(f"figs/phase_binned_spikes_{file_name}.png", dpi=300)
print(f"\nMichelson modulation depth (raw 36-bin) = {mod_depth*100:.2f}%")

# check PF coverage
arena_area_cm2 = float(n_grid_cm) ** 2
n_eff    = n_cells * np.pi * np.mean(sigma_code_per_cell ** 2) / arena_area_cm2
cv_peak  = max_rate_per_cell.std() / max_rate_per_cell.mean()
cv_cover = np.sqrt((1.0 + cv_peak ** 2) / (2.0 * n_eff))
if VERBOSE:
    print(f"\nPF coverage metrics:")
    print(f"  Effective number of place fields: {n_eff:.1f}")
    print(f"  Coefficient of variation (peak rates): {cv_peak:.2f}")
    print(f"  Coefficient of variation (coverage): {cv_cover:.2f}")
    print(np.nanmean(tot_spikes), np.nanstd(tot_spikes))
    
# save
os.makedirs("data", exist_ok=True)

# oh god leave as it is to not break the analysis downstream
_LABEL8 = ("1. x_true_array_cm, 2. y_true_array_cm, 3. x_mean_array_cm, "
           "4. y_mean_array_cm, 5. tot_spikes, 6. mu_x, 7. mu_y, "
           "8. max_rate_per_cell")

with open("data/" + file_name + ".txt", "wb") as f:
    for obj in (x_true_array_cm, y_true_array_cm, x_mean_array_cm, y_mean_array_cm,
                tot_spikes, mu_x, mu_y, max_rate_per_cell):
        pickle.dump(obj, f, 0)
    pickle.dump(_LABEL8, f, 0)

np.save("data/" + file_name + "_npformat.npy",
        np.array((x_true_array_cm, y_true_array_cm, x_mean_array_cm, y_mean_array_cm,
                  tot_spikes, mu_x, mu_y, max_rate_per_cell, _LABEL8),
                 dtype=object), allow_pickle=True)

# ------------------------------------------------------------------------------
# --- NEW: step-level (25 ms span) data + calibration, in a separate sidecar ---
np.save("data/" + file_name + "_stepdata.npy",
        np.array((step_spikes, step_phase, sigma_code_per_cell,
                  np.float64(replay_gain), np.float64(realized_step),
                  "1. step_spikes(25ms) [reps x n_windows-1], 2. step_phase_deg, "
                  "3. sigma_code_per_cell, 4. replay_gain, 5. realized_spikes_per_step"),
                 dtype=object), allow_pickle=True)

print("Done")
