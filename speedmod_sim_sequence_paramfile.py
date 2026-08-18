from pylab import *
import numpy as np
import scipy.stats as st
import new_colormaps as nc
#from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle

from sim_params_seq import params_id, n_grid_cm, DeltaT_window_sec, overlap, n_windows, n_reps, n_osc_cycles, sigma_code_cm, x_step_cm, y_step_cm, max_rate_Hz, n_mincells, gamma_mod_depth
from sys import argv

try:
    from sim_params_seq import rate_seed
except ImportError:
    rate_seed = None   # params_id blocks that predate seeding -> fresh random seed each run

# --- seed -------------------------------------------------------------------
# rate_seed is imported from sim_params_seq. Set it to None to draw a fresh random seed each time.
_seed = rate_seed
if _seed is None:
    _seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"[seed] rate_seed=None -> drew random seed = {_seed}")
else:
    print(f"[seed] using rate_seed = {_seed}")

rng = np.random.default_rng(_seed)          # place-field centres (as in sim_sequence_percell_rates.py)
rs  = np.random.default_rng(_seed + 1)      # simulation loop: start offsets + Poisson draws

# See color codes at https://cdn.elifesciences.org/author-guide/tables-colour.pdf

tab_blue   = '#90CAF9'
tab_green  = '#C5E1A5'
tab_orange = '#FFB74D'
tab_yellow = '#FFF176'
tab_purple = '#9E86C9'
tab_red    = '#E57373'
tab_pink   = '#F48FB1'
tab_grey   = '#E6E6E6'

ion()


figure(figsize=(5, 4), dpi=300) # 2x2 plot layout

fsize = 6
fsize_small = 4#.5
linew = 1.5
msize = 3 # 30 # 8

x_start_cm = 10 # 25
y_start_cm = 10 # 25


x_grid, y_grid = meshgrid(range(n_grid_cm), range(n_grid_cm))


n_plots_x = 3
n_plots_y = 2

x_true_array_cm = nan * ones([n_reps, n_windows])
y_true_array_cm = nan * ones([n_reps, n_windows])
x_mean_array_cm = nan * ones([n_reps, n_windows])
y_mean_array_cm = nan * ones([n_reps, n_windows])
tot_spikes      = nan * ones([n_reps, n_windows])
repdim, window_dim = (0, 1)

std_array_cm    = zeros(n_windows)
R               = zeros(n_windows)

n_cells: int = int(argv[1]) if len(argv) > 1 else 100   # argv[0] is the script name, so `if argv` was always True
mu_x = n_grid_cm * rng.random(n_cells) # Fixed PF center locations
mu_y = n_grid_cm * rng.random(n_cells)
speed_mod_depth = 1.0  # Could be too large


increment_fraction = 1 - overlap
increment_sec = increment_fraction * DeltaT_window_sec
n_increments = int(round(1.0 / increment_fraction))
# Throwaway front windows = ring-buffer depth. The i_wind == -n_burn fill puts n_increments
# draws at a SINGLE phase/position into the buffer; running from -n_burn guarantees every
# slot has been overwritten by a real increment before kept window 0 is recorded.
n_burn = n_increments
print(f"n_increments = {n_increments} | discarding {n_burn} burn-in window(s) before window 0")


subplot(n_plots_x, n_plots_y, 1)


for i_reps in range(n_reps):
    print("i_reps =", i_reps)

    call_counter = 0

    n_spikes_ringbuffer = nan * ones([n_cells, n_increments])

    rand_offset_x_cm = 10*(rs.random()-0.5) # [-5cm, 5cm]
    rand_offset_y_cm = 10*(rs.random()-0.5)

    # This mode integrates the trajectory forward, so back the start up by the burn-in
    # distance to make the FIRST KEPT window (i_wind = 0) land near x_start_cm again.
    x_old_cm = x_start_cm + rand_offset_x_cm - n_burn * x_step_cm
    y_old_cm = y_start_cm + rand_offset_y_cm - n_burn * y_step_cm

    for i_wind in range(-n_burn, n_windows):   # i_wind < 0 = throwaway burn-in
    
        frac_mincells = n_mincells / float(n_cells)
        osc_phase     = i_wind / float(n_windows) * n_osc_cycles * 2 * pi

        baseline       = 0.5 * (1 + frac_mincells)
        osc_modulation = baseline * (1 + gamma_mod_depth * cos(osc_phase))
        peak_rate_Hz = max_rate_Hz * osc_modulation

        #speed_modulation = (1 - 0.5*cos(osc_phase)) # in range [0.5, 1.5]; phase-inverted relative to the pop. oscillation (0.5x speed at peak activity, 1.5x speed at min. activity)
        # More general:
        speed_modulation = (1.0 - speed_mod_depth * cos(osc_phase)) # in range [1 - mod_depth, 1 + mod_depth]
        # Make sure that the modulation across windows sums to a value of 1, or the sequence will overshoot!
        # Local position: with i_wind < 0 an array write would land at the END of the array.
        x_t = x_old_cm + x_step_cm * speed_modulation
        y_t = y_old_cm + y_step_cm * speed_modulation
        x_old_cm, y_old_cm = x_t, y_t
        if i_wind >= 0:                     # record only the kept region
            x_true_array_cm[i_reps, i_wind] = x_t
            y_true_array_cm[i_reps, i_wind] = y_t

        #print "x_true_array_cm[i_reps, i_wind] = ", x_true_array_cm[i_reps, i_wind]

        poi_rate = zeros(n_cells)
        n_spikes = zeros(n_cells)

        # Draw ONCE per window (no resample-until-nonzero). A zero-spike window is left
        # un-decoded (NaN), as in P&F analysis. Old behavior: `while sum(n_spikes) == 0: redraw`.
        for i_cell in range(n_cells):

            # Now, draw spikes from a Poisson distribution, with a rate determined by the "true" represented location relative to the place field center, scaled by sigma_code_cm:

            virtualPF_xytrue_cell_i = exp(-((x_t - mu_x[i_cell])**2 + (y_t - mu_y[i_cell])**2)/sigma_code_cm**2)
            #poi_rate_cell_i = peak_rate_Hz * DeltaT_window_sec * virtualPF_xytrue_cell_i
            poi_rate_cell_i = peak_rate_Hz * increment_sec * virtualPF_xytrue_cell_i
            poi_rate[i_cell] = poi_rate_cell_i

            # (removed) n_spikes[i_cell] = poisson(poi_rate_cell_i)
            # This draw was overwritten a few lines below by the vectorised draw, so it only
            # consumed n_cells RNG values per window -- harmless when unseeded, but it would
            # desynchronise this script's stream from the smooth one now that the seed is fixed.


        if i_wind == -n_burn:
            # Ensure that the ring buffer is filled. All n_increments draws share one phase and
            # one position, which is why this now happens inside the throwaway burn-in region.
            for i_increments in range(n_increments):
                n_spikes = rs.poisson(poi_rate)
                n_spikes_ringbuffer[:, i_increments % n_increments] = n_spikes

        else:
            n_spikes = rs.poisson(poi_rate)
            #print "(i_wind + n_increments) % n_increments = ", (i_wind + n_increments) % n_increments
            n_spikes_ringbuffer[:, (i_wind + n_increments) % n_increments] = n_spikes

        if i_wind < 0:
            continue   # burn-in: buffer updated, but nothing is recorded or decoded

        #tot_spikes[i_reps, i_wind] = sum(n_spikes)
        tot_spikes[i_reps, i_wind] = sum(sum(n_spikes_ringbuffer))
        #print "tot_spikes = ", tot_spikes[i_reps, i_wind]


        x_mean, y_mean = mean(mu_x), mean(mu_y)
        #x_mean_array_cm[i_reps, i_wind], y_mean_array_cm[i_reps, i_wind] = x_mean, y_mean


        # Decode only if the window has spikes; otherwise leave x/y_mean at NaN (no resampling).
        if tot_spikes[i_reps, i_wind] > 0:
            #mu_x_mean = np.average(mu_x, weights=n_spikes)
            #mu_y_mean = np.average(mu_y, weights=n_spikes)
            mu_x_mean = np.average(mu_x, weights=sum(n_spikes_ringbuffer, 1))
            mu_y_mean = np.average(mu_y, weights=sum(n_spikes_ringbuffer, 1))

            x_mean_array_cm[i_reps, i_wind], y_mean_array_cm[i_reps, i_wind] = mu_x_mean, mu_y_mean

            #std_x = sqrt(np.average((mu_x - mu_x_mean)**2, weights=n_spikes))
            #std_y = sqrt(np.average((mu_y - mu_y_mean)**2, weights=n_spikes))
            std_x = sqrt(np.average((mu_x - mu_x_mean)**2, weights=sum(n_spikes_ringbuffer, 1)))
            std_y = sqrt(np.average((mu_y - mu_y_mean)**2, weights=sum(n_spikes_ringbuffer, 1)))

            std_xy = sqrt(std_x * std_y)
            std_array_cm[i_wind] = std_xy
        #print "std. [PF centers] (x,y)= ", std_xy


        #print "x_mean, y_mean [PF centers] = ", round(x_mean), round(y_mean)

        if i_reps==0:

            t=arange(0, 2*pi, 0.01*pi)
            if tot_spikes[i_reps, i_wind] > 0:
                # Konfidenzintervall zum Niveau alpha der t-Verteilung mit n-1 Freiheitsgraden:
                alpha=0.95
                #t_conf_lo, t_conf_up = st.t.interval(alpha, n_cells-1, 0, 1)
                t_conf_lo, t_conf_up = st.t.interval(alpha, tot_spikes[i_reps, i_wind]-1, 0, 1) # For Poisson activity
                # Konfidenzintervall eines normalverteilten Merkmals mit Stichprobenvarianz s^2:
                # [x_mean - t_conf * s / sqrt(n); x_mean + t_conf * s / sqrt(n) ] # Achtung, korrigierte Stichprobenvarianz benutzen!
                # CI_x_lower = x_mean + t_conf_lo * std_xy / sqrt(n_cells)

                #plot(mu_x, mu_y, '.', color='k', markersize=msize) # tab_blue
                #plot(mu_x, mu_y, '.', color=str(1 - n_cells / 200.0), markersize=msize) # tab_blue

                #plot(x_mean, y_mean, '+', markersize=msize, color='k')
                #R = t_conf_lo * std_xy / sqrt(n_cells)
                R[i_wind] = abs(t_conf_lo * std_array_cm[i_wind] / sqrt(tot_spikes[i_reps, i_wind]))
                #print "R = ", R[i_wind]
            else:
                R[i_wind] = nan  # empty bin: no spikes, no CI to compute

            #plot(mu_x_mean + R[i_wind] * cos(t), mu_y_mean + R[i_wind]*sin(t), color=tab_red, lw=linew) # CI around the sample mean

            #plot(x_mean + std_xy * cos(t), y_mean + std_xy*sin(t), color=tab_blue, lw=linew) # sample std # tab_green    

            axis([0, n_grid_cm, 0, n_grid_cm])
            ax=gca()
            ax.set_xticks([0, n_grid_cm])
            ax.set_xticklabels([0, n_grid_cm])
            ax.set_yticks([0, n_grid_cm])
            ax.set_yticklabels([0, n_grid_cm])
            for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                label.set_fontsize(fsize) # 5
            xlabel('x [cm]', fontsize=fsize)
            ylabel('y [cm]', fontsize=fsize)
            if i_wind==0:
                legend(('Place field center', '95% confidence area', 'spatial std.'), fontsize=fsize_small)#, loc=(0.05,-0.15)) # Sampled 
                title('Sample, \n n = '+str(n_cells)+' cells', fontsize=fsize)    
            else:
                title('n = '+str(n_cells)+' cells', fontsize=fsize)

            #axis('scaled')
            #ax=gca()
            #ax.set_aspect(1.0)


            nz_spikes = nonzero(n_spikes > 0)[0]
            for i_cell in nz_spikes:
                plot(mu_x[i_cell], mu_y[i_cell],  '.', color=str(osc_modulation), markersize=3)

    if i_reps==0:


        for i_wind in range(n_windows):
            plot(x_mean_array_cm[i_reps, i_wind] + R[i_wind] * cos(t), y_mean_array_cm[i_reps, i_wind] + R[i_wind]*sin(t), color=str(tot_spikes[i_reps, i_wind] / np.nanmax(tot_spikes) ), lw=linew) # CI around the sample mean




    
#tight_layout() # Creates sufficient spacing, but makes subplots smaller

            if i_wind == n_windows - 1:
                plot(x_true_array_cm[i_reps, :], y_true_array_cm[i_reps, :], 'k.-')
                plot(x_mean_array_cm[i_reps, :], y_mean_array_cm[i_reps, :], '.-', color=tab_red)


subplot(n_plots_x, n_plots_y, 3)
for i_wind in range(n_windows):
    if i_reps == 1:
        #bar(x_true_array_cm[i_reps, i_wind] - 0.5*x_step_cm, tot_spikes[i_reps, i_wind], width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * 4*pi)) )) 
        bar(i_wind * x_step_cm + x_start_cm, tot_spikes[i_reps, i_wind], width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * n_osc_cycles * 2*pi)) )) 
    else:
        #bar(x_true_array_cm[i_reps, i_wind] - 0.5*x_step_cm, nanmean(tot_spikes[:, i_wind]), width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * 4*pi)) ))
        bar(i_wind * x_step_cm + x_start_cm, nanmean(tot_spikes[:, i_wind]), width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * n_osc_cycles * 2*pi)) ))

ylabel('Spike count', fontsize=fsize)
axis([0, n_grid_cm, 0, np.nanmax(tot_spikes)])
ax=gca()
ax.set_aspect(0.2)
#ax.set_xticks([0, n_grid_cm])
#ax.set_xticklabels([0, n_grid_cm])
#ax.set_xticks([x_start_cm, x_start_cm + 0.5*n_windows * x_step_cm, x_start_cm + n_windows * x_step_cm])
xtick_locs = arange(  x_start_cm, int(x_start_cm + (n_windows+1) * x_step_cm),  (n_windows+1) * x_step_cm / float(n_osc_cycles) )
ax.set_xticks(xtick_locs)
ax.set_xticklabels(['0'] + ['{}$\pi$'.format(2*i) for i in range(1, len(xtick_locs))])
ymin = int(np.nanmin(tot_spikes)) - 1
ymax = int(np.nanmax(tot_spikes)) + 1
ax.set_yticks([ymin, ymax])
ax.set_yticklabels([ymin, ymax])
axis([0, n_grid_cm, ymin, ymax])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) # 5
xlabel('Oscillation phase', fontsize=fsize)

#'''#

subplot(n_plots_x, n_plots_y, 5)
true_xdiff_mean = mean( diff(x_true_array_cm, window_dim), repdim)
true_ydiff_mean = mean( diff(y_true_array_cm, window_dim), repdim)

L2_diff_perframe = sqrt( diff(x_mean_array_cm, window_dim)**2 + diff(y_mean_array_cm, window_dim)**2)
n_valid_reps_per_window = np.sum(np.isfinite(diff(x_mean_array_cm, window_dim)), axis=repdim)
mean_L2diff_acrossreps = np.nanmean(L2_diff_perframe, repdim)

estim_x_diff_mean = np.nanmean( diff(x_mean_array_cm, window_dim), repdim)
estim_y_diff_mean = np.nanmean( diff(y_mean_array_cm, window_dim), repdim)
if np.any(n_valid_reps_per_window == 0):
    n_empty_windows = int(np.sum(n_valid_reps_per_window == 0))
    print(f"Warning: {n_empty_windows} window(s) have zero reps with spikes; those cross-rep means are NaN.")

#plot(x_true_array_cm[0, 1:], sqrt( true_xdiff_mean**2 + true_ydiff_mean**2) )
#L2_of_meandiff = sqrt( estim_x_diff_mean**2 + estim_y_diff_mean**2)
#plot(x_true_array_cm[0, 1:], mean_L2diff_acrossreps)
#plot(x_true_array_cm[0, 1:], L2_of_meandiff)

plot(x_start_cm + x_step_cm * arange(1, n_windows), sqrt( true_xdiff_mean**2 + true_ydiff_mean**2) )
L2_of_meandiff = sqrt( estim_x_diff_mean**2 + estim_y_diff_mean**2)
plot(x_start_cm + x_step_cm * arange(1, n_windows), mean_L2diff_acrossreps)
plot(x_start_cm + x_step_cm * arange(1, n_windows), L2_of_meandiff)

plot(x_start_cm + x_step_cm * arange(n_windows), 10 * np.nanmean(tot_spikes, repdim) / tot_spikes.max())

#axis([0, n_grid_cm, 0.9*min( mean_L2diff_acrossreps ), 1.2*max( mean_L2diff_acrossreps )])

ylabel('Step size [cm]', fontsize=fsize)
ax=gca()
#ax.set_xticks([0, n_grid_cm])
#ax.set_xticklabels([0, n_grid_cm])
xtick_locs = x_start_cm + linspace(0, n_windows * x_step_cm, int(n_osc_cycles) + 1)
ax.set_xticks(xtick_locs)
ax.set_xticklabels(['0'] + ['{}$\pi$'.format(2*i) for i in range(1, len(xtick_locs))])
#ax.set_yticks([0, max(100, ylim()[1])])
#ax.set_yticklabels([0, max(100, ylim()[1])])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 
xlabel('Oscillation phase', fontsize=fsize)
#legend(('True step size', 'Mean L2 of estim. steps','L2 of mean estim. steps'), fontsize=fsize_small)
legend(('True step size', 'Mean L2 of estim. steps','L2 of mean estim. steps','spike count'), fontsize=fsize_small)

#'''

'''#
subplot(n_plots_x, n_plots_y, 2)
plot(tot_spikes[:, 1:], L2_diff_perframe, 'k.', markersize=msize) 
xlabel('Spike count per frame', fontsize=fsize)
ylabel('Estim. step size', fontsize=fsize)
ax=gca()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 
'''

subplot(n_plots_x, n_plots_y, 2)
# Caution: Spike count of which decoding window?! Pfeiffer & Foster (2015) use the sum of spikes of both consecutive (overlapping) windows!
# non-circular convolution for local array smoothing or summation:
# Can be implemented based on scipy.signal's gaussian window function, or a custom boxcar window (8 bins of 10 deg. each, as in Pfeiffer & Foster, 2015).
# Example:
datalen = n_windows
#box2 = append(ones(2), zeros(datalen-2)) # summation across windows
box2 = 0.5 * append(ones(2), zeros(datalen-2)) # averaging across windows
from scipy.signal import convolve
tot_spikes_summed = nan * ones([n_reps, n_windows - 1])
for i_reps in range(n_reps):
    tot_spikes_summed[i_reps, :] = convolve(tot_spikes[i_reps, :], box2)[1:datalen]
plot(tot_spikes_summed, L2_diff_perframe, 'r.', markersize=msize)
xlabel('Spike count per frame', fontsize=fsize)
ylabel('Estim. step size', fontsize=fsize)
ax=gca()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 

subplot(n_plots_x, n_plots_y, 4)
plot(tot_spikes_summed, diff(x_mean_array_cm, window_dim), 'k.', markersize=msize)
#plot([0, 120], mean(diff(x_mean_array_cm, window_dim)) * ones(2),'k--')

'''#
slope, intercept, r_value, p_value, std_err = st.linregress(reshape(tot_spikes_summed, np.prod(tot_spikes_summed.shape)), reshape(diff(x_mean_array_cm, window_dim), np.prod(diff(x_mean_array_cm, window_dim).shape)) )
print("Linear fit of x_step: R^2, p_value =", r_value**2, p_value)
print("slope, intercept =", slope, intercept)
xvals = arange(0, 150)
plot(xvals, intercept + xvals * slope, 'r--', lw=1)
'''
#'''#
spcount_vals = np.arange(0, tot_spikes.max()+10, 10)
for i_bin in range(len(spcount_vals)):
    nz_ireps, nz_jframe = nonzero( (tot_spikes[:, :-1] <= i_bin + 10) * (tot_spikes[:, :-1] > i_bin ) )
    if nz_ireps.size == 0:
        print(f"spike-count bin [{i_bin}, {i_bin+10}): empty, skipping")
        continue
    binmean_xdiff = np.nanmean( diff(x_mean_array_cm, window_dim)[nz_ireps, nz_jframe] )
    if not np.isfinite(binmean_xdiff):
        print(f"spike-count bin [{i_bin}, {i_bin+10}): all matched diffs are NaN, skipping")
        continue
    print("binmean of x-diff =", binmean_xdiff)
    plot([spcount_vals[i_bin], spcount_vals[i_bin]+10], binmean_xdiff * ones(2), 'r-')
#'''

xlabel('Spike count per frame', fontsize=fsize)
ylabel('x step size', fontsize=fsize)
ax=gca()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 


subplot(n_plots_x, n_plots_y, 6)
rsp_totspikes = reshape(tot_spikes[:, 1:], prod(tot_spikes[:, 1:].shape))
hist(rsp_totspikes, 20)
xlabel('Spike count per frame (2nd)', fontsize=fsize)
ylabel('Count', fontsize=fsize)
ax=gca()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 

# subplots_adjust(bottom=0.1, right=0.8, top=0.9)
import os 

file_name = f'speedmod{speed_mod_depth}_sim_seq_paramsid_{params_id}_{n_reps}reps_{int(n_mincells)}mincells_{n_cells}cells_{n_windows}steps_{gamma_mod_depth}moddepth'
os.makedirs('figs', exist_ok=True)

savefig('figs/plot_'+file_name + '.png', dpi=300)

# Phase-binned spike counts across all reps (à la P&F)
n_phase_bins   = 36
bin_width_deg  = 10

# oscillation phase (in degrees) for every window
osc_phases_deg = np.mod(
    np.arange(n_windows) / float(n_windows) * n_osc_cycles * 360.0,
    360.0,
)

# bin edges and centres
bin_edges   = np.arange(0, n_phase_bins * bin_width_deg + 1, bin_width_deg)
bin_centres = bin_edges[:-1] + bin_width_deg / 2.0

# assign each window to a phase bin
bin_idx = np.digitize(osc_phases_deg, bin_edges) - 1  # 0-based
bin_idx = np.clip(bin_idx, 0, n_phase_bins - 1)

# accumulate spike counts across all reps and windows into phase bins
bin_sum   = np.zeros(n_phase_bins)
bin_count = np.zeros(n_phase_bins)
for i_wind in range(n_windows):
    b = bin_idx[i_wind]
    bin_sum[b]   += np.nansum(tot_spikes[:, i_wind])
    bin_count[b] += np.sum(np.isfinite(tot_spikes[:, i_wind]))

bin_mean = np.where(bin_count > 0, bin_sum / bin_count, np.nan)

# modulation depth (Michelson contrast, same as modulation_depth_alt)
finite      = bin_mean[np.isfinite(bin_mean)]
mod_depth   = (finite.max() - finite.min()) / (finite.max() + finite.min()) if finite.size else np.nan

figure(figsize=(6, 3.5), dpi=300)
bar(bin_centres, bin_mean, width=bin_width_deg, color="0.7", edgecolor="white")
axhline(np.nanmean(bin_mean), color="k", lw=1, ls="--")
xlabel("Gamma phase (deg)", fontsize=fsize)
ylabel("Mean spikes per step", fontsize=fsize)
title(f"Spikes vs $\\gamma$ phase  (n={n_cells} cells, {n_reps} reps, speedmod)\n"
      f"modulation depth = {mod_depth*100:.2f}%   |   "
      f"$\\gamma_{{mod}}$ = {gamma_mod_depth}",
      fontsize=fsize)
xlim(0, 360)
ax = gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(fsize)
tight_layout()
savefig(f"figs/phase_binned_spikes_{file_name}.png", dpi=300)

print(np.nanmean(tot_spikes), np.nanstd(tot_spikes))

os.makedirs('data', exist_ok=True)
print("saving data...")
with open('data/'+file_name + '.txt','wb') as file_save:
    pickle.dump(x_true_array_cm, file_save, 0)
    pickle.dump(y_true_array_cm, file_save, 0)
    pickle.dump(x_mean_array_cm, file_save, 0)
    pickle.dump(y_mean_array_cm, file_save, 0)
    pickle.dump(tot_spikes, file_save, 0)
    pickle.dump(mu_x, file_save, 0)
    pickle.dump(mu_y, file_save, 0)
    pickle.dump("1. x_true_array_cm, 2. y_true_array_cm, 3. x_mean_array_cm, 4. y_mean_array_cm, 5. tot_spikes, 6. mu_x, 7. mu_y", file_save, 0)

# Also save in numpy format for easier loading
np.save('data/'+file_name + '_npformat.npy', 
        np.array((x_true_array_cm, y_true_array_cm, x_mean_array_cm, y_mean_array_cm, tot_spikes, "1. x_true_array_cm, 2. y_true_array_cm, 3. x_mean_array_cm, 4. y_mean_array_cm, 5. tot_spikes"), dtype=object), 
        allow_pickle=True)

print("Done")

ioff()
#show()
