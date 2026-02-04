from pylab import *
import numpy as np
import scipy.stats as st
import new_colormaps as nc
#from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle

from sim_params_seq import params_id, n_grid_cm, DeltaT_window_sec, overlap, n_windows, n_reps, n_osc_cycles, sigma_code_cm, x_step_cm, y_step_cm, max_rate_Hz, n_mincells

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

x_grid, y_grid = meshgrid(range(n_grid_cm), range(n_grid_cm)) # Bin size: 1cm

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

n_cells = 200
mu_x = n_grid_cm * rand(n_cells) # Fixed PF center locations
mu_y = n_grid_cm * rand(n_cells)

sigma_code_cm = 15.0 

DeltaT_window_sec = 0.02


increment_fraction = 1 - overlap
increment_sec = increment_fraction * DeltaT_window_sec
n_increments = int(1.0 / increment_fraction)       



subplot(n_plots_x, n_plots_y, 1)

for i_reps in xrange(n_reps):
    print "i_reps = ", i_reps

    call_counter = 0

    n_spikes_ringbuffer = nan * ones([n_cells, n_increments])

    rand_offset_x_cm = 10*(rand()-0.5) # [-5cm, 5cm]
    rand_offset_y_cm = 10*(rand()-0.5)


    for i_wind in xrange(n_windows):

        #n_mincells = 10.0 # 20.0
        frac_mincells = n_mincells / 200.0
        osc_phase = i_wind / float(n_windows) * n_osc_cycles * 2*pi

        osc_modulation = 0.5*(1 + frac_mincells + (1 - frac_mincells) * cos(osc_phase))
        #print "cos(osc_phase), osc_modulation = ", cos(osc_phase), osc_modulation
        n_cells = 200 # int(200 * osc_modulation )

        peak_rate_Hz = max_rate_Hz * osc_modulation

        x_true_array_cm[i_reps, i_wind] = x_start_cm + i_wind * x_step_cm + rand_offset_x_cm
        y_true_array_cm[i_reps, i_wind] = y_start_cm + i_wind * y_step_cm + rand_offset_y_cm

        PF_prod = ones([n_grid_cm, n_grid_cm])
        poi_rate = zeros(n_cells)
        n_spikes = zeros(n_cells)



        # Alternativen:
        # 1) Place fields fuer alle Zellen einmalig berechnen + speichern (1 Array der Groesse xbins x ybins); -> weniger exp()-Funktionsaufrufe;
        # 1a) Bin size einstellen: x_grid = range(0, n_grid_cm, binsize); x_bin = int(digitize(x, x_grid))
        # 1b) PF_i = exp(-((x_grid - mu_x[i_cell])**2 + (y_grid - mu_y[i_cell])**2)/sigma_code_cm**2)
        # 1c) x_bin = int(digitize(x, x_grid)); Feuerrate fuer Zelle i ergibt sich aus PF_i[x_bin, y_bin]


        # Zusaetzlich: Beruecksichtigung von Overlap, evtl. mittels Ringpuffer:
        # call_counter += 1
        # n_spikes_ringbuffer[call_counter % 20] = len(spikelist)


        while sum(n_spikes) == 0: 
        #while nansum(n_spikes_ringbuffer) == 0: # True only for i_wind==0
            #print "sum(n_spikes_ringbuffer) = ", sum(n_spikes_ringbuffer)


            for i_cell in xrange(n_cells):
                # Now, draw spikes from a Poisson distribution, with a rate determined by the "true" represented location relative to the place field center, scaled by sigma_code_cm:

                # PF_cell_i = exp(-((x_grid - mu_x[i_cell])**2 + (y_grid - mu_y[i_cell])**2)/sigma_code_cm**2)

                virtualPF_xytrue_cell_i = exp(-((x_true_array_cm[i_reps, i_wind] - mu_x[i_cell])**2 + (y_true_array_cm[i_reps, i_wind] - mu_y[i_cell])**2)/sigma_code_cm**2)
                #poi_rate_cell_i = peak_rate_Hz * DeltaT_window_sec * virtualPF_xytrue_cell_i * exp(- DeltaT_window_sec * virtualPF_xytrue_cell_i) # without overlap
                poi_rate_cell_i = peak_rate_Hz * increment_sec * virtualPF_xytrue_cell_i * exp(- increment_sec * virtualPF_xytrue_cell_i) # generalizing to windows with overlap

                poi_rate[i_cell] = poi_rate_cell_i

            if i_wind == 0:
                # Ensure that the ring buffer is filled
                for i_increments in xrange(n_increments):
                    n_spikes = poisson(poi_rate) # Is this correct??? We're using a constant phase value for n_increments "sub-windows" ?!?!
                    n_spikes_ringbuffer[:, i_increments % n_increments] = n_spikes

            else:
                n_spikes = poisson(poi_rate)
                #print "(i_wind + n_increments) % n_increments = ", (i_wind + n_increments) % n_increments
                n_spikes_ringbuffer[:, (i_wind + n_increments) % n_increments] = n_spikes


    
            #tot_spikes[i_reps, i_wind] = sum(n_spikes)
            tot_spikes[i_reps, i_wind] = sum(sum(n_spikes_ringbuffer))

            #print "tot_spikes = ", tot_spikes[i_reps, i_wind]

        #print "i_reps, i_wind, max(poi_rate), tot_spikes[i_reps, i_wind] = ", i_reps, i_wind, poi_rate.max(), tot_spikes[i_reps, i_wind]

        #mu_x_mean = np.average(mu_x, weights=n_spikes)
        #mu_y_mean = np.average(mu_y, weights=n_spikes)
        mu_x_mean = np.average(mu_x, weights=sum(n_spikes_ringbuffer, 1))
        mu_y_mean = np.average(mu_y, weights=sum(n_spikes_ringbuffer, 1))

        x_mean_array_cm[i_reps, i_wind], y_mean_array_cm[i_reps, i_wind] = mu_x_mean, mu_y_mean

        std_x = sqrt(np.average((mu_x - mu_x_mean)**2, weights=sum(n_spikes_ringbuffer, 1)))
        std_y = sqrt(np.average((mu_y - mu_y_mean)**2, weights=sum(n_spikes_ringbuffer, 1)))
        std_xy = sqrt(std_x * std_y)
        std_array_cm[i_wind] = std_xy

        if i_reps==0:

            # Konfidenzintervall zum Niveau alpha der t-Verteilung mit n-1 Freiheitsgraden:
            alpha=0.95

            t_conf_lo, t_conf_up = st.t.interval(alpha, tot_spikes[i_reps, i_wind]-1, 0, 1) # For Poisson activity

            # Konfidenzintervall eines normalverteilten Merkmals mit Stichprobenvarianz s^2:
            # [x_mean - t_conf * s / sqrt(n); x_mean + t_conf * s / sqrt(n) ] # Achtung, korrigierte Stichprobenvarianz benutzen!
            # CI_x_lower = x_mean + t_conf_lo * std_xy / sqrt(n_cells)

            t=arange(0, 2*pi, 0.01*pi)



            R[i_wind] = t_conf_lo * std_array_cm[i_wind] / sqrt(tot_spikes[i_reps, i_wind])
            R[i_wind] = abs(t_conf_lo * std_array_cm[i_wind] / sqrt(tot_spikes[i_reps, i_wind]))

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

            nz_spikes = nonzero(n_spikes > 0)[0]
            #nz_spikes = nonzero(sum(n_spikes_ringbuffer, 1) > 0)[0]

            for i_cell in nz_spikes:
                plot(mu_x[i_cell], mu_y[i_cell],  '.', color=str(osc_modulation), markersize=1.5)
                #sc = scatter(mu_x[i_cell], mu_y[i_cell], c=1 - n_spikes[i_cell]/max(n_spikes), cmap=cm.Greys, marker='.', s=1)


    if i_reps==0:


        for i_wind in xrange(n_windows):
            plot(x_mean_array_cm[i_reps, i_wind] + R[i_wind] * cos(t), y_mean_array_cm[i_reps, i_wind] + R[i_wind]*sin(t), color=str(tot_spikes[i_reps, i_wind] / np.nanmax(tot_spikes) ), lw=linew) # CI around the sample mean




    


            if i_wind == n_windows - 1:
                plot(x_true_array_cm[i_reps, :], y_true_array_cm[i_reps, :], 'k|-', markersize=5, lw=1)
                plot(x_mean_array_cm[i_reps, :], y_mean_array_cm[i_reps, :], '.-', color=tab_red, markersize=2.5) # 1.5


subplot(n_plots_x, n_plots_y, 3)
for i_wind in xrange(n_windows):
    if i_reps == 1:
        bar(x_true_array_cm[i_reps, i_wind] - 0.5*x_step_cm, tot_spikes[i_reps, i_wind], width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * n_osc_cycles * 2*pi)) )) # osc_modulation
    else:
        bar(x_true_array_cm[i_reps, i_wind] - 0.5*x_step_cm, nanmean(tot_spikes[:, i_wind]), width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * n_osc_cycles * 2*pi)) ))

ylabel('Spike count', fontsize=fsize)
axis([0, n_grid_cm, 0, np.nanmax(tot_spikes)])
ax=gca()
#ax.set_aspect(0.2)
#ax.set_xticks([x_start_cm, x_start_cm + 0.5*n_windows * x_step_cm, x_start_cm + n_windows * x_step_cm])
#ax.set_xticklabels([0, '2$\pi$', '4$\pi$']) 
ax.set_xticks(   arange(  x_start_cm, int(x_start_cm + (n_windows+1) * x_step_cm),  (n_windows+1) * x_step_cm / float(n_osc_cycles) )  )   
#ax.set_xticklabels( arange(0, (n_osc_cycles+1) * 2*pi, 2*pi) )
ax.set_xticklabels([0, '2$\pi$', '4$\pi$']) 
ax.set_yticks([0, max(100, ylim()[1])])
ax.set_yticklabels([0, int(max(100, ylim()[1])) ])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) # 5
xlabel('Oscillation phase', fontsize=fsize)

#'''#

subplot(n_plots_x, n_plots_y, 5)
true_xdiff_mean = mean( diff(x_true_array_cm, window_dim), repdim)
true_ydiff_mean = mean( diff(y_true_array_cm, window_dim), repdim)

L2_diff_perframe = sqrt( diff(x_mean_array_cm, window_dim)**2 + diff(y_mean_array_cm, window_dim)**2)
mean_L2diff_acrossreps = mean(L2_diff_perframe, repdim)

estim_x_diff_mean = mean( diff(x_mean_array_cm, window_dim), repdim) # This is the way it SHOULD be done, not the way Pfeiffer & Foster did it
estim_y_diff_mean = mean( diff(y_mean_array_cm, window_dim), repdim)

L2_of_meandiff = sqrt( estim_x_diff_mean**2 + estim_y_diff_mean**2)

plot(x_true_array_cm[0, 1:], sqrt( true_xdiff_mean**2 + true_ydiff_mean**2) )
plot(x_true_array_cm[0, 1:], mean_L2diff_acrossreps)
plot(x_true_array_cm[0, 1:], L2_of_meandiff)
#plot(x_true_array_cm[0, 1:], estim_x_diff_mean)
plot(x_true_array_cm[0, :], 0.1*mean(tot_spikes, repdim))


ylabel('Step size [cm]', fontsize=fsize)
ax=gca()
ax.set_xticks(   arange(  x_start_cm, int(x_start_cm + (n_windows+1) * x_step_cm),  (n_windows+1) * x_step_cm / float(n_osc_cycles) )  )   
ax.set_xticklabels([0, '2$\pi$', '4$\pi$'])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 
xlabel('Oscillation phase', fontsize=fsize)
legend(('True step size', 'Mean L2 of estim. steps','L2 of mean estim. steps'), fontsize=fsize_small)
#'''


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
for i_reps in xrange(n_reps):
    tot_spikes_summed[i_reps, :] = convolve(tot_spikes[i_reps, :], box2)[1:datalen]
plot(tot_spikes_summed, L2_diff_perframe, 'r.', markersize=msize)
xlabel('Spike count per frame', fontsize=fsize)
ylabel('Estim. step size', fontsize=fsize)
ax=gca()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 

subplot(n_plots_x, n_plots_y, 4)
plot(tot_spikes_summed, diff(x_mean_array_cm, window_dim), 'k.', markersize=msize)

print "tot_spikes.max() = ", tot_spikes.max()
print "tot_spikes.shape = ", tot_spikes.shape
print "x_mean_array_cm.shape = ", x_mean_array_cm.shape
spcount_vals = np.arange(0, tot_spikes.max(), 10)
#'''#
for i_bin in xrange(len(spcount_vals)):
    #nz_ireps, nz_jframe = nonzero( (tot_spikes <= i_bin + 10) * (tot_spikes > i_bin ) )
    nz_ireps, nz_jframe = nonzero( (tot_spikes[:, :-1] <= i_bin + 10) * (tot_spikes[:, :-1] > i_bin ) )
    #print "nz_ireps.shape = ", nz_ireps
    #print "max(nz_ireps), max(nz_jframe) = ", max(nz_ireps), max(nz_jframe)
    binmean_xdiff = mean( diff(x_mean_array_cm, window_dim)[nz_ireps, nz_jframe] ) # Problems in this line of code ?!
    print "binmean of x-diff = ", binmean_xdiff
    plot([spcount_vals[i_bin], spcount_vals[i_bin]+10], binmean_xdiff * ones(2), 'r-')

xlabel('Spike count per frame', fontsize=fsize)
ylabel('x step size', fontsize=fsize)
ax=gca()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 
#'''

subplot(n_plots_x, n_plots_y, 6)
print "sum(isnan(tot_spikes)) = ", sum(sum(isnan(tot_spikes)))
rsp_totspikes = reshape(tot_spikes[:, 1:], prod(tot_spikes[:, 1:].shape))
hist(rsp_totspikes, 20)
xlabel('Spike count per frame (2nd)', fontsize=fsize)
ylabel('Count', fontsize=fsize)
ax=gca()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) 



# subplots_adjust(bottom=0.1, right=0.8, top=0.9)



savefig('figs/plot_sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.png', dpi=300)

print "saving plot data..."
file_save = open('data/sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','w')
pickle.dump(x_true_array_cm, file_save, 0)
pickle.dump(y_true_array_cm, file_save, 0)
pickle.dump(x_mean_array_cm, file_save, 0)
pickle.dump(y_mean_array_cm, file_save, 0)
pickle.dump(tot_spikes, file_save, 0)
pickle.dump(mu_x, file_save, 0)
pickle.dump(mu_y, file_save, 0)
pickle.dump("1. x_true_array_cm, 2. y_true_array_cm, 3. x_mean_array_cm, 4. y_mean_array_cm, 5. tot_spikes, 6. mu_x, 7. mu_y", file_save, 0)
#pickle.dump(, file_save, 0)
file_save.close()
print "Done"

ioff()
show()









