from pylab import *
import numpy as np
import scipy.stats as st
import new_colormaps as nc
#from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle

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

n_grid_cm = 100

sigma_scale_factor = 1.0 


#figure(figsize=(3.8, 3.55), dpi=300)
#figure(figsize=(2.5, 4), dpi=300) # 4x1 plot layout
figure(figsize=(5, 4), dpi=300) # 2x2 plot layout

fsize = 6
fsize_small = 4#.5
linew = 1.5
msize = 3 # 30 # 8

x_start_cm = 10 # 25
y_start_cm = 10 # 25
x_step_cm = 2.5 # 5 # 10
y_step_cm = 2.5 # 5 # 10

x_grid, y_grid = meshgrid(range(n_grid_cm), range(n_grid_cm))

n_windows = 32 # 16 # decoding windows for a single simulated sequence
n_reps = 1000 # 101 # 1      # no. of sequence repetitions

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
mu_x = 100 * rand(n_cells) # Fixed PF center locations
mu_y = 100 * rand(n_cells)

sigma_code_cm = 15.0 * sigma_scale_factor
sigma_PF = 15.0
DeltaT_window_sec = 0.02


subplot(n_plots_x, n_plots_y, 1)


for i_reps in xrange(n_reps):
    print "i_reps = ", i_reps
    x_old_cm = x_start_cm
    y_old_cm = y_start_cm

    for i_wind in xrange(n_windows):

        n_mincells = 10.0 # 20.0
        frac_mincells = n_mincells / 200.0
        osc_phase = i_wind / float(n_windows) * 4*pi

        osc_modulation = 0.5*(1 + frac_mincells + (1 - frac_mincells) * cos(osc_phase))
        n_cells = 200 # int(200 * osc_modulation )

        peak_rate_Hz = (1/sigma_scale_factor)**2 * 300.0 * osc_modulation

        #speed_modulation = (1 - 0.5*cos(osc_phase)) # in range [0.5, 1.5]; phase-inverted relative to the pop. oscillation (0.5x speed at peak activity, 1.5x speed at min. activity)
        # More general:
        mod_depth = 1.0 # 0.5 # 0.2 
        speed_modulation = (1.0 - mod_depth * cos(osc_phase)) # in range [1 - mod_depth, 1 + mod_depth]
        # Make sure that the modulation across windows sums to a value of 1, or the sequence will overshoot!
        x_true_array_cm[i_reps, i_wind] = x_old_cm + x_step_cm * speed_modulation
        y_true_array_cm[i_reps, i_wind] = y_old_cm + y_step_cm * speed_modulation
        x_old_cm = x_true_array_cm[i_reps, i_wind]
        y_old_cm = y_true_array_cm[i_reps, i_wind]

        #print "x_true_array_cm[i_reps, i_wind] = ", x_true_array_cm[i_reps, i_wind]

        PF_sum = zeros([n_grid_cm, n_grid_cm])
        PF_prod = ones([n_grid_cm, n_grid_cm])
        poi_rate = zeros(n_cells)
        n_spikes = zeros(n_cells)

        while sum(n_spikes) == 0:
            for i_cell in xrange(n_cells):
                '''#
                # Draw x and y place field centers from a Gaussian distribution around the "true" mean:
                mu_x[i_cell] = sigma_code_cm * randn() + x_true_array_cm[i_reps, i_wind]
                mu_y[i_cell] = sigma_code_cm * randn() + y_true_array_cm[i_reps, i_wind]
                sigma_rand = 0.0 # 10*rand()
                PF_sum += exp(-((x_grid - mu_x[i_cell])**2 + (y_grid - mu_y[i_cell])**2)/(sigma_PF + sigma_rand)**2)
                PF_prod *= exp(-((x_grid - mu_x[i_cell])**2 + (y_grid - mu_y[i_cell])**2)/(sigma_PF + sigma_rand)**2)
                '''

                # Now, draw spikes from a Poisson distribution, with a rate determined by the "true" represented location relative to the place field center, scaled by sigma_code_cm:

                PF_cell_i = exp(-((x_grid - mu_x[i_cell])**2 + (y_grid - mu_y[i_cell])**2)/sigma_PF**2)

                virtualPF_xytrue_cell_i = exp(-((x_true_array_cm[i_reps, i_wind] - mu_x[i_cell])**2 + (y_true_array_cm[i_reps, i_wind] - mu_y[i_cell])**2)/sigma_code_cm**2)
                poi_rate_cell_i = peak_rate_Hz * DeltaT_window_sec * virtualPF_xytrue_cell_i * exp(- DeltaT_window_sec * virtualPF_xytrue_cell_i)
                poi_rate[i_cell] = poi_rate_cell_i

                n_spikes[i_cell] = poisson(poi_rate_cell_i)

                PF_sum += PF_cell_i

                #Bayes_denom *= (PF_cell_i)**(n_spikes[i_cell]) * poi_rate_cell_i

            tot_spikes[i_reps, i_wind] = sum(n_spikes)
            #print "tot_spikes = ", tot_spikes[i_reps, i_wind]
            

        x_mean, y_mean = mean(mu_x), mean(mu_y)
        #x_mean_array_cm[i_reps, i_wind], y_mean_array_cm[i_reps, i_wind] = x_mean, y_mean


        mu_x_mean = np.average(mu_x, weights=n_spikes)
        mu_y_mean = np.average(mu_y, weights=n_spikes)
        x_mean_array_cm[i_reps, i_wind], y_mean_array_cm[i_reps, i_wind] = mu_x_mean, mu_y_mean

        std_x = sqrt(np.average((mu_x - mu_x_mean)**2, weights=n_spikes))
        std_y = sqrt(np.average((mu_y - mu_y_mean)**2, weights=n_spikes))
        std_xy = sqrt(std_x * std_y)
        std_array_cm[i_wind] = std_xy
        #print "std. [PF centers] (x,y)= ", std_xy


        #print "x_mean, y_mean [PF centers] = ", round(x_mean), round(y_mean)

        if i_reps==0:

            # Konfidenzintervall zum Niveau alpha der t-Verteilung mit n-1 Freiheitsgraden:
            alpha=0.95
            #t_conf_lo, t_conf_up = st.t.interval(alpha, n_cells-1, 0, 1)
            t_conf_lo, t_conf_up = st.t.interval(alpha, tot_spikes[i_reps, i_wind]-1, 0, 1) # For Poisson activity
            # Konfidenzintervall eines normalverteilten Merkmals mit Stichprobenvarianz s^2:
            # [x_mean - t_conf * s / sqrt(n); x_mean + t_conf * s / sqrt(n) ] # Achtung, korrigierte Stichprobenvarianz benutzen!
            # CI_x_lower = x_mean + t_conf_lo * std_xy / sqrt(n_cells)

            t=arange(0, 2*pi, 0.01*pi)


            #plot(mu_x, mu_y, '.', color='k', markersize=msize) # tab_blue
            #plot(mu_x, mu_y, '.', color=str(1 - n_cells / 200.0), markersize=msize) # tab_blue


            #plot(x_mean, y_mean, '+', markersize=msize, color='k')
            #R = t_conf_lo * std_xy / sqrt(n_cells)
            R[i_wind] = t_conf_lo * std_array_cm[i_wind] / sqrt(tot_spikes[i_reps, i_wind])
            R[i_wind] = abs(t_conf_lo * std_array_cm[i_wind] / sqrt(tot_spikes[i_reps, i_wind]))
            #print "R = ", R[i_wind]

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


        for i_wind in xrange(n_windows):
            plot(x_mean_array_cm[i_reps, i_wind] + R[i_wind] * cos(t), y_mean_array_cm[i_reps, i_wind] + R[i_wind]*sin(t), color=str(tot_spikes[i_reps, i_wind] / np.nanmax(tot_spikes) ), lw=linew) # CI around the sample mean




    
#tight_layout() # Creates sufficient spacing, but makes subplots smaller

            if i_wind == n_windows - 1:
                plot(x_true_array_cm[i_reps, :], y_true_array_cm[i_reps, :], 'k.-')
                plot(x_mean_array_cm[i_reps, :], y_mean_array_cm[i_reps, :], '.-', color=tab_red)


subplot(n_plots_x, n_plots_y, 3)
for i_wind in xrange(n_windows):
    if i_reps == 1:
        #bar(x_true_array_cm[i_reps, i_wind] - 0.5*x_step_cm, tot_spikes[i_reps, i_wind], width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * 4*pi)) )) 
        bar(i_wind * x_step_cm + x_start_cm, tot_spikes[i_reps, i_wind], width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * 4*pi)) )) 
    else:
        #bar(x_true_array_cm[i_reps, i_wind] - 0.5*x_step_cm, nanmean(tot_spikes[:, i_wind]), width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * 4*pi)) ))
        bar(i_wind * x_step_cm + x_start_cm, nanmean(tot_spikes[:, i_wind]), width=x_step_cm, color=str( 0.5*(1 + frac_mincells + (1 - frac_mincells)*cos(i_wind / float(n_windows) * 4*pi)) ))

ylabel('Spike count', fontsize=fsize)
axis([0, n_grid_cm, 0, np.nanmax(tot_spikes)])
ax=gca()
ax.set_aspect(0.2)
#ax.set_xticks([0, n_grid_cm])
#ax.set_xticklabels([0, n_grid_cm])
ax.set_xticks([x_start_cm, x_start_cm + 0.5*n_windows * x_step_cm, x_start_cm + n_windows * x_step_cm])
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
ax.set_xticks([x_start_cm, x_start_cm + 0.5*n_windows * x_step_cm, x_start_cm + n_windows * x_step_cm])
ax.set_xticklabels([0, '2$\pi$', '4$\pi$'])
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
#plot([0, 120], mean(diff(x_mean_array_cm, window_dim)) * ones(2),'k--')

'''#
slope, intercept, r_value, p_value, std_err = st.linregress(reshape(tot_spikes_summed, np.prod(tot_spikes_summed.shape)), reshape(diff(x_mean_array_cm, window_dim), np.prod(diff(x_mean_array_cm, window_dim).shape)) )
print "Linear fit of x_step: R^2, p_value = ", r_value**2, p_value
print "slope, intercept = ", slope, intercept
xvals = arange(0, 150)
plot(xvals, intercept + xvals * slope, 'r--', lw=1)
'''
'''#
spcount_vals = np.arange(0, tot_spikes.max()+10, 10)
for i_bin in xrange(len(spcount_vals)):
    nz_ireps, nz_jframe = nonzero( (tot_spikes <= i_bin + 10) * (tot_spikes > i_bin ) )
    binmean_xdiff = mean( diff(x_mean_array_cm, window_dim)[nz_ireps, nz_jframe] )
    print "binmean of x-diff = ", binmean_xdiff
    plot([spcount_vals[i_bin], spcount_vals[i_bin]+10], binmean_xdiff * ones(2), 'r-')
'''

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

#tight_layout()

savefig('figs/plot_speedmod'+str(mod_depth)+'_sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.png', dpi=300)

print "saving plot data..."
file_save = open('data/speedmod_'+str(mod_depth)+'_sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','w')
pickle.dump(x_true_array_cm, file_save, 0)
pickle.dump(y_true_array_cm, file_save, 0)
pickle.dump(x_mean_array_cm, file_save, 0)
pickle.dump(y_mean_array_cm, file_save, 0)
pickle.dump(tot_spikes, file_save, 0)
pickle.dump("1. x_true_array_cm, 2. y_true_array_cm, 3. x_mean_array_cm, 4. y_mean_array_cm, 5. tot_spikes", file_save, 0)
#pickle.dump(, file_save, 0)
file_save.close()
print "Done"

ioff()
show()









