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

col_blue = '#56B4E9' # 'DodgerBlue' # 'b'
col_green = '#2B9F78' # 'SeaGreen' # 'g'
col_vermillion = '#D55E00' # 'Tomato' # 'PaleVioletRed' # 'r'
col_blue2 = '#0072B2' # 'SteelBlue' # 'c'
col_orange = '#E69F00' # 'Orange' #'m'
col_purple = '#CC79A7' 

col_biased = col_blue2 # col_purple
col_unbiased = col_vermillion # col_purple # col_blue # col_orange



ion()

n_grid_cm = 100
n_cells = 200
sigma_scale_factor = 1.0 
DeltaT_window_sec = 0.02

figure(figsize=(4, 3), dpi=300)

fsize = 6 # 3 # 6
fsize_small = 5 # 4 # 2 # 4
linew = 1 # 1.5 # 
msize_big = 5 # 3 # 30 # 8
msize = 2

x_start_cm = 10 # 25
y_start_cm = 10 # 25
x_step_cm = 2.5 # 5 # 10
y_step_cm = 2.5 # 5 # 10
x_grid, y_grid = meshgrid(range(n_grid_cm), range(n_grid_cm))
sigma_code_cm = 15.0
n_mincells = 10.0
frac_mincells = n_mincells / 200.0

n_reps = 100 # 1000 # 100
n_windows = 32
mod_depth = 1.0

osc_phase = arange(0, n_windows) / float(n_windows) * 4*pi
osc_modulation = 0.5*(1 + frac_mincells + (1 - frac_mincells) * cos(osc_phase))




file_co = open('data/sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','rb')
co_x_true_array_cm = pickle.load(file_co, encoding='latin1')
co_y_true_array_cm = pickle.load(file_co, encoding='latin1')
co_x_mean_array_cm = pickle.load(file_co, encoding='latin1')
co_y_mean_array_cm = pickle.load(file_co, encoding='latin1')
co_tot_spikes = pickle.load(file_co, encoding='latin1')
co_var_names = pickle.load(file_co, encoding='latin1')
file_co.close()

#'''#
file_spmod = open('data/speedmod_'+str(mod_depth)+'_sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','rb')
sm_x_true_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_y_true_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_x_mean_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_y_mean_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_tot_spikes = pickle.load(file_spmod, encoding='latin1')
sm_var_names = pickle.load(file_spmod, encoding='latin1')
file_spmod.close()
#'''

repdim, window_dim = (0, 1)
n_plots_x = 2# 4
n_plots_y = 2


sm_true_xdiff_mean = mean( diff(sm_x_true_array_cm, window_dim), repdim)
sm_true_ydiff_mean = mean( diff(sm_y_true_array_cm, window_dim), repdim)
sm_L2_diff_perframe = sqrt( diff(sm_x_mean_array_cm, window_dim)**2 + diff(sm_y_mean_array_cm, window_dim)**2)
sm_mean_L2diff_acrossreps = mean(sm_L2_diff_perframe, repdim)
sm_min_L2diff_acrossreps = sm_L2_diff_perframe.min(repdim)
sm_std_L2diff_acrossreps = std(sm_L2_diff_perframe, repdim)
#sm_sem_L2diff_acrossreps = std(sm_L2_diff_perframe, repdim) / sqrt(n_reps) # almost invisible
sm_estim_x_diff_mean = mean( diff(sm_x_mean_array_cm, window_dim), repdim) # This is the way it SHOULD be done, not the way Pfeiffer & Foster did it
sm_estim_y_diff_mean = mean( diff(sm_y_mean_array_cm, window_dim), repdim)
sm_estim_x_diff_std = std( diff(sm_x_mean_array_cm, window_dim), repdim)
sm_estim_y_diff_std = std( diff(sm_y_mean_array_cm, window_dim), repdim)
sm_L2_of_meandiff = sqrt( sm_estim_x_diff_mean**2 + sm_estim_y_diff_mean**2)
sm_estim_x_diff_P25 = percentile( diff(sm_x_mean_array_cm, window_dim), 25, repdim)
sm_estim_y_diff_P25 = percentile( diff(sm_y_mean_array_cm, window_dim), 25, repdim)
sm_L2_of_P25 = sqrt(sm_estim_x_diff_P25**2 + sm_estim_y_diff_P25**2)
sm_estim_P25 = sqrt(2) * 0.5*(sm_estim_x_diff_P25 + sm_estim_y_diff_P25)
sm_estim_x_diff_P75 = percentile( diff(sm_x_mean_array_cm, window_dim), 75, repdim)
sm_estim_y_diff_P75 = percentile( diff(sm_y_mean_array_cm, window_dim), 75, repdim)
sm_L2_of_P75 = sqrt(sm_estim_x_diff_P75**2 + sm_estim_y_diff_P75**2)
sm_estim_P75 = sqrt(2) * 0.5*(sm_estim_x_diff_P75 + sm_estim_y_diff_P75)

co_true_xdiff_mean = mean( diff(co_x_true_array_cm, window_dim), repdim)
co_true_ydiff_mean = mean( diff(co_y_true_array_cm, window_dim), repdim)
co_L2_diff_perframe = sqrt( diff(co_x_mean_array_cm, window_dim)**2 + diff(co_y_mean_array_cm, window_dim)**2)
co_mean_L2diff_acrossreps = mean(co_L2_diff_perframe, repdim)
co_std_L2diff_acrossreps = std(co_L2_diff_perframe, repdim)
co_estim_x_diff_mean = mean( diff(co_x_mean_array_cm, window_dim), repdim) # This is the way it SHOULD be done, not the way Pfeiffer & Foster did it
co_estim_y_diff_mean = mean( diff(co_y_mean_array_cm, window_dim), repdim)
co_estim_x_diff_std = std( diff(co_x_mean_array_cm, window_dim), repdim)
co_estim_y_diff_std = std( diff(co_y_mean_array_cm, window_dim), repdim)
co_L2_of_meandiff = sqrt( co_estim_x_diff_mean**2 + co_estim_y_diff_mean**2)

co_estim_x_diff_P25 = percentile( diff(co_x_mean_array_cm, window_dim), 25, repdim)
co_estim_y_diff_P25 = percentile( diff(co_y_mean_array_cm, window_dim), 25, repdim)
co_L2_of_P25 = sqrt(co_estim_x_diff_P25**2 + co_estim_y_diff_P25**2)
co_estim_x_diff_P75 = percentile( diff(co_x_mean_array_cm, window_dim), 75, repdim)
co_estim_y_diff_P75 = percentile( diff(co_y_mean_array_cm, window_dim), 75, repdim)
co_L2_of_P75 = sqrt(co_estim_x_diff_P75**2 + co_estim_y_diff_P75**2)


#subplot(n_plots_x, n_plots_y, 1)
ax = []
ax.append(subplot(n_plots_x, n_plots_y, 1))
ax_sm_meanL2=gca()
ax_sm_meanL2.fill_between(osc_phase[1:], sm_mean_L2diff_acrossreps - sm_std_L2diff_acrossreps, sm_mean_L2diff_acrossreps + sm_std_L2diff_acrossreps, color=col_biased, alpha=0.5, edgecolor='None') # ) # tab_red
line1 = plot(osc_phase[1:], sm_mean_L2diff_acrossreps, color=col_biased) # tab_red, tab_red)
line2 = plot(osc_phase[1:], sqrt( sm_true_xdiff_mean**2 + sm_true_ydiff_mean**2), 'k-', lw=linew)
ylabel('Step size \n [cm]', fontsize=fsize)
ax_sm_meanL2.set_xticks([osc_phase[0], osc_phase[int(0.5*n_windows)], osc_phase[-1]])
ax_sm_meanL2.set_xticklabels([0, '2$\pi$', '4$\pi$']) #
xlabel('Oscillation phase [rad]', fontsize=fsize, va='top')
ax_sm_meanL2.yaxis.grid()  
for label in ax_sm_meanL2.get_xticklabels() + ax_sm_meanL2.get_yticklabels(): 
    label.set_fontsize(fsize) 
legend((' Step size estimate \n (mean vector length)', ' True step size'), fontsize=fsize_small, loc=(0.075,-0.8))
ax.append(ax[0].twinx())
line3 = plot(osc_phase, 0.2 * np.nanmean(sm_tot_spikes, repdim) / (float(n_cells) * DeltaT_window_sec), '-', color=col_green) #, lw=linew )
ax21=gca()
ax21.yaxis.tick_right()
ax21.yaxis.set_label_position("right")
for label in ax21.get_xticklabels() + ax21.get_yticklabels(): 
    label.set_fontsize(fsize) 
    label.set_color(col_green) 
text(-0.35, 1.1, 'a', weight='bold', fontsize=1.5*fsize, transform=ax_sm_meanL2.transAxes)




ax_sm_meanL2_phase=subplot(n_plots_x, n_plots_y, 3, projection='polar')
imax = int(0.5*n_windows) #  n_windows #
sm_mean_L2diff_i = nan * ones(imax)
sm_true_step_i = nan * ones(imax)
sm_spikes_i = zeros(imax)
for i_phase in range(imax):
    nz_phase_i = nonzero(mod(osc_phase, 2*pi) == osc_phase[i_phase])[0]    
    nz_phase_i = nz_phase_i[ nonzero( nz_phase_i < len(sm_estim_x_diff_mean) ) ]
    sm_mean_L2diff_i[i_phase] = mean(sm_mean_L2diff_acrossreps[nz_phase_i]) 
    sm_true_step_i[i_phase] = sqrt( mean(sm_true_xdiff_mean[i_phase])**2 + mean(sm_true_ydiff_mean[i_phase])**2)
    sm_spikes_i[i_phase] = mean(sm_tot_spikes[:, nz_phase_i])
sm_spikes_i /= sm_spikes_i.max()
sm_spikes_i *= 15
#sm_mean_L2diff_i /= sm_mean_L2diff_i.max()
#sm_true_step_i /= sm_true_step_i.max()
#plot(osc_phase[0:imax], sm_mean_L2diff_i, '-', color=col_vermillion) # tab_red
#ylabel('Step size \n [cm]', fontsize=fsize)
#ylabel('Mean vector length of step', fontsize=fsize)
fig=gcf()
#fig.text(0.0, 0.905,'Mean vector length of steps', fontsize=fsize, rotation=90) # , weight='bold'
ax_sm_meanL2_phase.set_xticks(arange(0, 2*pi  , pi/6.0))
ax_sm_meanL2_phase.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])
#ax_sm_meanL2.set_yticks([0, 0, 25])
#ax_sm_meanL2_phase.set_yticklabels([])
ax_sm_meanL2_phase.set_yticks([ax_sm_meanL2_phase.axis()[2], ax_sm_meanL2_phase.axis()[3]])
ax_sm_meanL2_phase.set_yticklabels([]) #  
ax_sm_meanL2_phase.yaxis.grid()
for label in ax_sm_meanL2_phase.get_xticklabels() + ax_sm_meanL2_phase.get_yticklabels(): 
    label.set_fontsize(fsize) 
fig=gcf()
fig.text(0.103, 0.97,'Phase-locked movement', fontsize=1.25*fsize, weight='bold')
datalen = imax # n_windows
box_len = 3 # 2
box2 = 1.0 / box_len * append(ones(box_len), zeros(datalen-box_len)) # averaging across windows
sm_mean_L2diff_smoothed_i = convolve( append(sm_mean_L2diff_i, sm_mean_L2diff_i), box2)[box_len - 1 : datalen + box_len - 1]
offset = 0 # -1
plot(osc_phase[int(0.5*(box_len - 1) + offset) : int(len(box2) + 0.5*(box_len - 1) + offset)], sm_mean_L2diff_smoothed_i, '-', color=col_biased)
plot(osc_phase[0:imax], sm_true_step_i, '-', color='k')
plot(osc_phase[0:imax], sm_spikes_i, '-', color=col_green)
plot([osc_phase[0], osc_phase[-1]] , [sm_spikes_i[0], sm_spikes_i[-1]], '-', color=col_green)
text(-0.7, 1.1, 'c', weight='bold', fontsize=1.5*fsize, transform=ax_sm_meanL2_phase.transAxes)





#subplot(n_plots_x, n_plots_y, 2)
ax = []
ax.append(subplot(n_plots_x, n_plots_y, 2))
ax_co_meanL2=gca()
ax_co_meanL2.fill_between(osc_phase[1:], co_mean_L2diff_acrossreps - co_std_L2diff_acrossreps, co_mean_L2diff_acrossreps + co_std_L2diff_acrossreps, color=col_biased, alpha=0.5, edgecolor='None') # ) # tab_red
plot(osc_phase[1:], co_mean_L2diff_acrossreps, color=col_biased) # tab_red, tab_red)
plot(osc_phase[1:], sqrt( co_true_xdiff_mean**2 + co_true_ydiff_mean**2), 'k-', lw=linew)
ax_co_meanL2.set_xticks([osc_phase[0], osc_phase[int(0.5*n_windows)], osc_phase[-1]])
ax_co_meanL2.set_xticklabels([0, '2$\pi$', '4$\pi$']) #
#ax_co_meanL2.set_xticklabels([]) #
xlabel('Oscillation phase [rad]', fontsize=fsize, va='top')
ax_co_meanL2.yaxis.grid()
for label in ax_co_meanL2.get_xticklabels() + ax_co_meanL2.get_yticklabels(): 
    label.set_fontsize(fsize) 
ax.append(ax[0].twinx())
line3 = plot(osc_phase, 0.2 * np.nanmean(co_tot_spikes, repdim) / (float(n_cells) * DeltaT_window_sec), '-', color=col_green) #, lw=linew )
ax22=gca()
ax22.yaxis.tick_right()
ax22.yaxis.set_label_position("right")
ylabel('Population rate \n [sp/sec]', fontsize=fsize, color=col_green)
for label in ax22.get_xticklabels() + ax22.get_yticklabels(): 
    label.set_fontsize(fsize) 
    label.set_color(col_green) 
legend((' Population \n rate',''), fontsize=fsize_small, loc=(0.21, -0.68))#, loc='upper right')#, frameon=False)
text(-0.17, 1.1, 'b', weight='bold', fontsize=1.5*fsize, transform=ax_co_meanL2.transAxes)




ax_co_meanL2_polar=subplot(n_plots_x, n_plots_y, 4, projection='polar')
imax = int(0.5*n_windows) #  n_windows #
co_mean_L2diff_i = nan * ones(imax)
co_true_step_i = nan * ones(imax) 
co_spikes_i = zeros(imax)
for i_phase in range(imax):
    nz_phase_i = nonzero(mod(osc_phase, 2*pi) == osc_phase[i_phase])[0]    
    nz_phase_i = nz_phase_i[ nonzero( nz_phase_i < len(co_estim_x_diff_mean) ) ]
    co_mean_L2diff_i[i_phase] = mean(co_mean_L2diff_acrossreps[nz_phase_i]) 
    co_true_step_i[i_phase] = sqrt( mean(co_true_xdiff_mean[i_phase])**2 + mean(co_true_ydiff_mean[i_phase])**2)
    co_spikes_i[i_phase] = mean(co_tot_spikes[:, nz_phase_i])
plot(osc_phase[0:imax], co_true_step_i, '-', color='k')
co_spikes_i /= co_spikes_i.max()
co_spikes_i *= 15
plot(osc_phase[0:imax], co_spikes_i, '-', color=col_green)
ax_co_meanL2_polar.set_xticks(arange(0, 2*pi  , pi/6.0))
ax_co_meanL2_polar.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])

ax_co_meanL2_polar.set_yticks([ax_co_meanL2_polar.axis()[2], ax_co_meanL2_polar.axis()[3]])
ax_co_meanL2_polar.set_yticklabels([]) #  
ax_co_meanL2_polar.yaxis.grid()
for label in ax_co_meanL2_polar.get_xticklabels() + ax_co_meanL2_polar.get_yticklabels(): 
    label.set_fontsize(fsize) 
fig=gcf()
fig.text(0.56, 0.97,'Constant movement', fontsize=1.25*fsize, weight='bold')
datalen = imax # n_windows
box_len = 3 # 2
box2 = 1.0 / box_len * append(ones(box_len), zeros(datalen-box_len)) # averaging across windows
co_mean_L2diff_smoothed_i = convolve( append(co_mean_L2diff_i, co_mean_L2diff_i), box2)[box_len - 1 : datalen + box_len - 1]
offset = 0 # -1
plot(osc_phase[int(0.5*(box_len - 1) + offset) : int(len(box2) + 0.5*(box_len - 1) + offset)], co_mean_L2diff_smoothed_i, '-', color=col_biased) # tab_red
plot([osc_phase[0], osc_phase[-1]] , [co_spikes_i[0], co_spikes_i[-1]], '-', color=col_green)
text(-0.45, 1.1, 'd', weight='bold', fontsize=1.5*fsize, transform=ax_co_meanL2_polar.transAxes)



ymax_ax1234 = max(ax_sm_meanL2.axis()[3], ax_co_meanL2.axis()[3], 25)
ymax_old = ax_sm_meanL2.axis()[3]
ax_sm_meanL2.axis( [ax_sm_meanL2.axis()[0], osc_phase[-1], ax_sm_meanL2.axis()[2], ymax_ax1234])

ymax_tick = int(round(ymax_ax1234)) # 25 # 30 # 25
ax_sm_meanL2.set_yticks([0, ymax_tick])
ax_sm_meanL2.set_yticklabels([0, ymax_tick])
ax_co_meanL2.set_yticks([0, ymax_tick])
#ax_co_meanL2.set_yticklabels([]) #  


ax2max = 12 # 1.5 * max(ax21.axis()[3], ax22.axis()[3])
ax21.axis([ax21.axis()[0], ax21.axis()[1], ax21.axis()[2], ax2max])
ax22.axis([ax22.axis()[0], ax22.axis()[1], ax22.axis()[2], ax2max])

ax21.set_yticks([ax21.axis()[2], ax21.axis()[3]])
ax22.set_yticks([ax22.axis()[2], ax22.axis()[3]])


ax_co_meanL2.axis( [ax_co_meanL2.axis()[0], osc_phase[-1], ax_co_meanL2.axis()[2], ymax_ax1234])







tight_layout(pad=1.0, h_pad = 3, w_pad = 1)
#subplots_adjust(bottom=0.1, right=0.8, top=0.9)

savefig('comparison_newlayout_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.png', dpi=300)







figure(figsize=(4, 3), dpi=300)

ax = []
ax.append(subplot(n_plots_x, n_plots_y, 1))
sm_std_xy = sqrt(sm_estim_x_diff_std * sm_estim_x_diff_std)
ax_sm_L2ofMean=gca()
ax_sm_L2ofMean.fill_between(osc_phase[1:], sm_L2_of_meandiff - sm_std_xy, sm_L2_of_meandiff + sm_std_xy  , color=col_unbiased, alpha=0.5, edgecolor='None')
plot(osc_phase[1:], sm_L2_of_meandiff, col_unbiased)
plot(osc_phase[1:], sqrt( sm_true_xdiff_mean**2 + sm_true_ydiff_mean**2), 'k-', lw=linew)
legend(('bias-reduced estimate \n (vector length of the mean)', ' True step size'), fontsize=fsize_small, loc=(-0.015,-0.8))
ylabel('Step size \n [cm]', fontsize=fsize)
ax_sm_L2ofMean.set_xticks([osc_phase[0], osc_phase[int(0.5*n_windows)], osc_phase[-1]])
ax_sm_L2ofMean.set_xticklabels([0, '2$\pi$', '4$\pi$']) # 
ax_sm_L2ofMean.set_yticks([0, 0, 25])
ax_sm_L2ofMean.set_yticklabels(['', 0, 25])
for label in ax_sm_L2ofMean.get_xticklabels() + ax_sm_L2ofMean.get_yticklabels(): 
    label.set_fontsize(fsize) 
xlabel('Oscillation phase [rad]', fontsize=fsize)
ax.append(ax[0].twinx())
line3 = plot(osc_phase, 0.2 * np.nanmean(sm_tot_spikes, repdim) / (float(n_cells) * DeltaT_window_sec), '-', color=col_green) #, lw=linew )
ax21=gca()
ax21.yaxis.tick_right()
ax21.yaxis.set_label_position("right")
for label in ax21.get_xticklabels() + ax21.get_yticklabels(): 
    label.set_fontsize(fsize) 
    label.set_color(col_green) 
text(-0.35, 1.1, 'a', weight='bold', fontsize=1.5*fsize, transform=ax_sm_meanL2.transAxes)
fig=gcf()
fig.text(0.103, 0.97,'Phase-locked movement', fontsize=1.25*fsize, weight='bold')
fig.text(0.56, 0.97,'Constant movement', fontsize=1.25*fsize, weight='bold')


ax = []
ax.append(subplot(n_plots_x, n_plots_y, 2))
co_L2_of_meandiff = sqrt( co_estim_x_diff_mean**2 + co_estim_y_diff_mean**2)
co_std_xy = sqrt(co_estim_x_diff_std * co_estim_x_diff_std)
ax_co_L2ofMean=gca()
ax_co_L2ofMean.fill_between(osc_phase[1:], co_L2_of_meandiff - co_std_xy, co_L2_of_meandiff + co_std_xy  , color=col_unbiased, alpha=0.5, edgecolor='None')
plot(osc_phase[1:], co_L2_of_meandiff, col_unbiased)
plot(osc_phase[1:], sqrt( co_true_xdiff_mean**2 + co_true_ydiff_mean**2), 'k-', lw=linew)
ax_co_L2ofMean.set_xticks([osc_phase[0], osc_phase[int(0.5*n_windows)], osc_phase[-1]])
ax_co_L2ofMean.set_xticklabels([0, '2$\pi$', '4$\pi$']) # 
ax_co_L2ofMean.set_yticks([0, 0, 25])
ax_co_L2ofMean.set_yticklabels([]) #  
for label in ax_co_L2ofMean.get_xticklabels() + ax_co_L2ofMean.get_yticklabels(): 
    label.set_fontsize(fsize) 
xlabel('Oscillation phase [rad]', fontsize=fsize)

ax.append(ax[0].twinx())
line3 = plot(osc_phase, 0.2 * np.nanmean(co_tot_spikes, repdim) / (float(n_cells) * DeltaT_window_sec), '-', color=col_green) #, lw=linew )
ax22=gca()
ax22.yaxis.tick_right()
ax22.yaxis.set_label_position("right")
ylabel('Population rate \n [sp/sec]', fontsize=fsize, color=col_green)
for label in ax22.get_xticklabels() + ax22.get_yticklabels(): 
    label.set_fontsize(fsize) 
    label.set_color(col_green) 
legend((' Population \n rate',''), fontsize=fsize_small, loc=(0.21, -0.68))#, loc='upper right')#, frameon=False)
text(-0.17, 1.1, 'b', weight='bold', fontsize=1.5*fsize, transform=ax_co_meanL2.transAxes)



ax_sm_L2ofMean_polar=subplot(n_plots_x, n_plots_y, 3, projection='polar')
imax = int(0.5*n_windows) #  n_windows #
sm_L2ofmeandiff_i = nan * ones(imax)
sm_true_step_i = nan * ones(imax) 
for i_phase in range(imax):
    nz_phase_i = nonzero(mod(osc_phase, 2*pi) == osc_phase[i_phase])[0]    
    nz_phase_i = nz_phase_i[ nonzero( nz_phase_i < len(sm_estim_x_diff_mean) ) ]
    sm_L2ofmeandiff_i[i_phase] = sqrt( mean(sm_estim_x_diff_mean[nz_phase_i])**2 + mean(sm_estim_y_diff_mean[nz_phase_i])**2) 
    sm_true_step_i[i_phase] = sqrt( mean(sm_true_xdiff_mean[i_phase])**2 + mean(sm_true_ydiff_mean[i_phase])**2)
sm_std_xy = sqrt(sm_estim_x_diff_std * sm_estim_x_diff_std)
ax_sm_L2ofMean_polar.set_xticks(arange(0, 2*pi  , pi/6.0))
#ax_sm_L2ofMean_polar.set_xticklabels(['$0$', '','','$\pi / 2$','','','$\pi$','','','$3\pi / 2$','',''])
ax_sm_L2ofMean_polar.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])
ax_sm_L2ofMean_polar.set_yticks([ax_sm_L2ofMean_polar.axis()[2], ax_sm_L2ofMean_polar.axis()[3]])
ax_sm_L2ofMean_polar.set_yticklabels([]) #  
ax_sm_L2ofMean_polar.yaxis.grid()
for label in ax_sm_L2ofMean_polar.get_xticklabels() + ax_sm_L2ofMean_polar.get_yticklabels(): 
    label.set_fontsize(fsize) 
#fig.text(0.0, 0.4,'Vector length of mean steps', fontsize=fsize, rotation=90) # , weight='bold'
datalen = imax # n_windows
box_len = 3 # 2
box2 = 1.0 / box_len * append(ones(box_len), zeros(datalen-box_len)) # averaging across windows
sm_L2ofmeandiff_smoothed_i = convolve( append(sm_L2ofmeandiff_i, sm_L2ofmeandiff_i), box2)[box_len - 1 : datalen + box_len - 1]
plot(osc_phase[int(0.5*(box_len - 1) ): int(len(box2) + 0.5*(box_len - 1))], sm_L2ofmeandiff_smoothed_i, '-', color=col_unbiased)
plot(osc_phase[0:imax], sm_true_step_i, '-', color='k')
plot(osc_phase[0:imax], sm_spikes_i, '-', color=col_green)
plot([osc_phase[0], osc_phase[-1]] , [sm_spikes_i[0], sm_spikes_i[-1]], '-', color=col_green)
text(-0.7, 1.1, 'c', weight='bold', fontsize=1.5*fsize, transform=ax_sm_meanL2_phase.transAxes)


ax_co_L2ofMean_polar=subplot(n_plots_x, n_plots_y, 4, projection='polar')
imax = int(0.5*n_windows) #  n_windows #
co_L2ofmeandiff_i = nan * ones(imax)
co_true_step_i = nan * ones(imax) 
co_spikes_i = zeros(imax)
for i_phase in range(imax):
    nz_phase_i = nonzero(mod(osc_phase, 2*pi) == osc_phase[i_phase])[0]    
    nz_phase_i = nz_phase_i[ nonzero( nz_phase_i < len(co_estim_x_diff_mean) ) ]
    co_spikes_i[i_phase] = mean(co_tot_spikes[:, nz_phase_i])
    co_L2ofmeandiff_i[i_phase] = sqrt( mean(co_estim_x_diff_mean[nz_phase_i])**2 + mean(co_estim_y_diff_mean[nz_phase_i])**2) 
    co_true_step_i[i_phase] = sqrt( mean(co_true_xdiff_mean[i_phase])**2 + mean(co_true_ydiff_mean[i_phase])**2)
co_spikes_i /= co_spikes_i.max()
co_spikes_i *= 15
plot(osc_phase[0:imax], co_true_step_i, '-', color='k')
plot(osc_phase[0:imax], co_spikes_i, '-', color=col_green)
co_L2_of_meandiff = sqrt( co_estim_x_diff_mean**2 + co_estim_y_diff_mean**2)
co_std_xy = sqrt(co_estim_x_diff_std * co_estim_x_diff_std)
ax_co_L2ofMean_polar.set_xticks(arange(0, 2*pi  , pi/6.0))
ax_co_L2ofMean_polar.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])
ax_co_L2ofMean_polar.set_yticks([ax_sm_L2ofMean_polar.axis()[2], ax_sm_L2ofMean_polar.axis()[3]])
ax_co_L2ofMean_polar.set_yticklabels([]) #  
ax_co_L2ofMean_polar.yaxis.grid()
for label in ax_co_L2ofMean_polar.get_xticklabels() + ax_co_L2ofMean.get_yticklabels(): 
    label.set_fontsize(fsize) 
datalen = imax # n_windows
box_len = 3 # 2
box2 = 1.0 / box_len * append(ones(box_len), zeros(datalen-box_len)) # averaging across windows
co_L2ofmeandiff_smoothed_i = convolve( append(co_L2ofmeandiff_i, co_L2ofmeandiff_i), box2)[box_len - 1 : datalen + box_len - 1]
plot(osc_phase[int(0.5*(box_len - 1)) : int(len(box2) + 0.5*(box_len - 1))], co_L2ofmeandiff_smoothed_i, '-', color=col_unbiased) # 
plot([osc_phase[0], osc_phase[-1]] , [co_true_step_i[0], co_true_step_i[-1]], '-', color='k')
plot([osc_phase[0], osc_phase[-1]] , [co_spikes_i[0], co_spikes_i[-1]], '-', color=col_green)
text(-0.45, 1.1, 'd', weight='bold', fontsize=1.5*fsize, transform=ax_co_meanL2_polar.transAxes)




ymax_ax1234 = max(ax_sm_L2ofMean.axis()[3], ax_co_L2ofMean.axis()[3], 25)
ymax_old = ax_sm_L2ofMean.axis()[3]
ax_sm_L2ofMean.axis( [ax_sm_L2ofMean.axis()[0], osc_phase[-1], 0, ymax_ax1234])

ymax_tick = int(round(ymax_ax1234)) # 25 # 30 # 25
ax_sm_L2ofMean.set_yticks([0, ymax_tick])
ax_sm_L2ofMean.set_yticklabels([0, ymax_tick])
ax_co_L2ofMean.set_yticks([0, ymax_tick])
ax_co_L2ofMean.set_yticklabels([0, ymax_tick])

ax_co_L2ofMean.axis( [ax_co_meanL2.axis()[0], osc_phase[-1], ax_co_meanL2.axis()[2], ymax_ax1234])

ax2max = 12 # 1.5 * max(ax21.axis()[3], ax22.axis()[3])
ax21.axis([ax21.axis()[0], ax21.axis()[1], ax21.axis()[2], ax2max])
ax22.axis([ax22.axis()[0], ax22.axis()[1], ax22.axis()[2], ax2max])

ax21.set_yticks([ax21.axis()[2], ax21.axis()[3]])
ax22.set_yticks([ax22.axis()[2], ax22.axis()[3]])




tight_layout(pad=1.0, h_pad = 3, w_pad = 1)
#subplots_adjust(bottom=0.1, right=0.8, top=0.9)
savefig('comparison2_newlayout_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.png', dpi=300)



ioff()
show()
