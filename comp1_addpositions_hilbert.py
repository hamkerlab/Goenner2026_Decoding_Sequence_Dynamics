from pylab import *
import numpy as np
import scipy.stats as st
import new_colormaps as nc
#from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
from matplotlib import gridspec
from scipy.signal import hilbert

from sim_params_seq import params_id, n_grid_cm, DeltaT_window_sec, overlap, n_windows, n_reps, n_osc_cycles, sigma_code_cm, x_step_cm, y_step_cm, max_rate_Hz, n_mincells
from circ_stats import corr_circ_lin

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

#tab_red    = col_vermillion # col_purple # col_orange #
#tab_blue   = col_blue
#col_blue2 = col_green # col_blue
#tab_blue = col_green


ion()

n_cells = 200


figure(figsize=(3.75, 5), dpi=300)

fsize = 6 # 3 # 6
fsize_small = 5 # 4 # 2 # 4
linew = 1.5 # 0.75 # 1.5
msize_big = 5 # 3 # 30 # 8
msize = 2

x_grid, y_grid = meshgrid(range(n_grid_cm), range(n_grid_cm))


frac_mincells = n_mincells / 200.0


mod_depth = 1.0

osc_phase = arange(0, n_windows) / float(n_windows) * n_osc_cycles * 2*pi



'''#
#file_co = open('data/sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','r')
file_co = open('data/sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','r')
co_x_true_array_cm = pickle.load(file_co)
co_y_true_array_cm = pickle.load(file_co)
co_x_mean_array_cm = pickle.load(file_co)
co_y_mean_array_cm = pickle.load(file_co)
co_tot_spikes = pickle.load(file_co)
co_var_names = pickle.load(file_co)
file_co.close()
'''
co_x_true_array_cm, co_y_true_array_cm, co_x_mean_array_cm, co_y_mean_array_cm, co_tot_spikes, co_var_names = np.load('data/sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps_npformat.npy', allow_pickle=True, encoding="bytes")

np.save('data/sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps_npformat_export20220624_co_x_true_array_cm.npy', co_x_true_array_cm, allow_pickle=True)

import pandas as pd
data_const = {}
data_const['x_true_array_cm'] = np.zeros(co_x_true_array_cm.shape)
#data_const['x_true_array_cm'] = co_x_true_array_cm
#data_const['y_true_array_cm'] = np.zeros(co_y_true_array_cm.shape)
#data_const['y_true_array_cm'] = co_y_true_array_cm
#data_const['x_mean_array_cm'] = np.zeros(co_x_mean_array_cm.shape)
#data_const['x_mean_array_cm'] = co_x_mean_array_cm
#data_const['y_mean_array_cm'] = np.zeros(co_y_mean_array_cm.shape)
#data_const['y_mean_array_cm'] = co_y_mean_array_cm

#df_const = pd.DataFrame(data=data_const) # Doesn't work!

'''#
#file_spmod = open('data/speedmod_'+str(mod_depth)+'_sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','r')
file_spmod = open('data/speedmod_'+str(mod_depth)+'_sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','r')
sm_x_true_array_cm = pickle.load(file_spmod)
sm_y_true_array_cm = pickle.load(file_spmod)
sm_x_mean_array_cm = pickle.load(file_spmod)
sm_y_mean_array_cm = pickle.load(file_spmod)
sm_tot_spikes = pickle.load(file_spmod)
sm_var_names = pickle.load(file_spmod)
file_spmod.close()
'''
sm_x_true_array_cm, sm_y_true_array_cm, sm_x_mean_array_cm, sm_y_mean_array_cm, sm_tot_spikes, sm_var_names = np.load('data/speedmod_'+str(mod_depth)+'_sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps_npformat.npy', allow_pickle=True, encoding="bytes")

repdim, window_dim = (0, 1)
n_plots_x = 3 # 2 # 4
n_plots_y = 2


sm_true_xdiff_mean = mean( diff(sm_x_true_array_cm, window_dim), repdim)
sm_true_ydiff_mean = mean( diff(sm_y_true_array_cm, window_dim), repdim)
sm_L2_diff_perframe = sqrt( diff(sm_x_mean_array_cm, window_dim)**2 + diff(sm_y_mean_array_cm, window_dim)**2)
sm_mean_L2diff_acrossreps = mean(sm_L2_diff_perframe, repdim) # Possible only if phases are equal across repetitions
sm_min_L2diff_acrossreps = sm_L2_diff_perframe.min(repdim)
sm_std_L2diff_acrossreps = std(sm_L2_diff_perframe, repdim)
sm_estim_x_diff_mean = mean( diff(sm_x_mean_array_cm, window_dim), repdim) # This is the way it SHOULD be done, not the way Pfeiffer & Foster did it
sm_estim_y_diff_mean = mean( diff(sm_y_mean_array_cm, window_dim), repdim)
sm_estim_x_diff_std = std( diff(sm_x_mean_array_cm, window_dim), repdim)
sm_estim_y_diff_std = std( diff(sm_y_mean_array_cm, window_dim), repdim)
sm_L2_of_meandiff = sqrt( sm_estim_x_diff_mean**2 + sm_estim_y_diff_mean**2)
sm_hilb = hilbert(sm_tot_spikes - sm_tot_spikes.mean())
sm_hilb_phase = arctan2(sm_hilb.imag, sm_hilb.real)


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
co_hilb = hilbert(co_tot_spikes - co_tot_spikes.mean())
co_hilb_phase = arctan2(co_hilb.imag, co_hilb.real)

phase_vals = np.arange(min(np.nanmin(sm_hilb_phase), np.nanmin(co_hilb_phase)), max(np.nanmax(sm_hilb_phase), np.nanmax(co_hilb_phase)) + np.pi/18.0, np.pi/18.0) # np.pi + ...
sm_phase_bins = np.digitize(sm_hilb_phase[:, :-1], phase_vals)
co_phase_bins = np.digitize(co_hilb_phase[:, :-1], phase_vals)

gs = gridspec.GridSpec(n_plots_x, n_plots_y, width_ratios = [1, 1])#, 1]) #, height_ratios=[1, 1, 1])#)# 

x_shift = -0.08

subplot(gs[0])
alpha=0.95
R = zeros(n_windows)
i_rep = 2 # 0 # min. 3 spikes for i_rep=0
print("min(sm_tot_spikes[i_rep, :]) = ", min(sm_tot_spikes[i_rep, :]))
for i_wind in range(n_windows):
    t_conf_lo, t_conf_up = st.t.interval(alpha, sm_tot_spikes[i_rep, i_wind]-1, 0, 1) # For Poisson activity
    # Konfidenzintervall eines normalverteilten Merkmals mit Stichprobenvarianz s^2:
    # [x_mean - t_conf * s / sqrt(n); x_mean + t_conf * s / sqrt(n) ] # Achtung, korrigierte Stichprobenvarianz benutzen!
    # CI_x_lower = x_mean + t_conf_lo * std_xy / sqrt(n_cells)
    R[i_wind] = abs(t_conf_lo * sigma_code_cm / sqrt(sm_tot_spikes[i_rep, i_wind]))
    t=arange(0, 2*pi, 0.01*pi)
plot(sm_x_true_array_cm[i_rep, :], sm_y_true_array_cm[i_rep, :], '|-', markersize=3*msize, lw=0.5*linew, color='k') # '0.5') 
plot(sm_x_mean_array_cm[i_rep, :], sm_y_mean_array_cm[i_rep, :], '.-', color=tab_red, markersize=1.2*msize, lw=0.5*linew)
axis('scaled')
ax_sm_seq = gca()
for label in ax_sm_seq.get_xticklabels() + ax_sm_seq.get_yticklabels(): 
    label.set_fontsize(fsize) 
ax_sm_seq.axis([0, 100, 0, 100])
ax_sm_seq.set_xticks([0, n_grid_cm])
ax_sm_seq.set_xticklabels([0, n_grid_cm])
ax_sm_seq.set_yticks([0, n_grid_cm])
ax_sm_seq.set_yticklabels([0, n_grid_cm])
xlabel('x position [cm]', fontsize=fsize)#, va='bottom')
ylabel('y position \n [cm]', fontsize=fsize, va='center')
fig=gcf()

fig.text(0.02, 0.91, 'a', weight='bold', fontsize=1.5*fsize)
legend(('True position', 'Decoded'), fontsize=0.75*fsize, loc='center', bbox_to_anchor=(0.5, -0.8)) #, loc=(1.0,0.4)) #  \n position



subplot(gs[1])
i_rep = 2 # 0 # min. 3 spikes for i_rep=0
plot(co_x_true_array_cm[i_rep, :], co_y_true_array_cm[i_rep, :], '|-', markersize=3*msize, lw=0.5*linew, color='k') # '0.5')
plot(co_x_mean_array_cm[i_rep, :], co_y_mean_array_cm[i_rep, :], '.-', color=tab_red, markersize=1.2*msize, lw=0.5*linew)
axis('scaled')
ax_co_seq = gca()
for label in ax_co_seq.get_xticklabels() + ax_co_seq.get_yticklabels(): 
    label.set_fontsize(fsize) 
ax_co_seq.axis([0, 100, 0, 100])
ax_co_seq.set_xticks([0, n_grid_cm])
ax_co_seq.set_xticklabels([0, n_grid_cm])
ax_co_seq.set_yticks([0, n_grid_cm])
ax_co_seq.set_yticklabels([0, n_grid_cm])
xlabel('x position [cm]', fontsize=fsize)#, va='bottom')
#title('Constant movement', fontsize=1.25*fsize, va='bottom', weight='bold', alpha=0) # $H_0$: 
#fig.text(0.61, 0.95,'Constant movement', fontsize=1.25*fsize, weight='bold') # $H_1$:  
#text(-0.15, 1.1, 'b', weight='bold', fontsize=1.5*fsize, transform=ax_co_seq.transAxes)
fig.text(0.5+x_shift, 0.91, 'b', weight='bold', fontsize=1.5*fsize)



ax = []
ax.append(subplot(gs[2])) # subplot(n_plots_x, n_plots_y, 3))
ax_sm_meanL2=gca()
ax_sm_meanL2.fill_between(osc_phase[1:], sm_mean_L2diff_acrossreps - sm_std_L2diff_acrossreps, sm_mean_L2diff_acrossreps + sm_std_L2diff_acrossreps, color=col_blue2, alpha=0.4, edgecolor='None') # ) # tab_red
sm_effspikes_i = zeros(n_windows-1)
for i_phase in range(n_windows - 1):
    temp_effspikes_i = zeros(n_reps)
    for i_rep in range(n_reps):
        temp_effspikes_i[i_rep] =  1.0 / (1.0/sm_tot_spikes[i_rep, i_phase] + 1.0/sm_tot_spikes[i_rep, i_phase+1])
    sm_effspikes_i[i_phase] = mean(temp_effspikes_i[i_rep])
#ax_sm_meanL2.fill_between(osc_phase[1:], sm_mean_L2diff_acrossreps - sm_std_L2diff_acrossreps / sqrt(sm_effspikes_i), sm_mean_L2diff_acrossreps + sm_std_L2diff_acrossreps / sqrt(sm_effspikes_i), color=col_blue2, alpha=0.4, edgecolor='None') # ) # tab_red
line2 = plot(osc_phase[1:], sqrt( sm_true_xdiff_mean**2 + sm_true_ydiff_mean**2), color='k') # , linestyle=(0, (2,2)) # ,'k--', lw=linew
line1 = plot(osc_phase[1:], sm_mean_L2diff_acrossreps, color=col_blue2) # tab_red, tab_red)
ylabel('Step size \n [cm]', fontsize=fsize)
ax_sm_meanL2.set_xticks([osc_phase[0], osc_phase[int(0.5*n_windows)], osc_phase[-1]])
ax_sm_meanL2.set_xticklabels([0, '2$\pi$', '4$\pi$']) #
xlabel('Oscillation phase [rad]', fontsize=fsize, va='center')#, va='top')
ax_sm_meanL2.yaxis.grid()  
for label in ax_sm_meanL2.get_xticklabels() + ax_sm_meanL2.get_yticklabels(): 
    label.set_fontsize(fsize) 
#legend((' Step size estimate \n (mean vector length)', ' True step size'), fontsize=fsize_small, loc='center', bbox_to_anchor=(0.5, -0.8)) #, loc=(0.075,-1.1))
legend((' True step size', ' Step size estimate \n (mean vector length)'), fontsize=fsize_small, loc='center', bbox_to_anchor=(0.5, -0.8)) #, loc=(0.075,-1.1))
ax.append(ax[0].twinx())
line3 = plot(osc_phase, 0.2 * np.nanmean(sm_tot_spikes, repdim) / (float(n_cells) * DeltaT_window_sec), '-', color=col_green) #, lw=linew )
ax21=gca()
ax21.yaxis.tick_right()
ax21.yaxis.set_label_position("right")
for label in ax21.get_xticklabels() + ax21.get_yticklabels(): 
    label.set_fontsize(fsize) 
    label.set_color(col_green) 
#text(-0.35, 1.1, 'a', weight='bold', fontsize=1.5*fsize, transform=ax_sm_meanL2.transAxes)
fig.text(0.02, 0.55, 'c', weight='bold', fontsize=1.5*fsize)



ax = []
ax.append(subplot(gs[3])) # n_plots_x, n_plots_y, 4))
ax_co_meanL2=gca()
ax_co_meanL2.fill_between(osc_phase[1:], co_mean_L2diff_acrossreps - co_std_L2diff_acrossreps, co_mean_L2diff_acrossreps + co_std_L2diff_acrossreps, color=col_blue2, alpha=0.4, edgecolor='None') # ) # tab_red
co_effspikes_i = zeros(n_windows-1)
for i_phase in range(n_windows - 1):
    temp_effspikes_i = zeros(n_reps)
    for i_rep in range(n_reps):
        temp_effspikes_i[i_rep] =  1.0 / (1.0/co_tot_spikes[i_rep, i_phase] + 1.0/co_tot_spikes[i_rep, i_phase+1])
    co_effspikes_i[i_phase] = mean(temp_effspikes_i[i_rep])
#ax_co_meanL2.fill_between(osc_phase[1:], co_mean_L2diff_acrossreps - co_std_L2diff_acrossreps / sqrt(co_effspikes_i), co_mean_L2diff_acrossreps + co_std_L2diff_acrossreps / sqrt(co_effspikes_i), color=col_blue2, alpha=0.4, edgecolor='None') # ) # tab_red
plot(osc_phase[1:], co_mean_L2diff_acrossreps, color=col_blue2) # tab_red, tab_red)
plot(osc_phase[1:], sqrt( co_true_xdiff_mean**2 + co_true_ydiff_mean**2), color='k')# , linestyle=(0, (2,2)) # ,'k--', lw=linew, 'k--', lw=linew)
ax_co_meanL2.set_xticks([osc_phase[0], osc_phase[int(0.5*n_windows)], osc_phase[-1]])
ax_co_meanL2.set_xticklabels([0, '2$\pi$', '4$\pi$']) #
xlabel('Oscillation phase [rad]', fontsize=fsize, va='center')#, va='top')
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
legend((' Population \n rate',''), fontsize=fsize_small, loc='center', bbox_to_anchor=(0.5, -0.8) ) # loc=(0.21, -1.1))#, loc='upper right')#, frameon=False)
#text(-0.17, 1.1, 'd', weight='bold', fontsize=1.5*fsize, transform=ax_co_meanL2.transAxes)
fig.text(0.5+x_shift, 0.55, 'd', weight='bold', fontsize=1.5*fsize)




ax_sm_meanL2_phase=subplot(gs[4], projection='polar') # n_plots_x, n_plots_y, 5, projection='polar')
#imax = int(0.5*n_windows) #  n_windows #
imax = int(n_windows / float(n_osc_cycles)) #  n_windows #

#sm_mean_L2diff_i = nan * ones(imax)
sm_true_step_i = nan * ones(imax)
#sm_spikes_i = zeros(imax)
for i_phase in range(imax):
#    nz_phase_i = nonzero(mod(osc_phase, 2*pi) == osc_phase[i_phase])[0]    
#    nz_phase_i = nz_phase_i[ nonzero( nz_phase_i < len(sm_estim_x_diff_mean) ) ]
#    sm_mean_L2diff_i[i_phase] = mean(sm_mean_L2diff_acrossreps[nz_phase_i]) 
    sm_true_step_i[i_phase] = sqrt( mean(sm_true_xdiff_mean[i_phase])**2 + mean(sm_true_ydiff_mean[i_phase])**2)
#    sm_spikes_i[i_phase] = mean(sm_tot_spikes[:, nz_phase_i])


# Use with Hilbert transform:
imax = len(phase_vals)
sm_mean_L2diff_i = nan * ones(len(phase_vals))
#sm_true_step_i = nan * ones(len(phase_vals))
sm_spikes_i = zeros(len(phase_vals))
sm_true_xdiff_mean_i = zeros(len(phase_vals))
sm_true_ydiff_mean_i = zeros(len(phase_vals))
sm_estim_x_diff_mean_i = nan * ones(len(phase_vals))
sm_estim_y_diff_mean_i = nan * ones(len(phase_vals))
sm_L2ofmeandiff_i = nan * ones(len(phase_vals))

for i_phbin in range(len(phase_vals)):
    nz_phase_i = np.nonzero(sm_phase_bins == i_phbin)
    if len(nz_phase_i) > 0:
        sm_mean_L2diff_i[i_phbin] = np.nanmean(sm_L2_diff_perframe[nz_phase_i])
        sm_true_xdiff_mean_i[i_phbin] = nanmean( diff(sm_x_true_array_cm, window_dim)[nz_phase_i] )
        sm_true_ydiff_mean_i[i_phbin] = nanmean( diff(sm_y_true_array_cm, window_dim)[nz_phase_i] )
        # sm_true_step_i[i_phbin] = sqrt( mean(sm_true_xdiff_mean_i[i_phbin])**2 + mean(sm_true_ydiff_mean_i[i_phbin])**2)
        sm_spikes_i[i_phbin] = nanmean(sm_tot_spikes[nz_phase_i])
        sm_estim_x_diff_mean_i[i_phbin] = nanmean(diff(sm_x_mean_array_cm, window_dim)[nz_phase_i])
        sm_estim_y_diff_mean_i[i_phbin] = nanmean(diff(sm_y_mean_array_cm, window_dim)[nz_phase_i])
        sm_L2ofmeandiff_i = sqrt(sm_estim_x_diff_mean_i**2 + sm_estim_y_diff_mean_i**2)

sm_spikes_i /= nanmax(sm_spikes_i)
sm_spikes_i *= 15
fig=gcf()
#ax_sm_meanL2_phase.set_xticks(arange(0, 2*pi + pi/6.0, pi/6.0))
ax_sm_meanL2_phase.set_xticks(arange(0, 2*pi, pi/6.0))
ax_sm_meanL2_phase.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])
ax_sm_meanL2_phase.set_yticks([ax_sm_meanL2_phase.axis()[2], ax_sm_meanL2_phase.axis()[3]])
ax_sm_meanL2_phase.set_yticklabels([]) #  
ax_sm_meanL2_phase.yaxis.grid()
for label in ax_sm_meanL2_phase.get_xticklabels() + ax_sm_meanL2_phase.get_yticklabels(): 
    label.set_fontsize(fsize) 
fig=gcf()
#fig.text(0.05, 0.97,'Phase-locked movement', fontsize=1.25*fsize, weight='bold') # constant hight_ratio
fig.text(0.035+0.3*x_shift, 0.97,'Phase-locked movement', fontsize=1.25*fsize, weight='bold')
datalen = imax # n_windows
box_len = 3 # 2
box2 = 1.0 / box_len * append(ones(box_len), zeros(datalen-box_len)) # averaging across windows

nonnan_ind = nonzero(isnan(sm_mean_L2diff_i) == False)
sm_mean_L2diff_i = sm_mean_L2diff_i[nonnan_ind]
sm_mean_L2diff_smoothed_i = convolve( append(sm_mean_L2diff_i, sm_mean_L2diff_i), box2)[box_len - 1 : datalen + box_len - 1]
offset = 0 # -1
#plot(osc_phase[int(0.5*(box_len - 1) + offset) : int(len(box2) + 0.5*(box_len - 1) + offset)], sm_mean_L2diff_smoothed_i, '-', color=col_blue2)
plot(osc_phase[0:int(n_windows / float(n_osc_cycles)) ], sm_true_step_i, '-', color='k')
#plot(osc_phase[0:imax], sm_spikes_i, '-', color=col_green)
#plot([osc_phase[0], osc_phase[-1]] , [sm_spikes_i[0], sm_spikes_i[-1]], '-', color=col_green)

plot(phase_vals, sm_mean_L2diff_smoothed_i, '-', color=col_blue2)
#plot(phase_vals, sm_true_step_i, '-', color='k')
plot(phase_vals, sm_spikes_i, '-', color=col_green)
#plot([sm_phase_bins[0], osc_phase[-1]] , [sm_spikes_i[0], sm_spikes_i[-1]], '-', color=col_green)


#text(-0.7, 1.1, 'c', weight='bold', fontsize=1.5*fsize, transform=ax_sm_meanL2_phase.transAxes)
fig.text(0.02, 0.2, 'e', weight='bold', fontsize=1.5*fsize)





ax_co_meanL2_polar=subplot(gs[5], projection='polar') # n_plots_x, n_plots_y, 6, projection='polar')
#imax = int(0.5*n_windows) #  n_windows #
imax = int(n_windows / float(n_osc_cycles)) #  n_windows #

'''#
co_mean_L2diff_i = nan * ones(imax)
co_true_step_i = nan * ones(imax) 
co_spikes_i = zeros(imax)
for i_phase in range(imax):
    nz_phase_i = nonzero(mod(osc_phase, 2*pi) == osc_phase[i_phase])[0]    
    nz_phase_i = nz_phase_i[ nonzero( nz_phase_i < len(co_estim_x_diff_mean) ) ]
    co_mean_L2diff_i[i_phase] = mean(co_mean_L2diff_acrossreps[nz_phase_i]) 
    co_true_step_i[i_phase] = sqrt( mean(co_true_xdiff_mean[i_phase])**2 + mean(co_true_ydiff_mean[i_phase])**2)
    co_spikes_i[i_phase] = mean(co_tot_spikes[:, nz_phase_i])
'''

# Use with Hilbert transform:
imax = len(phase_vals)
co_mean_L2diff_i = nan * ones(len(phase_vals))
co_true_step_i = nan * ones(len(phase_vals))
co_spikes_i = zeros(len(phase_vals))
co_true_xdiff_mean_i = zeros(len(phase_vals))
co_true_ydiff_mean_i = zeros(len(phase_vals))
co_estim_x_diff_mean_i = nan * ones(len(phase_vals))
co_estim_y_diff_mean_i = nan * ones(len(phase_vals))
co_L2ofmeandiff_i = nan * ones(len(phase_vals))

for i_phbin in range(len(phase_vals)):
    nz_phase_i = np.nonzero(co_phase_bins == i_phbin)
    if len(nz_phase_i) > 0:
        co_mean_L2diff_i[i_phbin] = np.nanmean(co_L2_diff_perframe[nz_phase_i])
        co_true_xdiff_mean_i[i_phbin] = nanmean( diff(co_x_true_array_cm, window_dim)[nz_phase_i] )
        co_true_ydiff_mean_i[i_phbin] = nanmean( diff(co_y_true_array_cm, window_dim)[nz_phase_i] )
        co_true_step_i[i_phbin] = sqrt( mean(co_true_xdiff_mean_i[i_phbin])**2 + mean(co_true_ydiff_mean_i[i_phbin])**2)
        co_spikes_i[i_phbin] = nanmean(co_tot_spikes[nz_phase_i])
        co_estim_x_diff_mean_i[i_phbin] = nanmean(diff(co_x_mean_array_cm, window_dim)[nz_phase_i])
        co_estim_y_diff_mean_i[i_phbin] = nanmean(diff(co_y_mean_array_cm, window_dim)[nz_phase_i])
        co_L2ofmeandiff_i = sqrt(co_estim_x_diff_mean_i**2 + co_estim_y_diff_mean_i**2)



#plot(osc_phase[0:imax], co_true_step_i, '-', color='k')
co_spikes_i /= nanmax(co_spikes_i)
co_spikes_i *= 15
#ax_co_meanL2_polar.set_xticks(arange(0, 2*pi + pi/6.0, pi/6.0))
ax_co_meanL2_polar.set_xticks(arange(0, 2*pi, pi/6.0))
ax_co_meanL2_polar.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])

ax_co_meanL2_polar.set_yticks([ax_co_meanL2_polar.axis()[2], ax_co_meanL2_polar.axis()[3]])
ax_co_meanL2_polar.set_yticklabels([]) #  
ax_co_meanL2_polar.yaxis.grid()
for label in ax_co_meanL2_polar.get_xticklabels() + ax_co_meanL2_polar.get_yticklabels(): 
    label.set_fontsize(fsize) 
fig=gcf()
#fig.text(0.53, 0.97,'Constant movement', fontsize=1.25*fsize, weight='bold') # constant height ratio
fig.text(0.5+0.8*x_shift, 0.97,'Constant movement', fontsize=1.25*fsize, weight='bold') 
datalen = imax # n_windows
box_len = 3 # 2
box2 = 1.0 / box_len * append(ones(box_len), zeros(datalen-box_len)) # averaging across windows

nonnan_ind = nonzero(isnan(co_mean_L2diff_i) == False)
co_mean_L2diff_i = co_mean_L2diff_i[nonnan_ind]
co_mean_L2diff_smoothed_i = convolve( append(co_mean_L2diff_i, co_mean_L2diff_i), box2)[box_len - 1 : datalen + box_len - 1]

offset = 0 # -1
#plot(osc_phase[0:imax], co_spikes_i, '-', color=col_green)
#plot(osc_phase[int(0.5*(box_len - 1) + offset) : int(len(box2) + 0.5*(box_len - 1) + offset)], co_mean_L2diff_smoothed_i, '-', color=col_blue2) # tab_red
#plot([osc_phase[0], osc_phase[-1]] , [co_spikes_i[0], co_spikes_i[-1]], '-', color=col_green)
#text(-0.45, 1.1, 'f', weight='bold', fontsize=1.5*fsize, transform=ax_co_meanL2_polar.transAxes)
fig.text(0.5+x_shift, 0.2, 'f', weight='bold', fontsize=1.5*fsize)

plot(phase_vals, co_mean_L2diff_smoothed_i, '-', color=col_blue2)
plot(phase_vals, co_true_step_i, '-', color='k')
plot(phase_vals, co_spikes_i, '-', color=col_green)




ymax_ax1234 = max(ax_sm_meanL2.axis()[3], ax_co_meanL2.axis()[3], 25)
ymax_old = ax_sm_meanL2.axis()[3]
ax_sm_meanL2.axis( [ax_sm_meanL2.axis()[0], osc_phase[-1], ax_sm_meanL2.axis()[2], ymax_ax1234])

ymax_tick = int(round(ymax_ax1234)) # 25 # 30 # 25
ax_sm_meanL2.set_yticks([0, ymax_tick])
ax_sm_meanL2.set_yticklabels([0, ymax_tick])
ax_co_meanL2.set_yticks([0, ymax_tick])

ax2max = 12 # 1.5 * max(ax21.axis()[3], ax22.axis()[3])
#ax2max = 1.5 * max(ax21.axis()[3], ax22.axis()[3])
ax21.axis([ax21.axis()[0], ax21.axis()[1], ax21.axis()[2], ax2max])
ax22.axis([ax22.axis()[0], ax22.axis()[1], ax22.axis()[2], ax2max])

ax21.set_yticks([ax21.axis()[2], ax21.axis()[3]])
ax22.set_yticks([ax22.axis()[2], ax22.axis()[3]])


ax_co_meanL2.axis( [ax_co_meanL2.axis()[0], osc_phase[-1], ax_co_meanL2.axis()[2], ymax_ax1234])


ax_polar_ymax = max(ax_co_meanL2_polar.axis()[3], ax_sm_meanL2_phase.axis()[3])
ax_co_meanL2_polar.axis([ax_co_meanL2_polar.axis()[0], ax_co_meanL2_polar.axis()[1], ax_co_meanL2_polar.axis()[2], ax_polar_ymax])
ax_sm_meanL2_phase.axis([ax_sm_meanL2_phase.axis()[0], ax_sm_meanL2_phase.axis()[1], ax_sm_meanL2_phase.axis()[2], ax_polar_ymax])



#tight_layout(pad=1.0, h_pad = 3, w_pad = 1)
#tight_layout(pad=1.5, h_pad = 3, w_pad = 1)

#subplots_adjust(bottom=0.02, top=0.9, wspace=0.5, hspace=1.5, left=0.15, right=0.85 ) # constant height_ratio
subplots_adjust(bottom=0.02, top=0.9, wspace=1.25+10*x_shift, hspace=1.5, left=0.15)#, right=0.85 ) #  constant width_ratio

#savefig('figs/comparison_addpositions_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.png', dpi=300)
savefig('figs/comparison_addpos_hilb_paramsid_'+str(params_id)+'.png', dpi=300)




# Smoothing?!

nonnan_ind = nonzero(isnan(sm_L2ofmeandiff_i) == False)
sm_L2ofmeandiff_i = sm_L2ofmeandiff_i[nonnan_ind]
sm_L2ofmeandiff_smoothed_i = convolve( append(sm_L2ofmeandiff_i, sm_L2ofmeandiff_i), box2)[box_len - 1 : datalen + box_len - 1]

nonnan_ind = nonzero(isnan(co_L2ofmeandiff_i) == False)
co_L2ofmeandiff_i = co_L2ofmeandiff_i[nonnan_ind]
co_L2ofmeandiff_smoothed_i = convolve( append(co_L2ofmeandiff_i, co_L2ofmeandiff_i), box2)[box_len - 1 : datalen + box_len - 1]




figure(figsize=(4, 4), dpi=300)

subplot(121, projection='polar')
#subplot(223, projection='polar')
ax_sm_meanL2_phase = gca()
plot(osc_phase[0:int(n_windows / float(n_osc_cycles)) ], sm_true_step_i, '-', color='k')
#plot(osc_phase[0:imax], sm_spikes_i, '-', color=col_green)
#plot(osc_phase[int(0.5*(box_len - 1) + offset) : int(len(box2) + 0.5*(box_len - 1) + offset)], sm_mean_L2diff_smoothed_i, '-', color=col_blue2) # tab_red
#plot([osc_phase[0], osc_phase[-1]] , [sm_spikes_i[0], sm_spikes_i[-1]], '-', color=col_green)

plot(phase_vals, sm_mean_L2diff_smoothed_i, '-', color=col_blue2) # sm_mean_L2diff_i
#plot(phase_vals, sm_mean_L2diff_i, '-', color=col_blue2) # [nonnan_ind]
#plot(phase_vals, sm_true_step_i, '-', color='k')
plot(phase_vals, sm_spikes_i, '-', color=col_green)

plot(phase_vals, sm_L2ofmeandiff_smoothed_i, '-', color=col_orange)


ax_sm_meanL2_phase.set_xticks(arange(0, 2*pi , pi/6.0)) # + pi/6.0
ax_sm_meanL2_phase.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])
ax_sm_meanL2_phase.set_yticks([ax_sm_meanL2_phase.axis()[2], ax_sm_meanL2_phase.axis()[3]])
ax_sm_meanL2_phase.set_yticklabels([]) #  
ax_sm_meanL2_phase.yaxis.grid()
for label in ax_sm_meanL2_phase.get_xticklabels() + ax_sm_meanL2_phase.get_yticklabels(): 
    label.set_fontsize(fsize) 

title('Phase-locked movement', fontsize=1.25*fsize, weight='bold')

print("phase_vals.min(), phase_vals.max() = ", phase_vals.min(), phase_vals.max())
print("osc_phase[0], osc_phase[int(n_windows / float(n_osc_cycles))] = ", osc_phase[0], osc_phase[int(n_windows / float(n_osc_cycles))])





subplot(122, projection='polar')
#subplot(224, projection='polar')
ax_co_meanL2_polar = gca()
#plot(osc_phase[0:imax], co_true_step_i, '-', color='k')
#plot(osc_phase[0:imax], co_spikes_i, '-', color=col_green)
#plot(osc_phase[int(0.5*(box_len - 1) + offset) : int(len(box2) + 0.5*(box_len - 1) + offset)], co_mean_L2diff_smoothed_i, '-', color=col_blue2) # tab_red
#plot([osc_phase[0], osc_phase[-1]] , [co_spikes_i[0], co_spikes_i[-1]], '-', color=col_green)

plot(phase_vals, co_mean_L2diff_smoothed_i, '-', color=col_blue2)
plot(phase_vals, co_true_step_i, '-', color='k')
plot(phase_vals, co_spikes_i, '-', color=col_green)
plot(phase_vals, co_L2ofmeandiff_smoothed_i, '-', color=col_orange)


#ax_co_meanL2_polar.set_xticks(arange(0, 2*pi + pi/6.0, pi/6.0))
ax_co_meanL2_polar.set_xticks(arange(0, 2*pi, pi/6.0))
ax_co_meanL2_polar.set_xticklabels(['0', '','','$\pi$ / 2','','','$\pi$','','','3$\pi$ / 2','',''])
ax_co_meanL2_polar.set_yticks([ax_co_meanL2_polar.axis()[2], ax_co_meanL2_polar.axis()[3]])
ax_co_meanL2_polar.set_yticklabels([]) #  
ax_co_meanL2_polar.yaxis.grid()
for label in ax_co_meanL2_polar.get_xticklabels() + ax_co_meanL2_polar.get_yticklabels(): 
    label.set_fontsize(fsize) 



ax_polar_ymax = max(ax_co_meanL2_polar.axis()[3], ax_sm_meanL2_phase.axis()[3])
ax_co_meanL2_polar.axis([ax_co_meanL2_polar.axis()[0], ax_co_meanL2_polar.axis()[1], ax_co_meanL2_polar.axis()[2], ax_polar_ymax])
ax_sm_meanL2_phase.axis([ax_sm_meanL2_phase.axis()[0], ax_sm_meanL2_phase.axis()[1], ax_sm_meanL2_phase.axis()[2], ax_polar_ymax])


#legend(('Decoded step size', 'True step size', 'Population rate'), fontsize=fsize, loc='center', bbox_to_anchor=(0.5, -0.3))
legend(('Decoded step size', 'True step size', 'Population rate', 'Improved estimate'), fontsize=fsize, loc='center', bbox_to_anchor=(0.5, -0.3))

title('Smooth movement', fontsize=1.25*fsize, weight='bold')

'''#
subplot(221)
plot(sm_x_true_array_cm[i_rep, :], sm_y_true_array_cm[i_rep, :], '|-', markersize=3*msize, lw=0.5*linew, color='k') # '0.5') 
plot(sm_x_mean_array_cm[i_rep, :], sm_y_mean_array_cm[i_rep, :], '.-', color=tab_red, markersize=1.2*msize, lw=0.5*linew)
axis('scaled')
ax_sm_seq = gca()
for label in ax_sm_seq.get_xticklabels() + ax_sm_seq.get_yticklabels(): 
    label.set_fontsize(fsize) 
ax_sm_seq.axis([0, 200, 0, 200])
ax_sm_seq.set_xticks([0, n_grid_cm])
ax_sm_seq.set_xticklabels([0, n_grid_cm])
ax_sm_seq.set_yticks([0, n_grid_cm])
ax_sm_seq.set_yticklabels([0, n_grid_cm])
xlabel('x position [cm]', fontsize=fsize)#, va='bottom')
ylabel('y position \n [cm]', fontsize=fsize, va='center')
fig=gcf()
#fig.text(0.02, 0.91, 'a', weight='bold', fontsize=1.5*fsize)
legend(('True position', 'Decoded'), fontsize=0.75*fsize, loc='center', bbox_to_anchor=(0.5, -0.8)) #, loc=(1.0,0.4)) #  \n position


subplot(222)
plot(co_x_true_array_cm[i_rep, :], co_y_true_array_cm[i_rep, :], '|-', markersize=3*msize, lw=0.5*linew, color='k') # '0.5') 
plot(co_x_mean_array_cm[i_rep, :], co_y_mean_array_cm[i_rep, :], '.-', color=tab_red, markersize=1.2*msize, lw=0.5*linew)
axis('scaled')
ax_co_seq = gca()
for label in ax_co_seq.get_xticklabels() + ax_co_seq.get_yticklabels(): 
    label.set_fontsize(fsize) 
ax_co_seq.axis([0, 200, 0, 200])
ax_co_seq.set_xticks([])
ax_co_seq.set_xticklabels([])
ax_co_seq.set_yticks([])
ax_co_seq.set_yticklabels([])
'''


savefig('figs/comparison_addpos_hilb_paramsid_'+str(params_id)+'_addplot.png', dpi=300)

ioff()
show()
