# Problem encountered, 27.3.17:
# When excluding all-zero (failed) Bayesian decoding frames,
# these frames are still used for the step size calculations in adjacent frames!!!

# In the Brian datamanager simulation data, I should be able to see the "max-jump"
# resulting from pop. vector decoding with 4ms/0.2ms windowing!

# bump shift changes with the pop. oscillation - look at those shifts associated to ratio of max./75th percentile?!

import numpy as np
from azizi_newInitLearn import xypos, xypos_maze, xy_index_maze
from plotting_func import rotated_plot, rotated_plot_bayes, rotated_plot_bayes_vector, combined_plot_Both
import pickle
from pylab import *

# Imports for data analysis:
#from brian import *
#from brian.tools.datamanager import *
import pickle
import plotting_func as pf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from matplotlib import colors
import new_colormaps as nc
from scipy.interpolate import griddata
#from scipy.stats import mode
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets.samples_generator import make_blobs
from scipy.signal import argrelextrema, argrelmax, argrelmin, hilbert, detrend


if __name__ == '__main__':
 
    #file_data = open('loop_data/dataloop_out_scalew1_sigmafanX1_sigmaCA3X1_window5ms-2ms_bayes_160bins_subs160.txt', 'r') # 12 networks, 40 trials
    file_data = open('loop_data/dataloop_out_scalew1_sigmafanX1_sigmaCA3X1_window5ms-2ms_bayes_160bins_subs160corr.txt', 'rb')     

    use_popspikes_all =  False # True # False 

    n_networks = 12 # 12
    n_trials = 40
    n_grid = 80
    maze_edge_cm = 110 # for n_grid = 80
    L_maze_cm = 200 + 2 * maze_edge_cm
    window_len = 5.0 # 20.0
    window_shift_ms = 2.0 # 5.0
    n_decframes = int((400.0 - window_len)/window_shift_ms)+1 # 198 # 77 # 77 timesteps for 400 ms simulation, 20ms windowing with 5ms shift; 198 for 5ms/2ms?!
    print("n_decframes = ", n_decframes)

    xm = np.zeros([n_networks, n_trials, n_decframes])
    ym = np.zeros([n_networks, n_trials, n_decframes])
    ind_end = np.zeros([n_networks, n_trials])
    xs = np.zeros([n_networks, n_trials, n_decframes])
    ys = np.zeros([n_networks, n_trials, n_decframes])
    std_xy = np.zeros([n_networks, n_trials, n_decframes])
    delta_std = np.zeros([n_networks, n_trials, n_decframes])
    diff_x = np.zeros([n_networks, n_trials, n_decframes])
    diff_y = np.zeros([n_networks, n_trials, n_decframes])
    diff_xy = np.nan * np.ones([n_networks, n_trials, n_decframes])
    pop_spikes = np.nan * np.ones([n_networks, n_trials, n_decframes])
    popspikes_all = np.nan * np.ones([n_networks, n_trials, n_decframes])
    hilb_phase = np.nan * np.ones([n_networks, n_trials, n_decframes])
    phase_bins = np.nan * np.ones([n_networks, n_trials, n_decframes])
    sp_count_bins = np.nan * np.ones([n_networks, n_trials, n_decframes])
    times = np.arange(window_len, 400 - window_len + window_shift_ms, window_shift_ms)
    totdist = np.zeros([n_networks, n_trials])
    it_steady = np.nan * np.ones([n_networks, n_trials])

    maxind_sp = np.zeros([n_networks, n_trials, n_decframes])
    popmax_sp = np.zeros([n_networks, n_trials, n_decframes])
    n_clust = np.zeros([n_networks, n_trials, n_decframes])
    clust_cent0 = np.zeros([n_networks, n_trials, n_decframes, 2])

    n_subsets = 40
    n_subset_neurons = 160
    n_spatial_bins = int(2 * n_grid) # 1*
    summed_prob = np.zeros([n_networks, n_trials, n_spatial_bins, n_spatial_bins])
    #summed_prob_2subsets = np.zeros([n_networks, n_trials, 2, n_spatial_bins, n_spatial_bins])
    summed_prob_2subsets = np.zeros([2, n_networks, n_trials, n_spatial_bins, n_spatial_bins]) # order changed
    x_array_2subsets = np.zeros([n_networks, n_trials, 2, n_decframes])
    y_array_2subsets = np.zeros([n_networks, n_trials, 2, n_decframes])
    x_array_all = np.nan * np.ones([n_networks, n_trials, n_decframes])
    y_array_all = np.nan * np.ones([n_networks, n_trials, n_decframes])
    expsum_single = np.zeros([n_networks, n_trials, n_spatial_bins, n_spatial_bins])
    diff_xy_bayes = np.nan * np.ones([n_networks, n_trials, n_decframes])
    diff_xy_bayes_subset = np.nan * np.ones([n_networks, n_trials, n_decframes])

    curr_dist_togoal = np.nan * np.ones([n_networks, n_trials, n_decframes])
    diff_dist_togoal = np.nan * np.ones([n_networks, n_trials, n_decframes])
    atan_diff_xy = np.nan * np.ones([n_networks, n_trials, n_decframes])
    totdiff_x  = np.nan * np.ones([n_networks, n_trials])
    totdiff_y  = np.nan * np.ones([n_networks, n_trials])
    atan_totdiff_xy = np.nan * np.ones([n_networks, n_trials])
    proj_diff_xy_startend = np.nan * np.ones([n_networks, n_trials, n_decframes])
    fraction_proj_diff_xy_startend = np.nan * np.ones([n_networks, n_trials, n_decframes])

    # CAUTION: Exclude the first 50ms (init.) from analysis!!!
    firstind = int((50.0 - window_len)/window_shift_ms)+1 # 7 # 0 
    print("firstind = ", firstind)

    for i_netw in range(n_networks):
        for i_trial in range(n_trials):
            r = pickle.load(file_data, encoding='latin1')
            xm[i_netw, i_trial, :] = r['x_mean']
            ym[i_netw, i_trial, :] = r['y_mean']
            ind_end[i_netw, i_trial] = xy_index_maze(xm[i_netw, i_trial, -2], ym[i_netw, i_trial, -2], n_grid, n_grid)

            xs[i_netw, i_trial, :] = r['x_std']
            ys[i_netw, i_trial, :] = r['y_std']
            std_xy[i_netw, i_trial, :] = np.sqrt(xs[i_netw, i_trial, :]**2 + ys[i_netw, i_trial, :])

            delta_std[i_netw, i_trial, :] = abs(r['x_std'] - r['y_std']) / ((r['x_std'] < r['y_std'])*r['x_std'] + (r['y_std'] <= r['x_std'])*r['y_std']) # normalize difference by min. of std in x- and y-direction

            diff_x[i_netw, i_trial, :-2] = np.diff(xm[i_netw, i_trial, :-1])
            diff_y[i_netw, i_trial, :-2] = np.diff(ym[i_netw, i_trial, :-1])
            diff_xy[i_netw, i_trial, :-2] = np.sqrt(diff_x[i_netw, i_trial, :-2]**2 + diff_y[i_netw, i_trial, :-2]**2)
            totdist[i_netw, i_trial] = np.sqrt((xm[i_netw, i_trial, -2] - xm[i_netw, i_trial, firstind])**2 + (ym[i_netw, i_trial, -2] - ym[i_netw, i_trial, firstind])**2) ## CORRECTED!
            nz_it = nonzero( sqrt( (xm[i_netw, i_trial, firstind:] - xm[i_netw, i_trial, -2])**2 + (ym[i_netw, i_trial, firstind:] - ym[i_netw, i_trial, -2])**2) < 0.5 )[0]
            if len(nz_it) > 0:
                it_steady[i_netw, i_trial] = nz_it[0]


            maxind_sp[i_netw, i_trial, :] = r['maxind_spikes']
            popmax_sp[i_netw, i_trial, :] = r['popmax_spikes']

            n_clust[i_netw, i_trial, :] = r['n_clust']
            clust_cent0[i_netw, i_trial, :, 0] = r['clust_cent0'][:, 0]
            clust_cent0[i_netw, i_trial, :, 1] = r['clust_cent0'][:, 1]

            pop_spikes[i_netw, i_trial, :] = r['pop_spikes']
            if use_popspikes_all:
                popspikes_all[i_netw, i_trial, :] = r['popspikes_all']
            else:
                popspikes_all[i_netw, i_trial, :] = r['pop_spikes']
            #hilb = hilbert( pop_spikes[i_netw, i_trial, :-2] - np.nanmean(pop_spikes[i_netw, i_trial, :-2]) ) # zero-mean!!!
            hilb = hilbert( popspikes_all[i_netw, i_trial, :-2] - np.nanmean(popspikes_all[i_netw, i_trial, :-2]) ) # zero-mean!!!
            hilb_phase[i_netw, i_trial, :-2] = arctan2(hilb.imag, hilb.real)
            phase_vals = np.arange(np.nanmin(hilb_phase), np.pi + np.pi/18.0, np.pi/18.0)
            phase_bins[i_netw, i_trial, :-2] = np.digitize(hilb_phase[i_netw, i_trial, :-2], phase_vals)
            spcount_vals = range(0, 3200, 100)
            #sp_count_bins[i_netw, i_trial, :-2] = np.digitize(pop_spikes[i_netw, i_trial, :-2], spcount_vals)
            sp_count_bins[i_netw, i_trial, :-2] = np.digitize(popspikes_all[i_netw, i_trial, :-2], spcount_vals)

            summed_prob[i_netw, i_trial, :, :] = np.reshape(r['summed_prob'], (n_spatial_bins, n_spatial_bins)) 
            #summed_prob_2subsets[i_netw, i_trial, 0, :, :] = np.reshape(r['summed_prob_2subs'], (n_spatial_bins, n_spatial_bins, 2))[:, :, 0]
            #summed_prob_2subsets[i_netw, i_trial, 1, :, :] = np.reshape(r['summed_prob_2subs'], (n_spatial_bins, n_spatial_bins, 2))[:, :, 1]
            summed_prob_2subsets[0, i_netw, i_trial, :, :] = np.reshape(r['summed_prob_2subs'], (n_spatial_bins, n_spatial_bins, 2))[:, :, 0] # order changed
            summed_prob_2subsets[1, i_netw, i_trial, :, :] = np.reshape(r['summed_prob_2subs'], (n_spatial_bins, n_spatial_bins, 2))[:, :, 1]  
            x_array_all[i_netw, i_trial, :] = r['x_array_all']
            y_array_all[i_netw, i_trial, :] = r['y_array_all']
            x_array_2subsets[i_netw, i_trial, 0, :] = r['x_array_2subs'][0, :]
            x_array_2subsets[i_netw, i_trial, 1, :] = r['x_array_2subs'][1, :]
            y_array_2subsets[i_netw, i_trial, 0, :] = r['y_array_2subs'][0, :]
            y_array_2subsets[i_netw, i_trial, 1, :] = r['y_array_2subs'][1, :]
            # Substituting zero coordinates by nan values:
            zero_ind = np.nonzero(x_array_all[i_netw, i_trial, :] == 0)[0]
            x_array_all[i_netw, i_trial, zero_ind] = np.nan * np.ones(len(zero_ind))
            y_array_all[i_netw, i_trial, zero_ind] = np.nan * np.ones(len(zero_ind))
            # Substituting zero coordinates by nan values:
            zero_ind = np.nonzero(x_array_2subsets[i_netw, i_trial, 0, :] == 0)[0]
            x_array_2subsets[i_netw, i_trial, 0, zero_ind] = np.nan * np.ones(len(zero_ind))
            y_array_2subsets[i_netw, i_trial, 0, zero_ind] = np.nan * np.ones(len(zero_ind))

            expsum_single[i_netw, i_trial, :, :] = np.reshape(r['expsum_single'], (n_spatial_bins, n_spatial_bins))
            diff_x_bayes = np.diff(x_array_all[i_netw, i_trial, :-1])
            diff_y_bayes = np.diff(y_array_all[i_netw, i_trial, :-1])
            diff_xy_bayes[i_netw, i_trial, :-2] = np.sqrt(diff_x_bayes**2 + diff_y_bayes**2)
            diff_x_bayes_subset = np.diff(x_array_2subsets[i_netw, i_trial, 0, :-1])
            diff_y_bayes_subset = np.diff(y_array_2subsets[i_netw, i_trial, 0, :-1])
            diff_xy_bayes_subset[i_netw, i_trial, :-2] = np.sqrt(diff_x_bayes_subset**2 + diff_y_bayes_subset**2)

            curr_xdist_togoal = xm[i_netw, i_trial, :] - xm[i_netw, i_trial, -2] 
            curr_ydist_togoal = ym[i_netw, i_trial, :] - ym[i_netw, i_trial, -2]
            #curr_dist_togoal[i_netw, i_trial, :] = np.sqrt((xm[i_netw, i_trial, :] - xm[i_netw, i_trial, -2])**2 + (ym[i_netw, i_trial, :] - ym[i_netw, i_trial, -2])**2) 
            curr_dist_togoal[i_netw, i_trial, :] = np.sqrt( curr_xdist_togoal**2 + curr_ydist_togoal**2 )
            diff_dist_togoal[i_netw, i_trial, :-2] = np.diff(curr_dist_togoal[i_netw, i_trial, :-1])
            atan_diff_xy[i_netw, i_trial, :] = np.arctan2( diff_y[i_netw, i_trial, :], diff_x[i_netw, i_trial, :])
            totdiff_x[i_netw, i_trial] = xm[i_netw, i_trial, firstind] - xm[i_netw, i_trial, -2]
            totdiff_y[i_netw, i_trial] = ym[i_netw, i_trial, firstind] - ym[i_netw, i_trial, -2]
            atan_totdiff_xy[i_netw, i_trial] = np.arctan2(totdiff_y[i_netw, i_trial], totdiff_x[i_netw, i_trial])
            proj_diff_xy_startend[i_netw, i_trial, :-2] =  diff_x[i_netw, i_trial, :-2] * curr_xdist_togoal[:-2] + diff_y[i_netw, i_trial, :-2] * curr_ydist_togoal[:-2]  # scalar product / projection onto the direction towards the goal
            for i_frame in range(n_decframes):
                if diff_xy[i_netw, i_trial, i_frame] > 0:
                    fraction_proj_diff_xy_startend[i_netw, i_trial, i_frame] = proj_diff_xy_startend[i_netw, i_trial, i_frame] / (diff_xy[i_netw, i_trial, i_frame] * curr_dist_togoal[i_netw, i_trial, i_frame])


    file_data.close()

    ion()

    # Comparison between PV and Bayes coordinates:
    #'''

    #inetw_nz, jtr_nz, kframe_nz = np.nonzero(x_array_all[:, :, :-1] <> 0) # 
    inetw_nz, jtr_nz, kframe_nz = np.nonzero(np.isnan(x_array_all[:, :, :-1]) == 0) # use this when substituing zero coordinates by nan above
    print("Fraction of non-zero coordinates (combined): ", len(inetw_nz) / float(n_networks * n_trials * n_decframes))
    rms_x = np.sqrt( ((L_maze_cm / float(n_grid) * ym[inetw_nz, jtr_nz, kframe_nz] - L_maze_cm / float(n_spatial_bins) * x_array_all[inetw_nz, jtr_nz, kframe_nz])**2).mean() ) # x and y are swapped for Bayesian decoding relative to PV decoding!
    print("rms_x (combined, zeros excluded) = ", rms_x)

    #rms_diff_xy = np.sqrt( ((L_maze_cm / float(n_grid) * diff_xy[inetw_nz, jtr_nz, kframe_nz] - L_maze_cm / float(n_spatial_bins) * diff_xy_bayes[inetw_nz, jtr_nz, kframe_nz])**2).mean() ) # x and y are swapped for Bayesian decoding relative to PV decoding!
    rms_diff_xy = np.sqrt( np.nanmean( ((L_maze_cm / float(n_grid) * diff_xy[inetw_nz, jtr_nz, kframe_nz] - L_maze_cm / float(n_spatial_bins) * diff_xy_bayes[inetw_nz, jtr_nz, kframe_nz])**2) ) ) # x and y are swapped for Bayesian decoding relative to PV decoding!
    print("rms_diff_xy (combined, zeros excluded) = ", rms_diff_xy)


    inetw_nz, jtr_nz, kframe_nz = np.nonzero(x_array_2subsets[:, :, 0, :-1] != 0)
    print("Fraction of non-zero coordinates (subset 0): ", len(inetw_nz) / float(n_networks * n_trials * n_decframes))
    rms_x = np.sqrt( ((L_maze_cm / float(n_grid) * ym[inetw_nz, jtr_nz, kframe_nz] - L_maze_cm / float(n_spatial_bins) * x_array_2subsets[inetw_nz, jtr_nz, 0, kframe_nz])**2).mean() ) # x and y are swapped for Bayesian decoding relative to PV decoding!
    print("rms_x (subset 0, zeros excluded) = ", rms_x)

    rms_x = np.sqrt( ((L_maze_cm / float(n_grid) * ym[:, :, :-1] - L_maze_cm / float(n_spatial_bins) * x_array_all[:, :, :-1])**2).mean() ) # x and y are swapped for Bayesian decoding relative to PV decoding!
    print("rms_x (all) = ", rms_x)
    print("Sum of nan values = ", (np.isnan(x_array_all[:, :, :])).sum() )

    indlist = {}
    # calc. mean of diff_xy and pop_spikes for each "phase bin":

    diff_xy_mean = np.nan * np.ones(len(phase_vals))
    diff_xy_bayes_mean = np.nan * np.ones(len(phase_vals))
    diff_xy_bayes_std = np.nan * np.ones(len(phase_vals))
    popspikes_mean = np.nan * np.ones(len(phase_vals))
    frame_count = np.nan * np.ones(len(phase_vals))
    prop_nonzero_pos = np.nan * np.ones(len(phase_vals))
    diff_goaldist_mean = np.nan * np.ones(len(phase_vals))
    diff_currang_totang_mean = np.nan * np.ones(len(phase_vals))
    fraction_proj_diff_xy_startend_mean = np.nan * np.ones(len(phase_vals))

    len_factor = L_maze_cm / float(n_grid)


    inetw_nz, jtr_nz, kframe_nz = np.nonzero(np.isnan(x_array_all[:, :, :-1]) == 0) # use this when substituing zero coordinates by nan above
    if np.nanmax(prop_nonzero_pos) > 0: prop_nonzero_pos /= np.nanmax(prop_nonzero_pos)

    plot_bump_dynamics = True # False # 
    if plot_bump_dynamics:

        # Preparing to exclude decoded "zero" coordinates:
        for i_phbin in range(len(phase_vals)):
        	inetw, jtr, t_frame = np.nonzero(phase_bins[:, :, :-2] == i_phbin)
        	# Data from all networks:
        	#'''#
        	indlist['bin'+str(i_phbin)] = inetw, jtr, t_frame
        	frame_count[i_phbin] = len(inetw)
        	diff_xy_mean[i_phbin] = len_factor * np.nanmean(diff_xy[inetw, jtr, t_frame])
        	diff_goaldist_mean[i_phbin] = len_factor * np.nanmean(diff_dist_togoal[inetw, jtr, t_frame])
        	diff_currang_totang_mean[i_phbin] = np.nanmean( atan_diff_xy[inetw, jtr, t_frame] - atan_totdiff_xy[inetw, jtr] )
        	fraction_proj_diff_xy_startend_mean[i_phbin] = np.nanmean( fraction_proj_diff_xy_startend[i_netw, jtr, t_frame]  )

        	#diff_xy_bayes_mean[i_phbin] = L_maze_cm / float(n_spatial_bins) * np.nanmean(diff_xy_bayes[inetw, jtr, t_frame])                  # NO DATA EXCLUDED
        	#diff_xy_bayes_mean[i_phbin] = L_maze_cm / float(n_spatial_bins) * np.nanmean(diff_xy_bayes[inetw_excl, jtr_excl, t_frame_excl])   # EXCLUDING DECODING FAILURES (DECODED ZERO COORDINATES)        
        	diff_xy_bayes_mean[i_phbin] = L_maze_cm / float(n_spatial_bins) * np.nanmean(diff_xy_bayes_subset[inetw, jtr, t_frame])   		 # DECODING FROM SUBSET 0,  NO DATA EXCLUDED        
        	diff_xy_bayes_std[i_phbin] = L_maze_cm / float(n_spatial_bins) * np.nanstd(diff_xy_bayes_subset[inetw, jtr, t_frame])   		 # DECODING FROM SUBSET 0,  NO DATA EXCLUDED        
        	#diff_xy_bayes_mean[i_phbin] = L_maze_cm / float(n_spatial_bins) * np.nanmean(diff_xy_bayes_subset[inetw_excl, jtr_excl, t_frame_excl]) # DECODING FROM SUBSET 0,  EXCLUDING DECODING FAILURES (DECODED ZERO COORDINATES)        

        	#popspikes_mean[i_phbin] = np.nanmean(pop_spikes[inetw, jtr, t_frame])
        	popspikes_mean[i_phbin] = np.nanmean(popspikes_all[inetw, jtr, t_frame])
        	if len(inetw) > 0: 
        	    #prop_nonzero_pos[i_phbin] = len(inetw_excl) / float(len(inetw))
        	    prop_nonzero_pos[i_phbin] = len( np.nonzero( np.isnan(diff_xy_bayes_subset[inetw, jtr, t_frame])==0 )[0] ) / float(len(inetw))
        	else:
        	    prop_nonzero_pos[i_phbin] = 0

    
        diff_xy_mean -= np.nanmin(diff_xy_mean)
        diff_xy_mean /= np.nanmax(diff_xy_mean)
        diff_xy_bayes_mean -= np.nanmin(diff_xy_bayes_mean)
        #diff_xy_bayes_mean /= np.nanmax(diff_xy_bayes_mean)
        #diff_xy_bayes_std  /= np.nanmax(diff_xy_bayes_mean)
        diff_xy_bayes_mean /= np.nanmax(diff_xy_bayes_std)
        diff_xy_bayes_std  /= np.nanmax(diff_xy_bayes_std)

        popspikes_mean -= np.nanmin(popspikes_mean)
        popspikes_mean /= np.nanmax(popspikes_mean)
        frame_count /= np.nanmax(frame_count)
        diff_goaldist_mean /= np.nanmax(abs( diff_goaldist_mean ))
        diff_currang_totang_mean /= max(abs( diff_currang_totang_mean ))
        fraction_proj_diff_xy_startend_mean /= max(abs( fraction_proj_diff_xy_startend_mean ))


        #figure(figsize=(4, 4), dpi=300)
        figure(figsize=(6, 4), dpi=300)
        fsize=6 # 8
        linew = 1.25 # 1.5
        msize = 2.0 # 2.5
        j_test = 7 # 15 # 15 is good for i_netw=0 !!! # 10 # 16 # 19 # 10 # 10!!!
        netw_test = 0
        t_end = int((150.0 - window_len)/window_shift_ms)+1

 

        subplot(232)
        #plot(times[firstind : t_end], len_factor * diff_xy[netw_test, j_test, firstind : t_end], 'k', lw=linew)
        #plot(times[firstind : t_end], len_factor * diff_xy[netw_test, j_test, firstind : t_end], 'ro', markersize=msize)
        #plot(times[firstind : t_end], L_maze_cm / float(n_spatial_bins) * diff_xy_bayes[0, j_test, firstind : t_end], 'b', lw=linew)
        #plot(times[firstind : t_end], L_maze_cm / float(n_spatial_bins) * diff_xy_bayes[0, j_test, firstind : t_end], 'bo', markersize=msize)
        plot(times[firstind : t_end], L_maze_cm / float(n_spatial_bins) * diff_xy_bayes_subset[netw_test, j_test, firstind : t_end], 'k', lw=linew) # 'b'
        plot(times[firstind : t_end], L_maze_cm / float(n_spatial_bins) * diff_xy_bayes_subset[netw_test, j_test, firstind : t_end], 'ro', markersize=msize) # 'bo'
        #'''#
        #for i_min in argrelmin(pop_spikes[netw_test, j_test,  : ], order=2)[0]:
        for i_min in argrelmin(popspikes_all[netw_test, j_test,  : ], order=2)[0]:
            #plot([times[i_min], times[i_min]], [0, 8], 'k--', lw=0.5) # decoding from all subsets
            if i_min < len(times):
                plot([times[i_min], times[i_min]], [0, 35], 'k--', lw=0.5) # [0, 45] # single subset 
        #'''
        ylabel('Movement [cm]', fontsize=fsize)
        xlabel('Time [ms]', fontsize=fsize)
        xlim([50, 150])
        ax=gca()
        ax.set_xticks(range(50, 200, 50))
        ax.set_xticklabels(range(50, 200, 50))
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(fsize)

        ax=subplot(233, projection='polar')
        imin_diffxy_mean = np.argmin(diff_xy_mean[1:])
        imin_diffxy_bayes_mean = np.argmin(diff_xy_bayes_mean[1:])
        phasevals_shifted = np.append(phase_vals[imin_diffxy_mean: ], phase_vals[: imin_diffxy_mean] + 2*np.pi)
        phasevals_shifted_bayes = np.append(phase_vals[imin_diffxy_bayes_mean: ], phase_vals[: imin_diffxy_bayes_mean] + 2*np.pi)
        diffxy_vals_shifted = np.append(diff_xy_mean[imin_diffxy_mean:], diff_xy_mean[:imin_diffxy_mean])
        diffxy_bayes_vals_shifted = np.append(diff_xy_bayes_mean[imin_diffxy_bayes_mean:], diff_xy_bayes_mean[:imin_diffxy_bayes_mean])
        diffxy_bayes_std_shifted = np.append(diff_xy_bayes_std[imin_diffxy_bayes_mean:], diff_xy_bayes_std[:imin_diffxy_bayes_mean])
        # Cardioid plot
        #plot(phasevals_shifted, diffxy_vals_shifted, 'b', lw=linew)
        plot(phasevals_shifted_bayes, diffxy_bayes_vals_shifted, 'k', lw=linew)
        #errorbar(phasevals_shifted_bayes, diffxy_bayes_vals_shifted, xerr=0.0, yerr=diffxy_bayes_std_shifted, lw=linew, color='k') # doesn't work!
        #plot(phase_vals, diff_xy_mean, 'b', lw=linew)
        #plot(phase_vals, diff_xy_bayes_mean, 'k', lw=linew)
        plot(phase_vals, popspikes_mean / np.nanmax(popspikes_mean), 'r', lw=linew)
        #plot(phase_vals, prop_nonzero_pos, 'g', lw=linew)
        #plot(phase_vals, frame_count, 'y', lw=linew)

        # Close "missing" values in the cardioid plot:
        nan_index = np.nonzero(np.isnan(diffxy_vals_shifted))[0][0]
        nan_index_bayes = np.nonzero(np.isnan(diffxy_bayes_vals_shifted))[0][0]
        #if nan_index > 0:
        #    plot([phasevals_shifted[nan_index-1], phasevals_shifted[nan_index+1]], [diffxy_vals_shifted[nan_index-1], diffxy_vals_shifted[nan_index+1]], 'b', lw=linew)
        if nan_index_bayes > 0 and nan_index_bayes < len(phasevals_shifted_bayes)-1:
            plot([phasevals_shifted_bayes[nan_index_bayes-1], phasevals_shifted_bayes[nan_index_bayes+1]], [diffxy_bayes_vals_shifted[nan_index_bayes-1], diffxy_bayes_vals_shifted[nan_index_bayes+1]], 'k', lw=linew)
        plot([phase_vals[1], phase_vals[-1]], [popspikes_mean[1] / np.nanmax(popspikes_mean), popspikes_mean[-1] / np.nanmax(popspikes_mean)], 'r', lw=linew)
        #plot([phase_vals[1], phase_vals[-1]], [prop_nonzero_pos[1], prop_nonzero_pos[-1]], 'g', lw=linew)

        # Add arrow for weighted circular mean:
        #shift_wmean = np.nansum(phasevals_shifted * diffxy_vals_shifted) / np.nansum(diffxy_vals_shifted)
        #ax.arrow(shift_wmean,0, 0, 0.81, head_width=0.05, fc='b', ec='b') # , width=0.05 # , head_length=0.2 # ERRORS when dy extends beyound 0.81?!
        shift_wmean_bayes = np.nansum(phasevals_shifted_bayes * diffxy_bayes_vals_shifted) / np.nansum(diffxy_bayes_vals_shifted)
        ax.arrow(shift_wmean_bayes,0, 0, 0.81, head_width=0.05, fc='k', ec='k')
        popspikes_wmean = np.nansum(phase_vals * popspikes_mean) / np.nansum(popspikes_mean)
        ax.arrow(popspikes_wmean,0, 0, 0.81, head_width=0.05, fc='r', ec='r')
        ax.set_xticks(arange(0, 2*pi, pi/6.0))
        ax.set_xticklabels(['0$\degree$', '', '', '90$\degree$', '', '', '180$\degree$', '', '', '270$\degree$', '', ''])
        ax.set_yticks([])
        ax.set_yticklabels([])
        #legend(('Norm. step size', 'Norm. spike count'), loc='lower center', fontsize=6)
        #legend(('Norm. PV step size', 'Norm. Bayes step', 'Norm. spike count'), loc='lower center', fontsize=3) # 6
        #legend(('Norm. PV step size', 'Norm. Bayes step', 'Norm. spike count','Prop. nonzero dec.'), loc='lower center', fontsize=3)
        legend(('Norm. step size', 'Norm. spike count'), loc='lower center', fontsize=6) # 6
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(fsize)

        #plot(phasevals_shifted_bayes, diffxy_bayes_vals_shifted + diffxy_bayes_std_shifted, 'k--', lw=linew)
        #plot(phasevals_shifted_bayes, diffxy_bayes_vals_shifted - diffxy_bayes_std_shifted, 'k--', lw=linew)
        plot(phasevals_shifted_bayes, diffxy_bayes_std_shifted, 'k--', lw=linew)


        print("diffxy_bayes_vals_shifted = ", diffxy_bayes_vals_shifted)
        print("diffxy_bayes_std_shifted = ", diffxy_bayes_std_shifted)
        #subplot(234)
        #title('Std. of movement steps')

        subplot(231) # 234
 
        matshow(summed_prob_2subsets[0, netw_test, j_test, :, :], fignum=False, origin='lower', cmap=nc.inferno)
        lastind = int((250.0 - window_len)/window_shift_ms)+1
        plot( 0.25 * (y_array_2subsets[netw_test, j_test, 0, firstind : lastind-3] + y_array_2subsets[netw_test, j_test, 0, firstind+1 : lastind-2] + y_array_2subsets[netw_test, j_test, 0, firstind+2 : lastind-1] + y_array_2subsets[0, j_test, 0, firstind+3 : lastind]), 0.25*(x_array_2subsets[netw_test, j_test, 0, firstind : lastind-3] + x_array_2subsets[netw_test, j_test, 0, firstind+1 : lastind-2] + x_array_2subsets[netw_test, j_test, 0, firstind+2 : lastind-1]+  x_array_2subsets[netw_test, j_test, 0, firstind+3 : lastind]) , '-', color='w', lw=0.75) # 1.5 too broad

        #xlim([maze_edge_cm/420.0 * n_grid , (420-maze_edge_cm) / 420.0 * n_grid])
        #ylim([maze_edge_cm/420.0 * n_grid , (420-maze_edge_cm) / 420.0 * n_grid])
        xlim([maze_edge_cm/420.0 * n_spatial_bins , (420-maze_edge_cm) / 420.0 * n_spatial_bins])
        ylim([maze_edge_cm/420.0 * n_spatial_bins , (420-maze_edge_cm) / 420.0 * n_spatial_bins])
        ax=gca()
        ax.set_xticks([maze_edge_cm/420.0 * n_spatial_bins , (420-maze_edge_cm) / 420.0 * n_spatial_bins])
        ax.set_xticklabels([0, 2], position = (0, -0.2)) # (0, -0.1): too high!
        ax.set_yticks([maze_edge_cm/420.0 * n_spatial_bins , (420-maze_edge_cm) / 420.0 * n_spatial_bins])
        ax.set_yticklabels([0, 2])
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(fsize)
        xlabel('x Position [m]', fontsize=fsize)
        ylabel('y Position [m]', fontsize=fsize)
        axins = inset_axes(ax, width="7.5%", height="100%",loc=3, bbox_to_anchor=(1.02, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = colorbar(cax=axins, orientation = 'vertical')
        cbar.set_ticks([0, cbar.vmax])
        cbar.set_ticklabels([0, int(np.round(cbar.vmax, 1))])
        cbar.ax.set_ylabel('Posterior probability sum', fontsize=fsize) #, position=(0, 0.5), va='bottom',)# ha='left') # , rotation='vertical'
        for label in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels(): 
            label.set_fontsize(fsize) # 5




        subplot(235)
        plot(times[firstind : t_end], 1 / 6400.0 * pop_spikes[0, j_test, firstind : t_end], 'k', lw=linew)
        #'''#
        #for i_min in argrelmin(pop_spikes[0, j_test,  : ], order=2)[0]:
        for i_min in argrelmin(popspikes_all[netw_test, j_test,  : ], order=2)[0]:
            if i_min < len(times):
                plot([times[i_min], times[i_min]], [0, 0.25], 'k--', lw=0.5) # [0, 0.3]
        #'''
        ylabel('Spike probability', fontsize=fsize)
        xlabel('Time [ms]', fontsize=fsize)
        xlim([50, 150])
        ax=gca()
        ax.set_xticks(range(50, 200, 50))
        ax.set_xticklabels(range(50, 200, 50))
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(fsize)

        subplot(236)	
        # Step size as a function of spike count: Create "spike count bins" with np.digitize, calc. percentiles of diff_xy for each bin
        p_less = np.nan * np.ones(len(spcount_vals))
        p_greater = np.nan * np.ones(len(spcount_vals))
        p_less_bayes = np.nan * np.ones(len(spcount_vals))
        p_greater_bayes = np.nan * np.ones(len(spcount_vals))
        frame_count_sp = np.nan * np.ones(len(spcount_vals))
        thresh_less = 2.0
        thresh_greater = 5.0 # 4.0
        thresh_less_bayes = 10.0 # 12.0 # 2.0
        thresh_greater_bayes = 20.0 # 40.0

        for i_spbin in range(len(spcount_vals)):
            inetw, jtr, t_frame = np.nonzero(sp_count_bins[:, :, :-2] == i_spbin)
            frame_count_sp[i_spbin] = len(inetw)
            i_lastsmaller = 0
            i_lastgreater = 100
            i_lastsmaller_bayes = 0
            i_lastgreater_bayes = 100

            print("i_spbin, len(np.nonzero( isnan(diff_xy_bayes_subset[inetw, jtr, t_frame])==0)[0]) = ", i_spbin, len(np.nonzero( isnan(diff_xy_bayes_subset[inetw, jtr, t_frame])==0)[0]))

            if len(inetw) > 0:
                for i_perc in range(0, 100, 1):
                    if len_factor * np.percentile( diff_xy[inetw, jtr, t_frame], i_perc) < thresh_less: i_lastsmaller = i_perc
                p_less[i_spbin] = i_lastsmaller
                for i_perc in range(100, 0, -1):
                    if len_factor * np.percentile( diff_xy[inetw, jtr, t_frame], i_perc) >= thresh_greater: i_lastgreater = i_perc
                p_greater[i_spbin] = i_lastgreater
	        # For Bayesian decoding
            if len(np.nonzero( isnan(diff_xy_bayes_subset[inetw, jtr, t_frame])==0)[0]) > 0:
                #'''#
                inetw_nz, jtr_nz, kframe_nz = np.nonzero(np.isnan(diff_xy_bayes_subset[:, :, :-1]) == 0) # use this when substituing zero coordinates by nan above
                inetw_excl, jtr_excl, t_frame_excl = [], [], []
                for i_nz in range(len(inetw)):
            	    for i_nzlist in np.nonzero(inetw[i_nz] == inetw_nz)[0]: # i_nzlist should iterate over all network index values for which there is an overlap between inetw_nz (non-zero pos. decoding) and inetw (matching the current phase)
            	        if jtr_nz[i_nzlist] == jtr[i_nz] and kframe_nz[i_nzlist] == t_frame[i_nz]:
            	            inetw_excl.append(inetw[i_nz])
            	            jtr_excl.append(jtr[i_nz])
            	            t_frame_excl.append(t_frame[i_nz])
                #'''

                for i_perc in range(0, 100, 1):
                    #if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes[inetw, jtr, t_frame], i_perc) < thresh_less: i_lastsmaller_bayes = i_perc
                    #if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes_subset[inetw, jtr, t_frame], i_perc) < thresh_less_bayes: i_lastsmaller_bayes = i_perc
                    #if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes_subset[np.nonzero( isnan(diff_xy_bayes_subset[inetw, jtr, t_frame])==0)[0]], i_perc)  < thresh_less_bayes: i_lastsmaller_bayes = i_perc
                    if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes_subset[inetw_excl, jtr_excl, t_frame_excl], i_perc)  < thresh_less_bayes: i_lastsmaller_bayes = i_perc # Excluding nan values
                p_less_bayes[i_spbin] = i_lastsmaller_bayes
                for i_perc in range(100, 0, -1):
                    #if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes[inetw, jtr, t_frame], i_perc) >= thresh_greater: i_lastgreater_bayes = i_perc
                    #if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes_subset[inetw, jtr, t_frame], i_perc) >= thresh_greater_bayes: i_lastgreater_bayes = i_perc
                    #if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes_subset[np.nonzero( isnan(diff_xy_bayes_subset[inetw, jtr, t_frame])==0)[0]], i_perc)  >= thresh_greater_bayes: i_lastgreater_bayes = i_perc
                    if L_maze_cm / float(n_spatial_bins) * np.percentile( diff_xy_bayes_subset[inetw_excl, jtr_excl, t_frame_excl], i_perc) >= thresh_greater_bayes: i_lastgreater_bayes = i_perc  # Excluding nan values
                p_greater_bayes[i_spbin] = i_lastgreater_bayes

     
        #plot(spcount_vals, 0.01 * p_less, 'r', lw=linew)
        #plot(spcount_vals, 1 - 0.01 * p_greater, 'b', lw=linew)
        plot(spcount_vals, 0.01 * p_less_bayes, 'r', lw=linew)
        plot(spcount_vals, 1 - 0.01 * p_greater_bayes, 'b', lw=linew)
        #plot(spcount_vals, frame_count_sp / np.nanmax(frame_count_sp), 'k')
        #boxplot([])
        xlabel('Number of spikes', fontsize=fsize)
        ylabel('Probability', fontsize=fsize)
        ax=gca()
        xlim([0, 2500]) # 3000
        ylim([0, 0.8]) # necessary?
        ax.set_xticks([spcount_vals[1], 2500]) # 3000
        ax.set_xticklabels([spcount_vals[1], 2500]) # 2500
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(fsize)
        ##legend(('Step size\n $<$ '+str(int(thresh_less))+' cm', 'Step size\n $>$ '+str(int(thresh_greater))+' cm'), fontsize=6, loc='lower right')
        #legend(('Step size\n $<$ '+str(int(thresh_less))+' cm', 'Step size\n $>$ '+str(int(thresh_greater))+' cm', 'Bayes step \n $<$ '+str(int(thresh_less_bayes))+' cm', 'Bayes step \n $>$ '+str(int(thresh_greater_bayes))+' cm'), loc='lower right', fontsize=3) # 6
        #legend(('Step size \n $<$ '+str(int(thresh_less_bayes))+' cm', 'Step size \n $>$ '+str(int(thresh_greater_bayes))+' cm'), loc='lower right', fontsize=6) # 6
        legend(('Step size \n $<$ '+str(int(thresh_less_bayes))+' cm', 'Step size \n $>$ '+str(int(thresh_greater_bayes))+' cm'), loc=(0.5, 0.15), fontsize=6) # (0.5, 0.35)

        tight_layout()

        savefig('plots/bump_motion_dynamics_noexclude_sigmafanX1_sigmaCA3X1.png', dpi=300)


        figure()
        print("diff_goaldist_mean = ", diff_goaldist_mean)
        print("popspikes_mean / np.nanmax(popspikes_mean) = ", popspikes_mean / np.nanmax(popspikes_mean))
        subplot(161)
        plot(diff_goaldist_mean, 'b')
        plot(popspikes_mean / np.nanmax(popspikes_mean), 'r')
        ax=subplot(162, projection='polar')
        imin_diffgoaldist_mean = np.argmin(diff_goaldist_mean[1:])
        phasevals_shifted = np.append(phase_vals[imin_diffgoaldist_mean: ], phase_vals[: imin_diffgoaldist_mean] + 2*np.pi)
        diffgoaldist_shifted = np.append(diff_goaldist_mean[imin_diffgoaldist_mean:], diff_goaldist_mean[:imin_diffgoaldist_mean])
        print("diffgoaldist_shifted = ", diffgoaldist_shifted)
        # Cardioid plot
        #plot(phasevals_shifted, -1 * diffgoaldist_shifted, 'b', lw=linew)
        plot(phasevals_shifted, -1 * np.clip(diffgoaldist_shifted, -inf, 0), 'b', lw=linew)
        plot(phasevals_shifted,  1 * np.clip(diffgoaldist_shifted, 0, inf), 'b--.', lw=linew)
        plot(phase_vals, popspikes_mean / np.nanmax(popspikes_mean), 'r', lw=linew)
        legend(('Towards end', 'Away from end', '# spikes'),fontsize=6)
        title('Reduction in distance to end point')
        subplot(163)
        plot(diff_currang_totang_mean, 'b')
        print("diff_currang_totang_mean = ", diff_currang_totang_mean)
        plot(popspikes_mean / np.nanmax(popspikes_mean), 'r')
        ax=subplot(164, projection='polar')
        imin_diffang_mean = np.argmin(diff_currang_totang_mean[1:])
        phasevals_shifted = np.append(phase_vals[imin_diffang_mean: ], phase_vals[: imin_diffang_mean] + 2*np.pi)
        diffgoalang_shifted = np.append(diff_currang_totang_mean[imin_diffang_mean:], diff_currang_totang_mean[:imin_diffang_mean])
        print("diffgoalang_shifted = ", diffgoalang_shifted)
        # Cardioid plot
        #plot(phasevals_shifted, abs(diffgoalang_shifted), 'b', lw=linew)
        plot(phasevals_shifted, clip(diffgoalang_shifted, 0, inf), 'b', lw=linew)
        plot(phasevals_shifted, -1 * clip(diffgoalang_shifted, -inf, 0), 'b--', lw=linew)
        plot(phase_vals, popspikes_mean / np.nanmax(popspikes_mean), 'r', lw=linew)
        legend(('Pos. diff', 'Neg. diff', '# spikes'),fontsize=6)
        title('Abs. difference between \n current and start-end angle ')
        subplot(165)
        plot(fraction_proj_diff_xy_startend_mean, 'b')
        print("fraction_proj_diff_xy_startend_mean = ", fraction_proj_diff_xy_startend_mean)
        plot(popspikes_mean / np.nanmax(popspikes_mean), 'r')
        ax=subplot(166, projection='polar')
        imin_projstartgoal_mean = np.argmin(fraction_proj_diff_xy_startend_mean[1:])
        phasevals_shifted = np.append(phase_vals[imin_projstartgoal_mean: ], phase_vals[: imin_projstartgoal_mean] + 2*np.pi)
        fraction_proj_diff_xy_startend_shifted = np.append(fraction_proj_diff_xy_startend_mean[imin_projstartgoal_mean:], fraction_proj_diff_xy_startend_mean[:imin_projstartgoal_mean])
        # Cardioid plot
        #plot(phasevals_shifted, fraction_proj_diff_xy_startend_shifted, 'b', lw=linew)
        plot(phasevals_shifted, clip(fraction_proj_diff_xy_startend_shifted, 0, inf), 'b', lw=linew)
        plot(phasevals_shifted, -1 * clip(fraction_proj_diff_xy_startend_shifted, -inf, 0), 'b--', lw=linew)
        plot(phase_vals, popspikes_mean / np.nanmax(popspikes_mean), 'r', lw=linew)
        title('Fraction of total movement projected \n onto direction towards seq. end')
        legend(('Pos. fraction', 'Neg. fraction', '# spikes'),fontsize=6)
        



    #figure()
    #hist(reshape(np.nanmax(n_clust[:, :, firstind:], 2), n_networks * n_trials), 20)#, normed=True)
    #xlabel('No. of cluster centers')


    inet, jtr = np.nonzero( np.nanmax(n_clust[:, :, firstind:], 2)==1)
    print("Fraction of max. cluster centers = 1: ", len(inet) / float(n_networks * n_trials))

    #figure()
    #xlabel('Total seq. distance')
    #plot(totdist, np.nanmax(diff_xy[:, :, firstind:], 2), '.')
    #ylabel('Max. shift')



    old_plots = False # True # False # 

    if old_plots:

        #'''#
        # Compare "new" sequence endpoints with behavioral simulation data:
        N = 144
        nTrials = 40
        n_grid = 80 
        savefigs = False # True #  
        dpisize = 300 # 50 #         
        dpisize_save = 300     
        identifier = 'Square_maze_ngr80_N'+str(N)+'_rb_15cmps_newstart_Poi200-10_fanout2500-DGCA70nA_learnoldmult5nAp100ms_winit_02nA_tauTr0-1s_PFr25cmDetReal_sclw1_varH_locSrch_MF_clAw_smartR_5cm_negR05_300ms_DAdecr4_osc-tauinh_navcorr_'+str(nTrials) # on tractus and laptop
        #identifier = 'Square_maze_ngr80_N'+str(N)+'_rb_15cmps_newstart_Poi200-10_fanout2500-DGCA70nA_learnoldmult5nAp100ms_winit_02nA_tauTr0-1s_PFr25cmDetReal_sclw1_varH_locSrch_MF_clAw_smartR_5cm_negR05_300ms_DAdecr1st_osc-tauinh_navcorr_'+str(nTrials) # on globus

        dataman = DataManager(identifier)

        latencies, endpoints, weights, seqcount, seqstart, seqend, rand_nav_time, \
            goal_nav_time, focal_search_time, occupancyMap, weight_array_home, weight_array_away, center_mat_plot, center_mat_array, goal_index_array = zip(*dataman.values())

        valid_data = range(n_networks)

        print("valid_data =", valid_data)
        n_valid_nets = size(valid_data)
        print("Excluded %i network(s) from analysis" %(dataman.itemcount() - n_valid_nets))

        weight_array_home = reshape(weight_array_home, (dataman.itemcount(), nTrials, n_grid**2))
        weight_array_away = reshape(weight_array_away, (dataman.itemcount(), nTrials, n_grid**2))

        weight_array_home = reshape(weight_array_home[valid_data], (n_valid_nets, nTrials, n_grid**2))
        weight_array_away = reshape(weight_array_away[valid_data], (n_valid_nets, nTrials, n_grid**2))
        wsum = np.zeros([n_networks, n_trials])
        wmax = np.zeros([n_networks, n_trials])

        #endpoints = reshape(endpoints, (dataman.itemcount(), nTrials))

        goal_index_array = reshape(goal_index_array, (dataman.itemcount(), nTrials))

        #endpoints = reshape(endpoints[valid_data], (n_valid_nets, nTrials))
        endpoints = reshape(ind_end, (n_valid_nets, nTrials)) # NEW -- sequence data!

        goal_index_array = reshape(goal_index_array[valid_data], (n_valid_nets, nTrials))

        center_len = len(center_mat_plot[0][0,:])
        occ_len = len(occupancyMap[0][0][0,:])
        if len(occupancyMap[0][:]) == 20:
            occupancyMap = reshape(occupancyMap, (dataman.itemcount(), 20, occ_len, occ_len))
        else:
            occupancyMap = reshape(occupancyMap, (dataman.itemcount(), 40, occ_len, occ_len))        
        if len(occupancyMap[0][:]) == 20:
            occupancyMap = reshape(occupancyMap[valid_data], (n_valid_nets, 20, occ_len, occ_len))
        else:
            occupancyMap = reshape(occupancyMap[valid_data], (n_valid_nets, 40, occ_len, occ_len))  



        nz_even_Home = nonzero(mod(range(nTrials),2)-1)
        nz_odd_Away = nonzero(mod(range(nTrials),2))
        endpoints_Home = zeros([n_valid_nets, ceil(nTrials/2.0)])
        endpoints_Away = zeros([n_valid_nets, floor(nTrials/2.0)])
        goal_index_array_Home = zeros([n_valid_nets, ceil(nTrials/2.0)])
        goal_index_array_Away = zeros([n_valid_nets, floor(nTrials/2.0)])

        for i_netw in range(n_valid_nets):
            endpoints_Home[i_netw][:] = endpoints[i_netw][nz_even_Home]
            endpoints_Away[i_netw][:] = endpoints[i_netw][nz_odd_Away]
            goal_index_array_Home[i_netw][:] = goal_index_array[i_netw][nz_even_Home]
            goal_index_array_Away[i_netw][:] = goal_index_array[i_netw][nz_odd_Away]
        
    #'''


    n_netw_test = 12 # 12
    n_trials_test = 40 # 40
    dpisize = 300 # 150 #  

    i_netw, i_trial = 0, 2
    
 
    ioff()
    show()













