from pylab import *
import numpy as np
import scipy.stats as st
import new_colormaps as nc
#from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
from scipy.special import i0, i1
from scipy.optimize import curve_fit, leastsq

from sim_params_seq import params_id, n_grid_cm, DeltaT_window_sec, overlap, n_windows, n_reps, n_osc_cycles, sigma_code_cm, x_step_cm, y_step_cm, max_rate_Hz, n_mincells

def gaussian_pdf(x, mu, sigma):
    return 1.0/(sqrt(2*pi) * sigma) * exp(- (x - mu)**2 / sigma**2)

def sample_mean_dist(x, mu, n):
    return gaussian_pdf(x, mu, sigma / sqrt(n))

def rayleigh_pdf(x, sigma):
    return x / sigma**2 * exp(- x**2 /(2*sigma**2) )

def mean_stepsize_dist(x, sigma, n_equiv):
    return rayleigh_pdf(x, sigma / sqrt(n_equiv) )

def rayleigh_quantile(p, sigma):
    return sigma * sqrt(-2 * log(1 - p))

def rayleigh_mean(sigma):
    return sigma * sqrt(0.5 * pi)


def rayleigh_cdf(x, sigma):
    return 1 - exp(- x**2 / (2*sigma**2))


def Laguerre12(t):                                                            
    return exp(0.5*t) * ( (1-t)* i0(-0.5*t) - t * i1(-0.5*t) )


def rice_pdf(v, sigma, x):
    return x / sigma**2 * exp( -(x**2 + v**2) / (2*sigma**2) ) * i0(x*v / sigma**2)

def rice_mean(v, sigma):
    return sigma * sqrt(pi/2) * Laguerre12(-v**2 / (2*sigma**2) )

def rice_variance(v, sigma):
    return 2*sigma**2 + v**2 - 0.5*pi*sigma**2 * (Laguerre12(-v**2 / (2*sigma**2)))**2

def mean_stepsize_dist_rice(x, v, sigma, n_equiv):
    return rice_pdf(v, sigma / sqrt(n_equiv), x)

def func_pow_fit(params, x):
    return params[0] * x**params[1] + sqrt(2) * 2.5
    #return params[0] * x**(-0.5) + sqrt(2) * 2.5
    #return params[0] * exp(-x*params[1]) + sqrt(2) * 2.5

def residuals(params, y, x):
    return y - func_pow_fit(params, x)


# See color codes at https://cdn.elifesciences.org/author-guide/tables-colour.pdf

tab_blue   = '#90CAF9'
tab_green  = '#C5E1A5'
tab_orange = '#FFB74D'
tab_yellow = '#FFF176'
tab_purple = '#9E86C9'
tab_red    = '#E57373'
tab_pink   = '#F48FB1'
tab_grey   = '#E6E6E6'

col1 = '#56B4E9' # 'DodgerBlue' # 'b'
col3 = '#D55E00' # 'Tomato' # 'PaleVioletRed' # 'r'
col4 = '0.3' # '#0072B2' # 'SteelBlue' # 'c'
col5 = '#E69F00' # 'Orange' #'m'
col6 = '#CC79A7' # Purple

col_blue = '#56B4E9' # 'DodgerBlue' # 'b'
col_green = '#2B9F78' # 'SeaGreen' # 'g'
col_vermillion = '#D55E00' # 'Tomato' # 'PaleVioletRed' # 'r'
col_blue2 = '#0072B2' # 'SteelBlue' # 'c'
col_orange = '#E69F00' # 'Orange' #'m'
col_purple = '#CC79A7' 

col_x = '0.6' # '0.4' # 'k'
col_y = col_purple # col_green
col_biased = col_blue2 # col_purple
col_unbiased = col_vermillion # col_purple # col_blue # col_orange



ion()

n_cells = 200

# Now imported from param file:
# ----------------------------
# n_grid_cm = 100 
# DeltaT_window_sec = 0.02
# n_mincells = 10.0
# sigma_code_cm = 15.0
# n_reps = 100 # 1000 # 100
# n_windows = 32
# x_step_cm = 2.5
# y_step_cm = 2.5





figure(figsize=(4.5, 4), dpi=300)

fsize = 6 # 3 # 6
fsize_small = 4 # 2 # 4
linew = 1.5 # 0.75 # 1.5
msize_big = 5 # 3 # 30 # 8
msize = 2

x_start_cm = 10 # 25
y_start_cm = 10 # 25
co_true_xy_step_cm = sqrt(x_step_cm**2 + y_step_cm**2)
x_grid, y_grid = meshgrid(range(n_grid_cm), range(n_grid_cm))
frac_mincells = n_mincells / 200.0

mod_depth = 1.0

osc_phase = arange(0, n_windows) / float(n_windows) * 4*pi
osc_modulation = 0.5*(1 + frac_mincells + (1 - frac_mincells) * cos(osc_phase))


#file_co = open('data/sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','r')
file_co = open('data/sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','rb')
co_x_true_array_cm = pickle.load(file_co, encoding='latin1')
co_y_true_array_cm = pickle.load(file_co, encoding='latin1')
co_x_mean_array_cm = pickle.load(file_co, encoding='latin1')
co_y_mean_array_cm = pickle.load(file_co, encoding='latin1')
co_tot_spikes = pickle.load(file_co, encoding='latin1')
co_var_names = pickle.load(file_co, encoding='latin1')
file_co.close()

#'''#
#file_spmod = open('data/speedmod_'+str(mod_depth)+'_sim_sequence_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','r')
file_spmod = open('data/speedmod_'+str(mod_depth)+'_sim_seq_paramsid_'+str(params_id)+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.txt','rb')
sm_x_true_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_y_true_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_x_mean_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_y_mean_array_cm = pickle.load(file_spmod, encoding='latin1')
sm_tot_spikes = pickle.load(file_spmod, encoding='latin1')
sm_var_names = pickle.load(file_spmod, encoding='latin1')
file_spmod.close()
#'''

repdim, window_dim = (0, 1)
n_plots_x = 4 # 4
n_plots_y = 2


sm_true_xdiff_mean = mean( diff(sm_x_true_array_cm, window_dim), repdim)
sm_true_ydiff_mean = mean( diff(sm_y_true_array_cm, window_dim), repdim)
sm_L2_diff_perframe = sqrt( diff(sm_x_mean_array_cm, window_dim)**2 + diff(sm_y_mean_array_cm, window_dim)**2)
sm_mean_L2diff_acrossreps = mean(sm_L2_diff_perframe, repdim)
sm_std_L2diff_acrossreps = std(sm_L2_diff_perframe, repdim)
sm_estim_x_diff_mean = mean( diff(sm_x_mean_array_cm, window_dim), repdim) # This is the way it SHOULD be done, not the way Pfeiffer & Foster did it
sm_estim_y_diff_mean = mean( diff(sm_y_mean_array_cm, window_dim), repdim)
sm_estim_x_diff_std = std( diff(sm_x_mean_array_cm, window_dim), repdim)
sm_estim_y_diff_std = std( diff(sm_y_mean_array_cm, window_dim), repdim)
sm_L2_of_meandiff = sqrt( sm_estim_x_diff_mean**2 + sm_estim_y_diff_mean**2)

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




subplot(3,3,1)
datalen = n_windows
box2 = 1.0 * append(ones(2), zeros(datalen-2)) # summation of spike count across windows
from scipy.signal import convolve
sm_tot_spikes_plot = nan * ones([n_reps, n_windows - 1])
sm_tot_spikes_effective = nan * ones([n_reps, n_windows - 1])
for i_reps in range(n_reps):
    sm_tot_spikes_plot[i_reps, :] = convolve(sm_tot_spikes[i_reps, :], box2)[1:datalen]
    for i_window in range(n_windows - 1):
        sm_tot_spikes_effective[i_reps, i_window] = 1.0/(1.0/sm_tot_spikes[i_reps, i_window] + 1.0/sm_tot_spikes[i_reps, i_window+1]) 
        # If var(X)=sigma^2/nx and var(Y) = sigma^2/ny, then var(X+Y) = sigma^2/nx + sigma^2/ny = sigma^2 * (1/nx + 1/ny); 1/n_eff := 1/nx + 1/ny
sm_tot_spikes_plot =  sm_tot_spikes_effective # TEST!!!
slope_lin, intercept_lin, r_lin, p_lin, std_err = st.linregress(reshape(sm_tot_spikes_plot, np.prod(sm_tot_spikes_plot.shape)), reshape(sm_L2_diff_perframe, np.prod(diff(sm_x_mean_array_cm, window_dim).shape)) )
xvals_lin = arange(0, 50)
plot(xvals_lin, intercept_lin + xvals_lin * slope_lin, 'k-', lw=0.25*linew, alpha=0) # 'r--'
ax_vlen_sm=gca()
ax_vlen_sm.set_xticklabels([])
for label in ax_vlen_sm.get_xticklabels() + ax_vlen_sm.get_yticklabels(): 
    label.set_fontsize(fsize) 
t = 1.0/sqrt(sm_tot_spikes_plot) # No improvement in R^2 value when fitting 1/sqrt(x-xmin)!
slope_sqrt, intercept_sqrt, r_sqrt, p_sqrt, std_err = st.linregress(reshape(t, np.prod(t.shape)), reshape(sm_L2_diff_perframe , np.prod(diff(sm_x_mean_array_cm, window_dim).shape)))
xvals_sqrt = arange(0, 70)
plot(xvals_sqrt, intercept_sqrt + 1.0/sqrt(xvals_sqrt) * slope_sqrt, 'k.', lw=0.25*linew, markersize=0.5*msize) # 'k-.'
if intercept_sqrt > 0:
    legend(('$r=$'+str(round(r_lin,2))+', $R^2=$'+str(round(r_lin**2,2))+', $p=$'+'%1.1e'%p_lin, \
            str(round(slope_sqrt,1))+'$/ \sqrt{n}+$'+str(round(intercept_sqrt,1))+', $R^2=$'+str(round(r_sqrt**2,2))+', $p=$'+'%1.1e'%p_sqrt), fontsize=0.6*fsize, frameon=False, loc='upper left')
else:
    legend(('$r=$'+str(round(r_lin,2))+', $R^2=$'+str(round(r_lin**2,2))+', $p=$'+'%1.1e'%p_lin, \
            '$1/ \sqrt{n}$ fit, $R^2=$'+str(round(r_sqrt**2,2))+', $p=$'+'%1.1e'%p_sqrt), fontsize=0.6*fsize, frameon=False, loc='upper left', numpoints=3)
plot(sm_tot_spikes_plot, sm_L2_diff_perframe, '.', color=col_blue2, markersize=msize)
plot(xvals_lin, intercept_lin + xvals_lin * slope_lin, 'k-', lw=0.25*linew, alpha=0) # 'r--'
plot(xvals_sqrt, intercept_sqrt + 1.0/sqrt(xvals_sqrt) * slope_sqrt, 'k.', lw=0.25*linew, markersize=0.5*msize) # 'k-.'
ylabel('Vector length of step \n [cm]', fontsize=fsize)


fig=gcf()
fig.text(0.035, 0.97,'Phase-locked movement', fontsize=1.25*fsize, weight='bold')
#fig.text(0.38, 0.97,'Smooth movement', fontsize=1.25*fsize, weight='bold')
fig.text(0.4, 0.97,'Smooth movement', fontsize=1.25*fsize, weight='bold')


subplot(3,3,2)
co_tot_spikes_plot = nan * ones([n_reps, n_windows - 1])
co_tot_spikes_min = nan * ones([n_reps, n_windows - 1])
co_tot_spikes_effective = nan * ones([n_reps, n_windows - 1])
for i_reps in range(n_reps):
    co_tot_spikes_plot[i_reps, :] = convolve(co_tot_spikes[i_reps, :], box2)[1:datalen]
    for i_window in range(n_windows - 1):
        co_tot_spikes_min[i_reps, i_window] = min(co_tot_spikes[i_reps, i_window], co_tot_spikes[i_reps, i_window+1])
        co_tot_spikes_effective[i_reps, i_window] = 1.0/(1.0 / co_tot_spikes[i_reps, i_window] + 1.0 / co_tot_spikes[i_reps, i_window+1])
co_tot_spikes_plot = co_tot_spikes_effective # TEST!!!
ax_vlen_co=gca()
ax_vlen_co.set_xticklabels([])
for label in ax_vlen_co.get_xticklabels() + ax_vlen_co.get_yticklabels(): 
    label.set_fontsize(fsize) 
slope_lin, intercept_lin, r_lin, p_lin, std_err = st.linregress(reshape(co_tot_spikes_plot, np.prod(co_tot_spikes_plot.shape)), reshape(co_L2_diff_perframe , np.prod(diff(co_x_mean_array_cm, window_dim).shape)) )
print ("co: Linear fit of x_step: R^2, p_lin = ", r_lin**2, p_lin)
#rho, p = st.pearsonr(reshape(co_tot_spikes_plot, np.prod(co_tot_spikes_plot.shape)), reshape(co_L2_diff_perframe , np.prod(diff(co_x_mean_array_cm, window_dim).shape)))
xvals = arange(0, 50)
#text(2, 30, '$r=$'+str(round(r_lin,2))+', ', fontsize=0.6*fsize, color='r')
#text(27.5, 30, '$R^2=$'+str(round(r_lin**2,2))+', $p=$'+'%1.1e'%p_lin, fontsize=0.6*fsize, color='r')

pow_fit = 0.5 # 0.88 # 0.5 # 0.75 # 0.5 # 
t = 1.0/co_tot_spikes_plot**pow_fit
slope_sqrt, intercept_sqrt, r_sqrt, p_sqrt, std_err = st.linregress(reshape(t, np.prod(t.shape)), reshape(co_L2_diff_perframe , np.prod(diff(co_x_mean_array_cm, window_dim).shape)))
print ("co: 1/sqrt fit of x_step: R^2, p_sqrt = ", r_sqrt**2, p_sqrt)
print ("co: slope_sqrt, intercept_sqrt = ", slope_sqrt, intercept_sqrt)
xvals_sqrt = arange(0, 45)
plot(xvals_lin, intercept_lin + xvals_lin * slope_lin, 'k-', lw=0.25*linew, alpha=0) # 'r--' # Transparent line to generate legend
plot(xvals_sqrt, intercept_sqrt + 1.0/xvals_sqrt**pow_fit * slope_sqrt, 'k.', markersize=0.5*msize) # lw=0.25*linew)


plot([0, 1.5*co_tot_spikes_effective.max()], ones(2) * co_true_xy_step_cm, '--', lw=0.75*linew, color=col3) # 'k--', lw=0.5*linew)#, markersize=0.1*msize) # 'k:'
plot(xvals_sqrt, rice_mean(co_true_xy_step_cm, sigma_code_cm/sqrt(xvals_sqrt)), color=col5) #'0.6') # 'r'
plot(co_tot_spikes_plot, co_L2_diff_perframe, '.', color=col_blue2, markersize=msize)

plot(xvals_lin, intercept_lin + xvals_lin * slope_lin, 'k-', lw=0.25*linew, alpha=0) # 'r--'
plot(xvals_sqrt, intercept_sqrt + 1.0/xvals_sqrt**pow_fit * slope_sqrt, 'k.', markersize=0.5*msize) # lw=0.25*linew)
#plot(xvals_sqrt, intercept_sqrt + rice_mean(co_true_xy_step_cm, sigma_code_cm / sqrt(xvals_sqrt)) * slope_sqrt, 'k.', markersize=0.5*msize)
plot([0, 1.5*co_tot_spikes_effective.max()], ones(2) * co_true_xy_step_cm, '--', lw=0.75*linew, color=col3) # 'k--', lw=0.5*linew)#, markersize=0.1*msize) # 'k:'
plot(xvals_sqrt, rice_mean(co_true_xy_step_cm, sigma_code_cm/sqrt(xvals_sqrt)), color=col5) #'0.6')#, lw=0.5*linew) # 'r'



if intercept_sqrt > 0:
    legend(('$r=$'+str(round(r_lin,2))+', $R^2=$'+str(round(r_lin**2,2))+', $p=$'+'%1.1e'%p_lin, \
            '$1/ \sqrt{n}$ fit, $R^2=$'+str(round(r_sqrt**2,2))+', $p=$'+'%1.1e'%p_sqrt, \
            'True step size', 'Predicted mean of \nstep size estimates'), fontsize=0.6*fsize, frameon=False, loc='upper left', numpoints=3)
else:
    legend(('$r=$'+str(round(r_lin,2))+', $R^2=$'+str(round(r_lin**2,2))+', $p=$'+'%1.1e'%p_lin, \
            '$1/ \sqrt{n}$ fit, $R^2=$'+str(round(r_sqrt**2,2))+', $p=$'+'%1.1e'%p_sqrt, \
            'True step size', 'Predicted mean of \nstep size estimates'), fontsize=0.6*fsize, frameon=False, loc='upper left', numpoints=3)
#plot(xvals, sigma_code_cm * sqrt(pi/2.0) * 1.0/sqrt(xvals), 'w--', lw=linew) # Theoretical prediction from rayleigh distribution: E[step] = sigma * sqrt(pi/n), where 1/n = 1/n1 + 1/n2
sigma = 15.0
# Quantile der Rayleigh-Verteilung: Q(F, sigma) = sigma * sqrt(-2 * ln(1 - F))
#n=5; plot(n*ones(2), [0, 2*sigma/ sqrt(n)], col1)
# Erwartungswert der Rayleigh-Verteilung: sigma * sqrt(pi/2)
'''#
F1=0.05; F2=0.95
n=5; plot(n*ones(2), [rayleigh_quantile(F1, sigma/sqrt(n)), rayleigh_quantile(F2, sigma/sqrt(n))], color=col1, lw=0.5*linew)
plot(n, rayleigh_mean(sigma/sqrt(n)), '_', color=col1, markersize=2*msize)
n=10; plot(n*ones(2), [sigma/sqrt(n) * sqrt(-2 * log(1 - F1)), sigma/sqrt(n) * sqrt(-2 * log(1 - F2))], col_green, lw=0.5*linew)
plot(n, rayleigh_mean(sigma/sqrt(n)), '_', color=col_green, markersize=2*msize)
n=20; plot(n*ones(2), [sigma/sqrt(n) * sqrt(-2 * log(1 - F1)), sigma/sqrt(n) * sqrt(-2 * log(1 - F2))], col3, lw=0.5*linew)
plot(n, rayleigh_mean(sigma/sqrt(n)), '_', color=col3, markersize=2*msize)
n=30; plot(n*ones(2), [sigma/sqrt(n) * sqrt(-2 * log(1 - F1)), sigma/sqrt(n) * sqrt(-2 * log(1 - F2))], col4, lw=0.5*linew)
plot(n, rayleigh_mean(sigma/sqrt(n)), '_', color=col4, markersize=2*msize)
n=40; plot(n*ones(2), [sigma/sqrt(n) * sqrt(-2 * log(1 - F1)), sigma/sqrt(n) * sqrt(-2 * log(1 - F2))], col5, lw=0.5*linew)
plot(n, rayleigh_mean(sigma/sqrt(n)), '_', color=col5, markersize=2*msize)
'''
v = co_true_xy_step_cm
'''#
n=2; plot(n, rice_mean(v, sigma/sqrt(n)), '_', color=col5, markersize=2*msize)
#plot(n*ones(2), [rice_mean(v, sigma/sqrt(n)) - rice_variance(v, sigma/sqrt(n)), rice_mean(v, sigma/sqrt(n)) + rice_variance(v, sigma/sqrt(n))], color=col5, lw=0.5*linew)
#n=3; plot(n, rice_mean(v, sigma/sqrt(n)), '_', color='k', markersize=2*msize)
n=5; plot(n, rice_mean(v, sigma/sqrt(n)), '_', color=col1, markersize=2*msize)
n=10; plot(n, rice_mean(v, sigma/sqrt(n)), '_', color=col_green, markersize=2*msize)
n=20; plot(n, rice_mean(v, sigma/sqrt(n)), '_', color=col3, markersize=2*msize)
n=30; plot(n, rice_mean(v, sigma/sqrt(n)), '_', color=col4, markersize=2*msize)
#n=40; plot(n, rice_mean(v, sigma/sqrt(n)), '_', color=col5, markersize=2*msize)
'''


subplot(3,3,3)
t=arange(0, 2*sigma, 0.01)
'''#
plot(t, mean_stepsize_dist(t, sigma, 5), col1)
plot(t, mean_stepsize_dist(t, sigma, 10), col_green)
plot(t, mean_stepsize_dist(t, sigma, 20), col3)
plot(t, mean_stepsize_dist(t, sigma, 30), col4)
plot(t, mean_stepsize_dist(t, sigma, 40), col5)
'''

#plot(t-v, mean_stepsize_dist_rice(t, v, sigma, 2), col5)
plot(t-v, mean_stepsize_dist_rice(t, v, sigma, 5), col1)
plot(t-v, mean_stepsize_dist_rice(t, v, sigma, 10), col_green)
plot(t-v, mean_stepsize_dist_rice(t, v, sigma, 20), col3)
plot(t-v, mean_stepsize_dist_rice(t, v, sigma, 30), col4)
plot(t-v, mean_stepsize_dist_rice(t, v, sigma, 40), col5)

#plot(v*ones(2), [0, 0.25], 'k--')
#plot((v-v)*ones(2), [0, 0.25], 'k--')
m5 = rice_mean(v, sigma/sqrt(5))
#plot(m5*ones(2), [0, rice_pdf(v, sigma/sqrt(5), m5)], ':', color=col1)
#plot((m5-v)*ones(2), [0, rice_pdf(v, sigma/sqrt(5), m5)], ':', color=col1)

#m5=rayleigh_mean(sigma / sqrt(5))
#plot(m5 * ones(2), [0, mean_stepsize_dist(m5, sigma, 5)], '-', color=col1, lw=0.25*linew)

legend(('$n=5$', '$n=10$', '$n=20$', '$n=30$', '$n=40$'), fontsize=0.5*fsize, frameon=False)
#legend(('$n=2$', '$n=5$', '$n=10$', '$n=20$', '$n=30$'), fontsize=0.5*fsize, frameon=False)
ax2=gca()
for label in ax2.get_xticklabels() + ax2.get_yticklabels(): 
    label.set_fontsize(fsize)
#title(' Theoretical distribution \n of step size decoding error', fontsize=0.8*fsize) # for constant movement #  (Rayleigh)
ylabel('Probability \n density', fontsize=fsize) # function
ax2.yaxis.set_label_position('right')
xlabel('Step size decoding error [cm]', fontsize=0.8*fsize, va='bottom')


subplot(3,3,6)
t=arange(-sigma, sigma, 0.01)
plot(t, sample_mean_dist(t, 0, 5), col1)
plot(t, sample_mean_dist(t, 0, 10), col_green)
plot(t, sample_mean_dist(t, 0, 20), col3)
plot(t, sample_mean_dist(t, 0, 30), col4)
plot(t, sample_mean_dist(t, 0, 40), col5)
legend(('$n=5$', '$n=10$', '$n=20$', '$n=30$', '$n=40$'), fontsize=0.5*fsize, frameon=False)
ax=gca()
#ax.yaxis.tick_right()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize)
#title(' Theoretical distribution \n of decoded $x$-steps', fontsize=0.8*fsize) # for constant movement #  (Gaussian)
ylabel('Probability \n density', fontsize=fsize) # function
xlabel('$x$-step decoding error [cm]', fontsize=0.8*fsize, va='bottom')
ax.yaxis.set_label_position('right')
ymax12 = max(ax.axis()[3], ax2.axis()[3])

offset = 0.0 # 0.01 # 0.2
#ax2.set_xticks([0, sigma, 2*sigma])
ax2.axis([-15, 15, 0, 0.2])
#ax2.set_xticks([-sigma, 0, sigma])
ax2.set_xticks([-sigma, -sqrt(2)*x_step_cm, 0, sigma]) # 
ax2.set_xticklabels([-15, '$-v$', 0, 15])


#ax2.set_xticklabels(['$\mu_{xy}$', '$\mu_{xy} + \sigma$', '$\mu_{xy} + 2\sigma$'])
#ax2.set_xticklabels(['$\mu_{xy}$', '$\mu_{xy} + \sigma$', '$\mu_{xy} + 2\sigma$'])
#ax2.set_yticks(arange(0, ymax12 + offset, 0.2))
ax2.set_yticks([0, ymax12 + offset])
#ax2.set_yticklabels([0, '', '', '', '', '', '', ymax12])
ax2.set_yticklabels([0, ymax12])

ax.set_xticks([-sigma, 0, sigma])
#ax.set_xticklabels(['$\mu_x - \sigma$', '$\mu_x$', '$\mu_x + \sigma$'])
#ax.set_yticks(arange(0, ymax12 + offset, 0.2))
#ax.set_yticks([0, ymax12 + offset])
#ax.set_yticklabels([0, '', '', '', '', '', '', ymax12])
#ax.set_yticklabels([0, ymax12])
ax.set_yticks([0, max(0.2, ax.axis()[3]) ])


subplot(3, 3, 4)
plot(sm_tot_spikes_plot, diff(sm_x_mean_array_cm, window_dim), '.', markersize=msize, color=col_x)
rho, p = st.pearsonr(reshape(sm_tot_spikes_plot, np.prod(sm_tot_spikes_plot.shape)), reshape(diff(sm_x_mean_array_cm, window_dim) , np.prod(diff(sm_x_mean_array_cm, window_dim).shape)))
text(5, 20, '$r=$'+str(round(rho,2))+', $p=$'+'%1.1e'%p, fontsize=0.6*fsize, color='k') # 'r'
slope, intercept, r_value, p_value, std_err = st.linregress(reshape(sm_tot_spikes_plot, np.prod(sm_tot_spikes_plot.shape)), reshape(diff(sm_x_mean_array_cm, window_dim), np.prod(diff(sm_x_mean_array_cm, window_dim).shape)) )
xvals = arange(0, 70)
#plot(xvals, intercept + xvals * slope, '-', lw=0.25*linew, color='k')
ax_xstep_sm=gca()
#plot([xvals[0], xvals[-1]], ones(2) * mean(diff(sm_x_mean_array_cm, window_dim)), 'g-.')
ax_xstep_sm.set_xticklabels([])
for label in ax_xstep_sm.get_xticklabels() + ax_xstep_sm.get_yticklabels(): 
    label.set_fontsize(fsize) 
ylabel('$x$-step \n [cm]', fontsize=fsize)
sm_sp_P25, sm_sp_P75 = percentile(sm_tot_spikes_plot, [25, 75])
nz_Qlower_i, nz_Qlower_j = nonzero(sm_tot_spikes_plot <= sm_sp_P25)
nz_Qupper_i, nz_Qupper_j = nonzero(sm_tot_spikes_plot >= sm_sp_P75)
n_min = min(nz_Qlower_i.shape[0], nz_Qupper_i.shape[0])
#(sm_x_mean_array_cm, window_dim)[nz_Qupper_i, nz_Qupper_j])
rsp_diff_Qlower = reshape(diff(sm_x_mean_array_cm, window_dim)[nz_Qlower_i, nz_Qlower_j], nz_Qlower_i.shape)
rsp_diff_Qupper = reshape(diff(sm_x_mean_array_cm, window_dim)[nz_Qupper_i, nz_Qupper_j], nz_Qupper_i.shape)
sm_res = st.wilcoxon(rsp_diff_Qlower[:n_min], rsp_diff_Qupper[:n_min])



subplot(3, 3, 5)
plot([xvals[0], xvals[-1]], ones(2) * x_step_cm, 'k--', lw=0.5*linew, alpha=0)
plot([xvals[0], xvals[-1]], ones(2) * x_step_cm, 'k--', lw=0.5*linew)
rho, p = st.pearsonr(reshape(co_tot_spikes_plot, np.prod(co_tot_spikes_plot.shape)), reshape(diff(co_x_mean_array_cm, window_dim) , np.prod(diff(co_x_mean_array_cm, window_dim).shape)))
#text(5, 20, '$r=$'+str(round(rho,2))+', $p=$'+'%1.1e'%p, fontsize=0.6*fsize, color='k') # 'r'
legend(('$r=$'+str(round(rho,2))+', $p=$'+'%1.1e'%p, 'True $x$-step'), fontsize=0.6*fsize, frameon=False, loc='upper left')
plot(co_tot_spikes_plot, diff(co_x_mean_array_cm, window_dim), '.', markersize=msize, color=col_x)
slope, intercept, r_value, p_value, std_err = st.linregress(reshape(co_tot_spikes_plot, np.prod(co_tot_spikes_plot.shape)), reshape(diff(co_x_mean_array_cm, window_dim), np.prod(diff(co_x_mean_array_cm, window_dim).shape)) )
xvals = arange(0, 60)
#plot(xvals, intercept + xvals * slope, '-', lw=0.25*linew, color='k') #'r'
ax_xstep_co=gca()
plot([xvals[0], xvals[-1]], ones(2) * x_step_cm, 'k--', lw=0.5*linew)
#plot([xvals[0], xvals[-1]], ones(2) * mean(diff(co_x_mean_array_cm, window_dim)), 'g-.')
ax_xstep_co.set_xticklabels([])
for label in ax_xstep_co.get_xticklabels() + ax_xstep_co.get_yticklabels(): 
    label.set_fontsize(fsize) 
co_sp_P25, co_sp_P75 = percentile(co_tot_spikes_plot, [25, 75])
nz_Qlower_i, nz_Qlower_j = nonzero(co_tot_spikes_plot <= co_sp_P25)
nz_Qupper_i, nz_Qupper_j = nonzero(co_tot_spikes_plot > co_sp_P75)
n_min = min(nz_Qlower_i.shape[0], nz_Qupper_i.shape[0])
#(co_x_mean_array_cm, window_dim)[nz_Qupper_i, nz_Qupper_j])
rsp_diff_Qlower = reshape(diff(co_x_mean_array_cm, window_dim)[nz_Qlower_i, nz_Qlower_j], nz_Qlower_i.shape)
rsp_diff_Qupper = reshape(diff(co_x_mean_array_cm, window_dim)[nz_Qupper_i, nz_Qupper_j], nz_Qupper_i.shape)
co_res = st.wilcoxon(rsp_diff_Qlower[:n_min], rsp_diff_Qupper[:n_min])


subplot(3, 3, 7)
rho, p = st.pearsonr(reshape(sm_tot_spikes_plot, np.prod(sm_tot_spikes_plot.shape)), reshape(diff(sm_y_mean_array_cm, window_dim) , np.prod(diff(sm_y_mean_array_cm, window_dim).shape)))
plot(sm_tot_spikes_plot, diff(sm_y_mean_array_cm, window_dim), '.', markersize=msize, color=col_y)
text(5, 20, '$r=$'+str(round(rho,2))+', $p=$'+'%1.1e'%p, fontsize=0.6*fsize, color='k') #'r'
#legend(('$r=$'+str(round(rho,2))+', $p=$'+'%1.1e'%p, 'True $x$-step'), fontsize=0.6*fsize, frameon=False, loc='upper left')
slope, intercept, r_value, p_value, std_err = st.linregress(reshape(sm_tot_spikes_plot, np.prod(sm_tot_spikes_plot.shape)), reshape(diff(sm_y_mean_array_cm, window_dim), np.prod(diff(sm_y_mean_array_cm, window_dim).shape)) )
xvals = arange(0, 70)
#plot(xvals, intercept + xvals * slope, '--', lw=linew, color='k') # 'r'
ax_ystep_sm=gca()
for label in ax_ystep_sm.get_xticklabels() + ax_ystep_sm.get_yticklabels(): 
    label.set_fontsize(fsize) 
#xlabel('Equivalent sample size', fontsize=fsize)
xlabel('$N_{step}$', fontsize=fsize, va='bottom') # \n (Harmonic mean spike count $\\times$ 0.5)
ylabel('$y$-step \n [cm]', fontsize=fsize)


subplot(3, 3, 8)
plot([xvals[0], xvals[-1]], ones(2) * y_step_cm, 'k--', lw=0.5*linew, alpha=0)
plot([xvals[0], xvals[-1]], ones(2) * y_step_cm, 'k--', lw=0.5*linew)
rho, p = st.pearsonr(reshape(co_tot_spikes_plot, np.prod(co_tot_spikes_plot.shape)), reshape(diff(co_y_mean_array_cm, window_dim) , np.prod(diff(co_y_mean_array_cm, window_dim).shape)))
legend(('$r=$'+str(round(rho,2))+', $p=$'+'%1.1e'%p, 'True $y$-step'), fontsize=0.6*fsize, frameon=False, loc='upper left')
plot(co_tot_spikes_plot, diff(co_y_mean_array_cm, window_dim), '.', markersize=msize, color=col_y)
plot([xvals[0], xvals[-1]], ones(2) * y_step_cm, 'k--', lw=0.5*linew)
#text(5, 20, '$r=$'+str(round(rho,2))+', $p=$'+'%1.1e'%p, fontsize=0.6*fsize, color='k') # 'r'
slope, intercept, r_value, p_value, std_err = st.linregress(reshape(co_tot_spikes_plot, np.prod(co_tot_spikes_plot.shape)), reshape(diff(co_y_mean_array_cm, window_dim), np.prod(diff(co_y_mean_array_cm, window_dim).shape)) )
xvals = arange(0, 60)
#plot(xvals, intercept + xvals * slope, '--', lw=linew, color='k') # 'r'
ax_ystep_co=gca()
for label in ax_ystep_co.get_xticklabels() + ax_ystep_co.get_yticklabels(): 
    label.set_fontsize(fsize) 
#xlabel('Equivalent sample size', fontsize=fsize)
xlabel('$N_{step}$', fontsize=fsize, va='bottom') #  \n (Harmonic mean spike count $\\times$ 0.5)



fig=gcf()
x_shift = 0.0
fig.text(0.08, 0.92, 'a', weight='bold', fontsize=1.5*fsize)
fig.text(0.36, 0.92, 'b', weight='bold', fontsize=1.5*fsize)
fig.text(0.65, 0.92, 'c', weight='bold', fontsize=1.5*fsize)
fig.text(0.08, 0.61, 'd', weight='bold', fontsize=1.5*fsize)
fig.text(0.36, 0.61, 'e', weight='bold', fontsize=1.5*fsize)
fig.text(0.65, 0.61, 'f', weight='bold', fontsize=1.5*fsize)
fig.text(0.08, 0.31, 'g', weight='bold', fontsize=1.5*fsize)
fig.text(0.36, 0.31, 'h', weight='bold', fontsize=1.5*fsize)





xmax_ax345678 = max(ax_vlen_sm.axis()[1], ax_vlen_co.axis()[1], ax_xstep_co.axis()[1], ax_xstep_sm.axis()[1], ax_ystep_co.axis()[1], ax_ystep_sm.axis()[1])

#ymax_ax34 = int(2*sigma_code_cm) # max(ax_vlen_sm.axis()[3], ax_vlen_co.axis()[3])
#ymax_ax5678 =  int(2*sigma_code_cm) # max(ax_xstep_co.axis()[3], ax_xstep_sm.axis()[3], ax_ystep_co.axis()[3], ax_ystep_sm.axis()[3])
ymax_ax34 = min(50, max(int(2*sigma_code_cm), ax_vlen_sm.axis()[3], ax_vlen_co.axis()[3]))
ymax_ax5678 =  max(int(2*sigma_code_cm), ax_xstep_co.axis()[3], ax_xstep_sm.axis()[3], ax_ystep_co.axis()[3], ax_ystep_sm.axis()[3])


ax_vlen_sm.axis( [ax_vlen_sm.axis()[0], xmax_ax345678, 0.0, ymax_ax34])
ax_vlen_co.axis( [ax_vlen_co.axis()[0], xmax_ax345678, 0.0, ymax_ax34])
ax_xstep_sm.axis( [ax_xstep_sm.axis()[0], xmax_ax345678, ax_xstep_sm.axis()[2], ymax_ax5678])
ax_xstep_co.axis( [ax_xstep_co.axis()[0], xmax_ax345678, -ymax_ax5678, ymax_ax5678]) # ax_xstep_co.axis()[2]
ax_ystep_sm.axis( [ax_ystep_sm.axis()[0], xmax_ax345678, ax_ystep_sm.axis()[2], ymax_ax5678])
ax_ystep_co.axis( [ax_ystep_co.axis()[0], xmax_ax345678, ax_ystep_co.axis()[2], ymax_ax5678])

#ax_vlen_sm.set_yticks([0, sigma_code_cm, 2*sigma_code_cm, ymax_ax34])
#ax_vlen_sm.set_yticklabels([0, '$\sigma$', '2$\sigma$', ymax_ax34])
ax_vlen_sm.set_yticks([0, ymax_ax34])
ax_vlen_sm.set_yticklabels([0, ymax_ax34])
ax_vlen_co.set_yticks([0, ymax_ax34])
ax_vlen_co.set_yticklabels([])
#ax_vlen_co.set_yticks([0, round(co_true_xy_step_cm, 1), int(ymax_ax34)])
#ax_vlen_co.yaxis.grid() 

#ax_xstep_sm.set_yticks([-ymax_ax5678, 0, sigma_code_cm, ymax_ax5678])
#ax_xstep_sm.set_yticklabels([-ymax_ax5678, 0, '$\sigma$', ymax_ax5678])
ax_xstep_sm.set_yticks([-ymax_ax5678, 0, ymax_ax5678])
#ax_xstep_sm.yaxis.grid()
ax_xstep_co.set_yticks([-ymax_ax5678, 0, ymax_ax5678])
ax_xstep_co.set_yticklabels([])
#ax_xstep_co.yaxis.grid() 
ax_ystep_sm.set_yticks([-ymax_ax5678, 0, ymax_ax5678])
#ax_ystep_sm.yaxis.grid()
ax_ystep_co.set_yticks([-ymax_ax5678, 0, ymax_ax5678])
ax_ystep_co.set_yticklabels([])
#ax_ystep_co.yaxis.grid() 



#tight_layout(pad=0.01, h_pad = 0.5, w_pad = 0.05)
#subplots_adjust(bottom=0.08, top=0.9, wspace=0.25, hspace=0.75, left=0.12)#, right=0.85 ) # 
subplots_adjust(bottom=0.06, top=0.9, wspace=0.4, hspace=0.35, left=0.11, right=0.94 ) # 

#savefig('figs/comparison3_ricedist_sigma_code_'+str(int(sigma_code_cm))+'_'+str(n_reps)+'reps_'+str(int(n_mincells))+'mincells_'+str(n_windows)+'steps.png', dpi=300)
savefig('figs/comparison3_ricedist_paramsid_'+str(params_id)+'.png', dpi=300)


ioff()
show()
