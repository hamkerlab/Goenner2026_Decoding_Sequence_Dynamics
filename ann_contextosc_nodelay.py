# Implementation of the network described in 
# Azizi et al., Front. Comp. Neurosci. 2013

# TODO: Test for membrane potential oscillations ?! (Cf. English et al. 2014)

#from random import *

import pickle
import matplotlib.cm as cm
import gc

from ANNarchy import *
import numpy as np
from pylab import *
from scipy.sparse import lil_matrix
from time import time
import numpy as np

setup(dt = 0.2)
#setup(dt = 1.0) # test 9.11.15


#--------------------------------------------------------------------------------------------------------
def xypos(mat_index, L):
    # Given that an array of L^2 neurons is arranged on a toroidal LxL grid,  returns x and y grid coordinates for a given index, values are in [0, L] 
    # Inverse (arrangement) formula: mat_index = x*L + y    
    mat_index = array(mat_index)    
    L = float(L)
    x = floor(mat_index / L)
    y = mat_index - L * floor(mat_index / L)    
    return x,y
#--------------------------------------------------------------------------------------------------------
def xypos_maze(mat_index, nGrid, L_maze):
    # Given that an array of nGrid^2 neurons is arranged on a square (nGrid x nGrid) grid, 
    # returns x and y grid coordinates corresponding to the centers of the "checkerboard fields"
    # Inverse (arrangement) formula: mat_index =  nGrid*(x/DeltaL - 0.5) + (y/DeltaL - 0.5)    
    mat_index = array(mat_index)
    nGrid = float(nGrid)
    DeltaL = L_maze / nGrid    
    x = (floor(mat_index / nGrid) + 0.5) * DeltaL   #floor(mat_index / L)
    y = (mat_index - nGrid * floor(mat_index / nGrid) + 0.5) * DeltaL    
    return x,y
#--------------------------------------------------------------------------------------------------------
def xy_index_maze(x, y, nGrid, L_maze):
    DeltaL = L_maze / float(nGrid)
    mat_index = floor(y/DeltaL) + nGrid * floor(x/DeltaL)    
    return mat_index
#--------------------------------------------------------------------------------------------------------
def plot_sequence(nGrid, nValueCells, M_sp_PlaceCells, placevalMat, dt):    
    # plot a matrix showing which neurons spiked:
    print("Creating sequence plot...")
    L = nGrid
    spCountMat = zeros([L, L])
    spTimeMat = nan*ones([L,L]) 
    placeValueMat = zeros([L, L])
    for iNeuron in range(nValueCells):
        x,y = xypos(iNeuron, L)        
        placeValueMat[x, y] = placevalMat[iNeuron]
        #times = M_sp_PlaceCells.spiketimes[iNeuron] / ms
        times = np.array(M_sp_PlaceCells[iNeuron]) * dt
        if len(times) > 0:
            spTimeMat[x, y]  = times[len(times) -1]            
    figure()        
    subplot(1,2,1)
    matshow(transpose(spTimeMat), origin='lower', cmap=cm.YlOrRd, fignum=False)

    colorbar()
    title('Spike times of place cells [ms]')
    xlabel('x position')
    ylabel('y position')    
    subplot(1,2,2)
    matshow(transpose(placeValueMat), origin='lower',fignum=False)
    colorbar()
    title('Place-value')
    xlabel('x position')
    ylabel('y position')
    return
#--------------------------------------------------------------------------------------------------------
def Exp_xyValue(mat_index, L, scale_noise, goal_index):
    # Given that an array of L^2 neurons is arranged on a plain LxL grid, 
    # returns "place-value" for a given MATRIX index
    # EXPONENTIALLY increasing in x and y directions    
    x,y = xypos(mat_index, L)
    xGoal, yGoal = xypos(goal_index, L)
    L = float(L)    

    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / 20 ) + scale_noise * rand(size(mat_index), size(mat_index)) # test for smaller DG place fields: Works with (large) fan-out of 4000, scaling x1
    #val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / (20*90/63.0 ) ) + scale_noise * rand(size(mat_index), size(mat_index)) # test for n_grid = 90
    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / (20*80/63.0 ) ) + scale_noise * rand(size(mat_index), size(mat_index)) # test for n_grid = 80

    val /= (1 + scale_noise)    
    return val
#-------------------------------------------------------------------------------------------------------
def Exp_xyValue_normalized(mat_index, L, scale_noise, goal_index):
    # Given that an array of L^2 neurons is arranged on a toroidal LxL grid, 
    # returns "place-value" for a given MATRIX index
    # EXPONENTIALLY increasing in x and y directions    

    x,y = xypos(mat_index, L)
    xGoal, yGoal = xypos(goal_index, L)
    L = float(L)    
    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / (20*80/63.0 ) ) + scale_noise * rand(size(mat_index)) # n_grid=70
    val /= (1 + scale_noise)

    refsum = 1.0 * 3913.66 # For n_grid=80, max. sum of values (for goal_index=3199). 3913.66 too strong?! - bump at the start is larger, starts to shrink?! # 0.8*3913.66 still quite strong - 0.6 * 3913.66 OK?!
    valsum = sum(val)

    val *= refsum / valsum
    
    return val
#--------------------------------------------------------------------------------------------------------
def weight_mod(factor, mat_index, nmax, L):
 
    x,y = xypos(mat_index, L)
    L = float(L)
    N=20
    ci = randint(0, nmax, N)
    Z_mod = zeros(len(mat_index))
    for k in range(N):  
        xc, yc = xypos(ci[k], L)
        Z_mod += exp(- ((x-xc)**2 + (y-yc)**2) / 10.0**2 ) 
    Z_mod -= Z_mod.mean(); Z_mod /= Z_mod.max() # zero mean, max. 1
 
    wmod = ones(len(mat_index)) + factor * Z_mod # mean = 1, max = 1 + factor
  
    return wmod 

#-------------------------------------------------------------------------------------------------------------

i_sigma_dist_cm, i_wmax_exc_exc, i_wmax_exc_inh,\
i_wmax_inh_inh, i_wmax_inh_exc, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

#-------------------------------------------------------------------------------------------------------------

t_init = time()

#n_grid = 63 # 54 # 36 # TEST # 72 # 70 # 35 # 45 #         fixed arrangement on a grid
n_grid = 80 # 90 # 72 # test!!!
print("n_grid = ", n_grid)
n_exc = int(n_grid**2)
#n_inh = int(0.2*n_exc) # Used with n_grid = 35
n_inh = int(0.2*36**2) # works also with n_grid=70 - 1:20 ratio
#n_inh = 100 # int(0.2*20**2) # TEST 24.9.14- works if inhibitory connections are appropriately scaled, but no speed benefit, doesn't look very nice

#C = 300*pF # 281*pF #  300 pF used for setup of "fan-out" connectivity!
scale_length = 2

#netw_params = 	[sigma_dist_cm, wmax_exc_exc, wmax_exc_inh, wmax_inh_inh, wmax_inh_exc,         pconn_recur,    tau_membr, tau_exc, tau_inh,   dt]
#netw_params = 	[30.0, 	        2*1.6e-9*amp, 3.9e-11*amp, 1.3e-10*amp, 3.1e-11*amp * 1.0,      1, 10*ms, 3*ms,    2*ms,  0.2*ms] # testing for n_grid=63! Works good for "ideal" place-value, also with STP!


#netw_params = 	[50.0, 	        1.04*72/63.0*0.3*nA, 0.8125*pA, 13*pA, 4.9*pA,  1, 10*ms, 12*ms,    2*ms,  0.2*ms] # not bad! n_grid=72 # Brian 1
#netw_params = 	[50.0, 	        1.04*72/63.0*0.3*1e3, 0.8125, 13.0, 4.9,  1, 10, 12,    2,  0.2] # not bad! n_grid=72 (ANNarchy)
#netw_params = 	[50.0, 	        0.4114e3,             0.8125, 13.0, 4.9,  1, 10, 12,    2,  0.2] # test for n_grid=80: bump stops?
netw_params = 	[50.0, 	        0.4114e3,             0.8125, 13.0, 4.1,  1, 10, 12,    2,  0.2] # test for n_grid=80: modified for ANNarchy

'''#
netw_params = 	[50.0, 	        0.4114e3,             0.8125, 2*13.0, 2*4.1,  1, 10, 12,    2,  0.2] # test: stronger w_ii for stronger oscillations? works
netw_params = 	[50.0, 	        0.4114e3,             0.8125, 4*13.0, 4*4.1,  1, 10, 12,    2,  0.2] # test: stronger w_ii for stronger oscillations? works
netw_params = 	[50.0, 	        0.4114e3,             0.8125, 8*13.0, 8*4.1,  1, 10, 12,    2,  0.2] # test: stronger w_ii for stronger oscillations? works
netw_params = 	[50.0, 	        0.4114e3,             8*0.8125, 8*13.0, 4.1,  1, 10, 12,    2,  0.2] # test: stronger w_ii for stronger oscillations? Cool!!!
netw_params = 	[50.0, 	        0.4114e3,             4*0.8125, 4*13.0, 4.1,  1, 10, 12,    2,  0.2] # test: stronger w_ii for stronger oscillations? cool! Both taus=12ms
'''
#netw_params = 	[50.0, 	        0.4114e3,             4*0.8125, 2*4*13.0, 2*4.1,  1, 10, 12,    2,  0.2] # test: tau_inh = 6ms, good!
netw_params = 	[50.0, 	        0.4114e3,             6*0.8125, 2*6*13.0, 2*4.1,  1, 10, 12,    2,  0.2] # test: tau_inh = 6ms: good, slightly fading for random input
netw_params = 	[50.0, 	        2*0.4114e3,           12*0.8125, 6*6*13.0, 5*4.1,  1, 10, 12,    2,  0.2] # test: tau_inh =2ms, tau-exc=6ms - cool! 

#netw_params = 	[50.0, 	        0.8228e3,             9.75,      468,      20.5,  1, 10, 12,    2,  0.2]
netw_params = 	[50.0, 	        0.8228e3,             9.75,      468,      0.9*20.5,  1, 10, 12,    2,  0.2]

#netw_params = 	[50.0, 	        2*0.4114e3,           2*4*0.8125, 4*13.0, 4.1,  1, 10, 12,    2,  0.2] # test: tau_exc = 6ms
#netw_params = 	[50.0, 	        2*0.4114e3,           2*4*0.8125, 2*4*13.0, 2*4.1,  1, 10, 12,    2,  0.2] # test: tau_exc = 6ms, tau_inh = 6ms

#netw_params = 	[30.0, 	        0.4114e3,      0.8125, 13.0, 4.1,  1, 10, 12,    2,  0.2] # test for n_grid=63


#netw_params = 	[50.0, 	        0.5e3,               0.8125, 13.0, 4.1,  1, 10, 12,    2,  0.2] # test for n_grid=80: modified for ANNarchy; works, large bump
#netw_params = 	[50.0, 	        0.49e3,               0.8125, 13.0, 4.1,  1, 10, 12,    2,  0.2] # test for n_grid=80: modified for ANNarchy; works; reaches the goal (up to ca.10-16cm)
# large bump - stops at (115 cm, 115 cm) for L_maze_cm = 444 when stimlated at (0,0)! (Approx. (21,21) in cell coordinates)

#netw_params = 	[50.0, 	        1.15*90/63.0*0.3*1e3, 0.8125, 13.0, 5.1,  1, 10, 12,    2,  0.2] # test for n_grid=90
#netw_params = 	[50.0, 	        0.493*1e3, 	      0.8125, 13.0, 5.1,  1, 10, 12,    2,  0.2] # same, n_grid=90
#netw_params = 	[50.0, 	        0.493*1e3, 	      0.8125, 13.0, 4.1,  1, 10, 12,    2,  0.2] # modified for ANNarchy (n_grid=90), goal missed by 16 cm!!!

#netw_params = 	[50.0, 	        0.57*1e3, 	      0.8125, 13.0, 5.1,  1, 10, 12,    2,  0.2] # test: vary w_ee
#netw_params = 	[50.0, 	        0.493*1e3, 	      1.5, 20.0, 2.5,  1, 10, 12,    2,  0.2] # test: vary w_ei; slightly increase I_ext; inh. cells fire early & synchronously!
#netw_params = 	[50.0, 	        0.493*1e3, 	      2.0, 5.0, 1.0,  1, 10, 12,    2,  0.2] # test: vary w_ei; slightly increase I_ext

# Note: almost 8GB memory consumption without weight cutoff! With weight cutoff: Generating the weights takes like forever!

# max w_ee = 3.9324, sum(w_ee) = 257784468.324; sum= 65534982 * max (original: sum = 12067000 * max); 1206700/65534982 = 0.1841; 1/... = 5.43	
# S_DG_Exc_sum = 6.561e-5 (max= 1e-12); sum = 6.561e7*max

# If I get lower rates only with a larger bump - should I increase the network size? Idea: w_ee_max determines the max. rate, w_ee_max TIMES bump size TIMES rate must produce an I_exc > 1.5nA
# Next test: Adjust inhibition in the case of shorter initial stimulation (80-100ms) ?
# Remark: Recurrent excitation increases very rapidly - inhibition follows much later? --> make inhibition respond earlier (Problem: If inhibition starts early, it is already near its max.)

t_refr_exc = 3
t_refr_inh = 4

print("netw_params= ", netw_params)

# Constants

tau_membr = netw_params[i_tau_membr]
tau_exc = netw_params[i_tau_exc] 
tau_inh = netw_params[i_tau_inh]

# L_maze_cm = 350 * (n_grid / 63.0)
#maze_edge_cm = 75 * (n_grid / 63.0) # Only the area inside [maze_edge_cm, L_maze_cm - maze_edge_cm]^2 can be visited
#L_maze_cm = 430 # for n_grid=80, sigma=50
maze_edge_cm = 110 # 150 # 115 # for n_grid=80, sigma=50
L_maze_cm = 200 + 2 * maze_edge_cm

print("L_maze_cm = ", L_maze_cm)
print("maze_edge_cm = ", maze_edge_cm)
scale_dist = (350.0 / float(n_grid)) / (300.0 / 72.0)

sigma_dist_cm = netw_params[i_sigma_dist_cm] # factor scale_length is applied in the weight formula
sigma_dist_new_cm = netw_params[i_sigma_dist_cm] # 30 # scale_length * sigma_dist_cm

eqs_if_scaled_new = Neuron(
	parameters = '''gL_nS = 30 : population
        		C_pF = 300 : population
        		EL_mV = -70.6 : population
		        sigma_mV = 2.0* 31.62 : population
			tau_exc_ms = 6.0 : population
			tau_inh_ms = 2.0 : population
			I_ext_pA = 0.0
			t_refr = 3.0 : population''', # 
	equations = ''' noise = Normal(0.0, 1.0) 
        		C_pF/gL_nS * du_mV/dt = - (u_mV - EL_mV) + 1/gL_nS * (g_exc - g_inh + I_ext_pA) + sigma_mV * noise : init = -70.6
			dg_exc/dt = - g_exc / tau_exc_ms : init = 0.0
			dg_inh/dt = - g_inh / tau_inh_ms : init = 0.0
                        ext_input = I_ext_pA''', 
	name = 'eqs_if_scaled_new', spike = 'u_mV > (EL_mV + 20)', reset = 'u_mV = EL_mV', refractory = 't_refr') # *0.2  #### multiply sigma_mV with 0.2 for dt=1 ms ?!

eqs_if_scaled_new_DG = Neuron(
	parameters = '''gL_nS = 30 : population
        		C_pF = 300 : population
        		EL_mV = -70.6 : population
			tau_exc_ms = 6.0 : population ''',
	equations = ''' C_pF/gL_nS * du_mV/dt = - (u_mV - EL_mV) + 1/gL_nS * (g_exc - g_inh)  : init = -70.6
			dg_exc/dt = - g_exc / tau_exc_ms : init = 0.0
			dg_inh/dt = - g_inh / tau_exc_ms : init = 0.0 ''', 
	name = 'eqs_if_scaled_new', spike = 'u_mV > (EL_mV + 20)', reset = 'u_mV = EL_mV') 

eqs_if_trace = Neuron(
	parameters = '''EL_mV = -70.6 : population
        		tau_membr = 10.0 : population''',
        equations = ''' u_sp = g_copy_mV : init = 0.0  
			dg_copy_mV / dt = -g_copy_mv / 1000.0''', 
	name = 'eqs_if_trace', 	
    spike = 'u_sp > 0', 
    reset = '''g_copy_mV = 0.0''') 



Exc_neurons = Population(geometry=(n_grid, n_grid), neuron = eqs_if_scaled_new, name = "Exc_neurons")
Inh_neurons = Population(n_inh, neuron = eqs_if_scaled_new, name = "Inh_neurons")
Inh_neurons.t_refr = t_refr_inh
#DG_neurons = Population(n_exc, neuron = eqs_if_scaled_new_DG, name = "DG_neurons")
DG_neurons = Population(geometry=(n_grid, n_grid), neuron = eqs_if_scaled_new_DG, name = "DG_neurons")
rate_factor_test = 1 #  Test 6.8.2015
#poisson_inp_hi = PoissonPopulation(geometry=(n_grid, n_grid), rates='1.5 * 100.0 * (1.0 + 1.0*sin(2*pi*25*t/1000.0))' ) # 25 Hz 
#poisson_inp_lo = PoissonPopulation(geometry=(n_grid, n_grid), rates='5.0 * (1.0 + 1.0*sin(2*pi*25*t/1000.0))' ) 
poisson_inp_hi = PoissonPopulation(geometry=(n_grid, n_grid), rates='1.5 * 100.0 * (1.0 + 1.0*sin(2*pi*25*t/1000.0 - 0.5*pi))' ) # 25 Hz 
poisson_inp_lo = PoissonPopulation(geometry=(n_grid, n_grid), rates='5.0 * (1.0 + 1.0*sin(2*pi*25*t/1000.0 - 0.5*pi))' ) 

context_pop_home = Population(geometry=(n_grid, n_grid), neuron = eqs_if_trace, name = "context_pop_home")
context_pop_home.u_mV = context_pop_home.EL_mV

placevalMat = zeros(n_exc)
start_bias = zeros(n_exc)
start_index = xy_index_maze(maze_edge_cm,  maze_edge_cm, n_grid, L_maze_cm)
#start_index = xy_index_maze(100+maze_edge_cm,  maze_edge_cm, n_grid, L_maze_cm)
#start_index = xy_index_maze(0.5*L_maze_cm,  0.5*L_maze_cm, n_grid, L_maze_cm)
#start_index = xy_index_maze(L_maze_cm - maze_edge_cm - 50, L_maze_cm - maze_edge_cm - 50, n_grid, L_maze_cm)
#start_index = xy_index_maze(L_maze_cm - maze_edge_cm,  L_maze_cm - maze_edge_cm, n_grid, L_maze_cm)
print("start_index = ", start_index)
print("start location (x,y) = ", xypos_maze(start_index, n_grid, L_maze_cm))

goal_index = xy_index_maze(200+maze_edge_cm, maze_edge_cm, n_grid, L_maze_cm) # center?
#goal_index = xy_index_maze(0, 0, n_grid, L_maze_cm) # test for bump radius
print("goal_index = ", goal_index)

nonfiring_index = xy_index_maze(75, 275, n_grid, L_maze_cm) # 

xr,yr = xypos(goal_index, n_grid)
print("reward location (x,y) = ", xypos_maze(goal_index, n_grid, L_maze_cm)) # L_maze_cm / float(n_grid)*xr, L_maze_cm / float(n_grid)*yr

#start_bias[range(n_exc)] = Exp_xyValue(range(n_exc), n_grid, 0, start_index) # original version
#placevalMat[range(n_exc)] = Exp_xyValue(range(n_exc), n_grid, 0, goal_index) # * 1.0e-9 
placevalMat[range(n_exc)] = np.random.rand(n_exc) * 0.3 # * 0.4 # defined in nA - caution, tau_exc is now 4x larger than previously!!! # 0.4e-9

filename = 'wdata_seqmodel'


#'''#
file_wdata_VC_PC = open('data/'+filename, 'rb') 
Wdata_VC_PC=pickle.load(file_wdata_VC_PC, encoding='latin1') 
file_wdata_VC_PC.close()
print("file = ", filename)
placevalMat = Wdata_VC_PC * 1e9 #* 0.5 # convert to nA - wdata from file are in ampere, max. ca. 1e-9 - caution, tau_exc is now 4x larger than previously!!!
#placevalMat = sqrt(1 - (1-Wdata_VC_PC * 1e9)**2) #* 0.5 # test 27.1.
#placevalMat = sqrt(Wdata_VC_PC * 1e9) # test 11.7.16
placevalMat = (Wdata_VC_PC * 1e9)**0.635 # test 11.7.16
#'''
print("wmax (placevalMat) = ", np.max(placevalMat))
#print "sum (placevalMat) = ", np.sum(placevalMat)

simple_syn =Synapse(pre_spike=""" g_target += w """) 
copy_syn =Synapse(pre_spike=""" g_target += w """)

#max_synapse_exc = 1.0*(n_grid/45.0) * 1/(sqrt(2*pi) * sigma_dist_cm)
#min_synapse_exc = 1.0*(n_grid/45.0) / (sqrt(2*pi) * sigma_dist_cm) * exp(-6.0/2.0) # 35.0/45.0 *
max_synapse_exc = netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * sigma_dist_cm)
min_synapse_exc = 0.0 ## 1.0e-6 * netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * sigma_dist_cm)
sigma_dist_sheet_units =  sigma_dist_cm * scale_dist / L_maze_cm ### / (L_maze_cm / n_grid) * 0.1841 * 1e-3 # test ## Distances between neurons are normalized to the [0,1] range by ANNarchy!!!
("max_synapse_exc = ", max_synapse_exc)
print("sigma_dist_sheet_units = ", sigma_dist_sheet_units)

S_Exc_Exc = Projection(
	pre=Exc_neurons, post=Exc_neurons, target='exc', synapse=simple_syn
).connect_gaussian( amp = max_synapse_exc, sigma = sigma_dist_sheet_units, limit = min_synapse_exc, allow_self_connections = False, delays = 0.1) # 2.5

S_Exc_Inh = Projection(
    pre=Exc_neurons, post=Inh_neurons, target='exc', synapse=simple_syn
).connect_all_to_all( weights = netw_params[i_wmax_exc_inh], delays = 0.1) # 2.5

S_Inh_Inh = Projection(
    pre=Inh_neurons, post=Inh_neurons, target='inh', synapse=simple_syn
).connect_all_to_all( weights = netw_params[i_wmax_inh_inh], allow_self_connections = False, delays = 0.1) # 2.5

S_Inh_Exc = Projection(
    pre=Inh_neurons, 
    post=Exc_neurons, 
    target='inh',
    synapse=simple_syn
).connect_all_to_all( weights = netw_params[i_wmax_inh_exc], delays = 0.1) # 2.5

S_Poi_lo_cont_home = Projection(
	pre = poisson_inp_lo, post = context_pop_home, target = 'copy_mV', synapse = copy_syn
).connect_one_to_one( weights = 21.0 ) # orig. value 21.0 # simple_syn

S_Poi_hi_cont_home = Projection(
	pre = poisson_inp_hi, post = context_pop_home, target = 'copy_mV', synapse = copy_syn
).connect_one_to_one( weights = 0.0 ) # orig. value 21.0 # simple_syn


'''#
S_context_home_DG = Projection(
	pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = simple_syn
).connect_from_matrix( weights =  np.zeros([n_exc, n_exc]) ) # works here
'''

#S_context_home_DG = Projection(
#	pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = simple_syn
#).connect_from_matrix( weights = np.diag(placevalMat) ) # works, weights will be changed below!!!

#lil_w = lil_matrix((n_exc, n_exc))
lil_w = lil_matrix( np.diag(1/rate_factor_test * placevalMat * 1e3) )
#for i in range(n_exc):
#    lil_w[i, i] = 1/rate_factor_test * placevalMat[i]
S_context_home_DG = Projection(
	pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = simple_syn
).connect_from_sparse( weights = lil_w ) # remove weight assignment below!



fan_out = 5000.0 
fan_out = 10000.0
fan_out = 2500.0 
sigmafan_sheet_units = sqrt(0.5 * fan_out) / L_maze_cm # / n_grid)
#max_synapse_dg_exc = 7.3 * (10000.0 / fan_out) # 0.73 # = 7.3e-13*ampere # 1e-12 # max_synapse_exc
max_synapse_dg_exc = 0.9 * 7.3 * (10000.0 / fan_out) 
min_synapse_dg_exc = 0.1 # 0.05 # 0.26855 # 

print("max_synapse_dg_exc = ", max_synapse_dg_exc)

S_DG_Exc = Projection(
	pre = DG_neurons, post = Exc_neurons, target = 'exc', synapse = simple_syn
).connect_gaussian( amp = max_synapse_dg_exc, sigma = sigmafan_sheet_units, limit = min_synapse_dg_exc, delays = 0.1) # 2.0

print('After S_DG_Exc')

start_bias = zeros(n_exc)
#start_index = n_grid * (n_grid - 1)
 # Test 15.9.14 # CAUTION 6.12.2016: Not normalized across start positions!!!
start_bias[range(n_exc)] = Exp_xyValue_normalized(range(n_exc), n_grid, 0, start_index) # normalized - slower movement?!
("max. start_bias = ", max(start_bias))

#Exc_neurons.I_ext = start_bias* 4.5e-10*amp # 1-1.3 too low, 1.5-2 seems sufficient
Exc_neurons.I_ext_pA = start_bias* 0.9e3 
#Exc_neurons.I_ext_pA = 1000.0 # Test: Difference between "old" and "clean" script for uniform I_ext_pA=200 and normal (unmodified) recurrent synapses! No difference for inactive rec. synapses

#print "sum(start_bias) = ", np.sum(start_bias)
#print "sum(Exc_neurons.I_ext_pA) = ", np.sum(Exc_neurons.I_ext_pA)

#Exc_neurons.g_exc = start_bias* 0.9e3 * 3.25 # test: Does this work? 

stim_time_ms = 35 
init_time_ms = 50 
run_time_ms = 350 

global total_run_time_ms
total_run_time_ms = init_time_ms + run_time_ms

'''#
global temp_rate
temp_rate = zeros([n_exc, int(rint(total_run_time_ms * ms / rate_clock.dt))+1])
global max_rate
max_rate = zeros(int(rint(total_run_time_ms * ms / rate_clock.dt))+1)
global rateTimeMat
rateTimeMat = nan*ones([2*n_grid, 2*n_grid]) 
'''

print("Initiating the network...")
print("Poisson to DG synapses already active!")
#S_context_home_DG.w[:,:] = 1/rate_factor_test * placevalMat # * amp
##



# Don't forget to set the same weights below!!!

print("DG to CA3 synapses already active!")

# ------------------------------------------------------------------------------------

ion()

cont_popview = context_pop_home[0:50]

compile()

#'''#
print("Modifying synaptic weights...")
for dend in S_Exc_Exc.dendrites:
    dend.w *= (1.2*np.random.rand(dend.size)+0.4) 
print("Done")
#'''



# initiation period
Exc_neurons.u_mV = Exc_neurons.EL_mV # + rand(n_exc)*5
Inh_neurons.u_mV = Inh_neurons.EL_mV # + rand(n_exc)*5
DG_neurons.u_mV = DG_neurons.EL_mV



# Start recording
M_Exc = Monitor(Exc_neurons, ['spike', 'u_mV', 'g_exc', 'g_inh', 'ext_input']) # , 'I_ext_pA'
M_Inh = Monitor(Inh_neurons, ['spike', 'u_mV', 'g_exc', 'g_inh'])
M_DG = Monitor(DG_neurons, ['spike']) 
M_contpop = Monitor(cont_popview, ['spike']) #

center_mat = -np.inf*ones([100, 100]) # Test 1.10.14: Not bad
center_len = len(center_mat[0,:])
n_edge = ceil(maze_edge_cm / float(L_maze_cm) * center_len)


simulate(stim_time_ms, measure_time=True) # 35 ms

print("Continuing init without I_ext")

# end of initiation period
Exc_neurons.I_ext_pA = 0

simulate(init_time_ms - stim_time_ms, measure_time=True)

print("Switching to place-value noise...")

print("S_Poi_lo_cont_home.w (before) = ", S_Poi_lo_cont_home.w)
print( "S_Poi_hi_cont_home.w (before) = ", S_Poi_hi_cont_home.w)

S_Poi_lo_cont_home.w = 0.0
S_Poi_hi_cont_home.w = 21.0
print("S_Poi_lo_cont_home.w (now) = ", S_Poi_lo_cont_home.w)
print("S_Poi_hi_cont_home.w (now) = ", S_Poi_hi_cont_home.w)


t_start = time()
t_sim = run_time_ms # 200
t_curr = 0
for step in range(10):
    simulate(t_sim * 0.1) # , measure_time=True)
    t_estim = int( (9-step) * (time()-t_start) / (step+1) )
    print("%i %% done, approx. %i seconds remaining" %( int((step+1)*10),  t_estim))
print("Duration: %i seconds" %( int(time() - t_start) ))


ion()

dpisize= 300 #        

fsize = 5 # 7 # 
col_blue2 = '#0072B2' #
col_green = '#2B9F78'
col5 = '#E69F00' # 'Orange'
col_purple = '#CC79A7'
col3 = '#D55E00' # 'Tomato'

t_end = init_time_ms + run_time_ms # 50 # 625

figure(figsize=(5,4), dpi=dpisize) 

subplot(711)
contpop_spikes = M_contpop.get(['spike'])

print("Mean Context population rate = ", np.mean( np.mean(M_contpop.population_rate(contpop_spikes, smooth=100.0)) ))
te, ne = M_contpop.raster_plot(contpop_spikes)
tmin_cont, tmax_cont = te.min(), te.max()
plot(te, ne, 'k.', markersize=1.0)

xlabel('')
ylabel('context', fontsize=fsize)
ax=gca()
#ax.set_xticks([0, t_end])
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
ax.set_yticks([0, 50])
ax.set_yticklabels([0, 50])
axis([0, 400, axis()[2], axis()[3]])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6


#subplot(412)
#subplot(712)
ax_wrapper=[]
ax_wrapper.append(subplot(712))
dg_spikes = M_DG.get(['spike'])
te, ne = M_DG.raster_plot(dg_spikes)
plot(te, ne, 'k.', markersize=1.0)
xlabel('')
ylabel('DG', fontsize=fsize)
ax=gca()
#ax.set_xticks([0, t_end])
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
ax.set_yticks([0, 6400])
ax.set_yticklabels([0, 6400])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6

#subplot(511)
#subplot(413)
subplot(713)
exc_spikes = M_Exc.get(['spike'])
te, ne = M_Exc.raster_plot(exc_spikes)

plot(te, ne, 'k.', markersize=0.1) # 0.25 # 1.0
xlabel('')
ylabel('CA3 exc.', fontsize=fsize)
ax=gca()
#ax.set_xticks([0, t_end])
#ax.set_xticklabels([0, t_end])
#    ax.set_xticks([])
#ax.set_xticks([0, t_end])
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()

ax.set_yticks([0, 6400])
ax.set_yticklabels([0, 6400])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6

#subplot(512)
#subplot(414)
subplot(714)
inh_spikes = M_Inh.get(['spike'])
ti, ni = M_Inh.raster_plot(inh_spikes)
plot(ti, ni, 'k.', markersize=1.0)
#xlabel('t [ms]', fontsize=fsize) # 6
ylabel('CA3 inh.', fontsize=fsize) # 
ax=gca()
#ax.set_xticks([0, t_end])
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
ax.set_yticks([0, n_inh])
ax.set_yticklabels([0, n_inh])

for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) # 6

#tight_layout()
savefig('ann_raster_contextosc_nodelay_new.png', dpi=300)

print("Done")





#rates = M_Exc.smoothed_rate(exc_spikes, smooth=20.0)
rates = M_Exc.smoothed_rate(exc_spikes, smooth=5.0)
#rates = M_Exc.smoothed_rate(exc_spikes) # unsmoothed

x_centers, y_centers = xypos_maze(range(n_exc), n_grid, L_maze_cm) ## Is this correct?
bump_center_x_cm = np.inf * np.ones(rates.shape[1])
bump_center_y_cm = np.inf * np.ones(rates.shape[1])
center_mat = np.inf * np.ones([100, 100])
scale = len(center_mat[0,:]) / float(L_maze_cm)
for t_ind in range(rates.shape[1]): 
    if np.sum(rates[:, t_ind]) > 0:
        bump_center_x_cm[t_ind] = np.average(x_centers, weights = rates[:, t_ind]) #  rates (unsmoothed) contains the reciprocal inter-spike-interval - alternative?
        bump_center_y_cm[t_ind] = np.average(y_centers, weights = rates[:, t_ind])
        center_mat[int(scale * bump_center_x_cm[t_ind]), int(scale * bump_center_y_cm[t_ind])] = dt() * t_ind

print("x_init_rate, y_init_rate = %i, %i" %( int(bump_center_x_cm[50*5]), int(bump_center_y_cm[50*5]) ))
print("x_end_rate, y_end_rate = %i, %i" %( int(bump_center_x_cm[-1]), int(bump_center_y_cm[-1]) ))

diff_xy_cm = sqrt( (diff(bump_center_x_cm))**2 + (diff(bump_center_y_cm))**2 )
diff_x_cm = diff(bump_center_x_cm)
diff_y_cm = diff(bump_center_y_cm)

#figure()
it_start = 50*5
#it_start = 50

#subplot(311)
subplot(717)
#plot(diff_xy_cm[it_start : ])
#ylabel('PV Step size [cm / (0.2 ms)]  ', fontsize=fsize)

#plot(append(nan * ones(it_start), diff_xy_cm[it_start : ]) ) # units: cm/(0.2ms)
plot(append(nan * ones(it_start), diff_xy_cm[it_start : ] / 0.02), color=col_blue2)#, lw=0.5) # units: m/sec
plot(append(nan * ones(it_start), diff_x_cm[it_start : ] / 0.02), '--', color=col3) # col5) #, lw=0.5) # units: m/sec
legend(('2D step size','x-step'), fontsize=fsize)
ylabel('Step size \n [m/sec]  ', fontsize=fsize)
ax=gca()
poprate_dg = M_DG.population_rate(dg_spikes, smooth=5.0) # 10.0
poprate_exc = M_DG.population_rate(exc_spikes, smooth=5.0) # 10.0
poprate_context = M_contpop.population_rate(contpop_spikes, smooth=5.0) # 10.0

print("poprate_dg.shape = ", poprate_dg.shape)
print("len(poprate_dg) = ", len(poprate_dg))

ax.set_xticks(5 * arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()


axis([axis()[0], axis()[1], 0, 15]) # units: m/sec
ax.set_yticks([axis()[2], axis()[3]])

for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
xlabel('t [ms]', fontsize=fsize)


ax_wrapper=[]
ax_wrapper.append(subplot(715))

DG_col = 'k' # '0.4'
plot(poprate_dg[ : ], color=DG_col)
ax_twin=gca()
ax_twin.xaxis.grid()
ax_twin.set_xticks(5 * arange(0, 440, 40))
labels = append(repeat("", len(arange(0, 400, 40))), '400')
labels[0] = 0
ax_twin.set_yticks([axis()[2], axis()[3]])
ax_twin.set_xticklabels([])

for label in ax_twin.get_xticklabels() + ax_twin.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
    label.set_color(DG_col)
ylabel('DG \n pop. rate \n [sp/sec]', fontsize=fsize, color=DG_col)

ax_wrapper.append(ax_wrapper[0].twinx())

plot(poprate_exc[ : ], color=col_green)
ylabel('CA3 exc.\n pop. rate \n [sp/sec]', fontsize=fsize, color=col_green)
ax=gca()
ax.set_yticks([axis()[2], axis()[3]])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
    label.set_color(col_green)





subplot(716)
plot(bump_center_x_cm - maze_edge_cm, '0.4')
plot(bump_center_y_cm - maze_edge_cm, color=col_purple)
ylabel('Decoded \n location \n [cm]', fontsize=fsize)
legend(('x', 'y'), fontsize=fsize)
ax=gca()
ax.set_xticks(5 * arange(0, 440, 40))
labels = append(repeat("", len(arange(0, 400, 40))), '400')
labels[0] = 0
ax.set_xticklabels([])
ax.xaxis.grid()
axis([axis()[0], axis()[1], 0, axis()[3]])
ax.set_yticks([axis()[2], axis()[3]])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6


subplots_adjust(bottom=0.08, top=0.95, hspace=0.5)


savefig('ann_raster_contextosc_nodelay_stepsizes_mod_combined.png', dpi=300)


print("Total time = ", time()-t_init)

print("tmin_cont, tmax_cont = ", tmin_cont, tmax_cont)




ioff()
show()

















