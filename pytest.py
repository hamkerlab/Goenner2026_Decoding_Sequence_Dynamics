import numpy as np

#def read_pkl():
#    co_x_true_array_cm, co_y_true_array_cm, co_x_mean_array_cm, co_y_mean_array_cm, co_tot_spikes, co_var_names \
#            = np.load('H:/AFS-Home/Einreichungen/Sequence dynamics/plots/data/altspeed_1.0_sim_seq_paramsid_4_1000reps_1mincells_79steps_npformat.npy', allow_pickle=True, encoding="bytes")
#    return co_x_true_array_cm, co_y_true_array_cm, co_x_mean_array_cm, co_y_mean_array_cm, co_tot_spikes, co_var_names
	
def read_pkl(filename):
    co_x_true_array_cm, co_y_true_array_cm, co_x_mean_array_cm, co_y_mean_array_cm, co_tot_spikes, co_var_names \
            = np.load(filename, allow_pickle=True, encoding="bytes")
    return co_x_true_array_cm, co_y_true_array_cm, co_x_mean_array_cm, co_y_mean_array_cm, co_tot_spikes, co_var_names	