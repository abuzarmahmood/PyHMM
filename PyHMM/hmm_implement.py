# Load spiking data
# Define model, Fit HMM
# 

# Import stuff# Impor 
import numpy as np
import DiscreteHMM
import pylab as plt
import matplotlib
import tables
import os
import shutil



#  _                     _             _ _    _                   _       _        
# | |                   | |           (_) |  (_)                 | |     | |       
# | |     ___   __ _  __| |  ___ _ __  _| | ___ _ __   __ _    __| | __ _| |_ __ _ 
# | |    / _ \ / _` |/ _` | / __| '_ \| | |/ / | '_ \ / _` |  / _` |/ _` | __/ _` |
# | |___| (_) | (_| | (_| | \__ \ |_) | |   <| | | | | (_| | | (_| | (_| | || (_| |
# |______\___/ \__,_|\__,_| |___/ .__/|_|_|\_\_|_| |_|\__, |  \__,_|\__,_|\__\__,_|
#                               | |                    __/ |                       
#                               |_|                   |___/
#

# Read blech.dir, and cd to that directory
f = open('blech.dir', 'r')
dir_name = []
for line in f.readlines():
	dir_name.append(line)
f.close()
os.chdir(dir_name[0][:-1])

# Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
units_file = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files
	if files[-10:] == 'hmm_params':
		params_file = files
	if files[-9:] == 'hmm_units':
		units_file = files

# Read the .hmm_params file
f = open(params_file, 'r')
params = []
for line in f.readlines():
	params.append(line)
f.close()

# Assign the params to variables
min_states = int(params[0])
max_states = int(params[1])
max_iterations = int(params[2])
threshold = float(params[3])
seeds = int(params[4])
edge_inertia = float(params[5])
dist_inertia = float(params[6])
#taste = int(params[7])
pre_stim = int(params[8])
bin_size = int(params[9])
pre_stim_hmm = int(params[10])
post_stim_hmm = int(params[11])

# Read the chosen units
f = open(units_file, 'r')
chosen_units = []
for line in f.readlines():
	chosen_units.append(int(line))
chosen_units = np.array(chosen_units) -1

# Open up hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Import all data and store relevant variables in lists
spikes = []
off_spikes = []
on_spikes = []

for taste in range(4):
    exec('spikes.append(hf5.root.spike_trains.dig_in_%i.spike_array[:])' % taste)
        
    # Slice out the required portion of the spike array, and bin it
    spikes[taste] = spikes[taste][:, chosen_units , pre_stim - pre_stim_hmm:pre_stim + post_stim_hmm]
    
    
    exec('dig_in = hf5.root.spike_trains.dig_in_%i' % taste)
    laser_exists = []
    try:
        laser_exists = dig_in.laser_durations[:]
    except:
        pass
    on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
    off_trials = np.where(dig_in.laser_durations[:] == 0.0)[0]
    
    # Bin spikes (might decrease info for fast spiking neurons)
    binned_spikes = np.zeros((spikes[taste].shape[0],spikes[taste].shape[1], int((pre_stim_hmm + post_stim_hmm)/bin_size)))
    time = []
    for i in range(spikes[taste].shape[0]):
        for j in range(spikes[taste].shape[1]):
            for k in range(binned_spikes.shape[2]):
                if (np.sum(spikes[taste][i, j, k*bin_size:(k+1)*bin_size]) > 0):
                    binned_spikes[i,j,k] = 1 
                    
    # Data must be recast into (neurons X trials X time) before it is fit
    dat_shape = (binned_spikes.shape[1],binned_spikes.shape[0],binned_spikes.shape[2])
    binned_spikes_re = binned_spikes.reshape(dat_shape)
    off_spikes.append(binned_spikes_re[:,off_trials,:])
    on_spikes.append(binned_spikes_re[:,on_trials,:])

#  ______ _ _     __  __           _      _ 
# |  ____(_) |   |  \/  |         | |    | |
# | |__   _| |_  | \  / | ___   __| | ___| |
# |  __| | | __| | |\/| |/ _ \ / _` |/ _ \ |
# | |    | | |_  | |  | | (_) | (_| |  __/ |
# |_|    |_|\__| |_|  |_|\___/ \__,_|\___|_|
#

# =============================================================================
# num_states = 3
# seeds = 10
# best_log_lik = 0
# 
# for seed in range(seeds):
#     
#     np.random.seed(seed)
#     model = DiscreteHMM.IndependentBernoulliHMM(num_states = num_states, num_emissions = binned_spikes.shape[1], 
#     max_iter = 1000, threshold = 1e-9)
#     
#     # Define probabilities and pseudocounts
#     p_transitions = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05) #(num_states X num_states)
#     p_emissions = np.random.random(size=(num_states, binned_spikes.shape[1])) #(num_states X num_emissions)
#     p_start = np.random.random(num_states) #(num_states)
#     start_pseuedocounts = np.ones(num_states) #(num_states)
#     transition_pseudocounts = np.abs(np.eye(num_states)*1500 - np.random.rand(num_states,num_states)*1500*0.05) #(num_states X num_states)
#     
#     # Emission pseudocounts : Average count of a neuron/trial, on and off in same ratio as firing probability 
#     avg_firing_p = np.tile(np.mean(binned_spikes,axis = (0,2)),(num_states,1))
#     avg_off_p = np.tile(np.ones((1,binned_spikes.shape[1])) - np.mean(binned_spikes,axis = (0,2)), (num_states,1))
#     emission_pseudocounts =  np.dstack((avg_firing_p,avg_off_p))*150 #(num_states X num_emissions X 2)
#     
#     model.fit(data=binned_spikes_re, p_transitions=p_transitions, p_emissions=p_emissions, 
#               p_start=p_start, transition_pseudocounts=transition_pseudocounts, emission_pseudocounts=emission_pseudocounts, 
#               start_pseudocounts=start_pseuedocounts, verbose = 0)
#     
#     current_log_lik = model.log_likelihood[-1] 
#     print(current_log_lik)
#     
#     if best_log_lik == 0:
#         best_log_lik = current_log_lik
#         best_model = model
#     elif current_log_lik < best_log_lik:
#         best_log_lik = current_log_lik
#         best_model = model
#     
# alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = best_model.E_step()
# =============================================================================

#  _____  _       _     _____        _        
# |  __ \| |     | |   |  __ \      | |       
# | |__) | | ___ | |_  | |  | | __ _| |_ __ _ 
# |  ___/| |/ _ \| __| | |  | |/ _` | __/ _` |
# | |    | | (_) | |_  | |__| | (_| | || (_| |
# |_|    |_|\___/ \__| |_____/ \__,_|\__\__,_|
#
# Define a function to plot data so that it can be called from inside a loop
# Function takes entire processed data and outputs plots

#plt.plot(range(0,int(binned_spikes.shape[2])*bin_size,bin_size),expected_latent_state[:, 0, :].T)
# Add more arguments, I dare you -_-

def plot_figs(binned_spikes, spikes, bin_size, expected_latent_state,chosen_units, taste_num, states_num, laser):
    
    try:
        os.makedirs('HMM_plots/%s/dig_in_%i/states_%i' % (laser,taste_num, states_num))
    except:
        shutil.rmtree('HMM_plots/%s/dig_in_%i/states_%i' % (laser,taste_num, states_num))
        os.makedirs('HMM_plots/%s/dig_in_%i/states_%i' % (laser,taste_num, states_num))
    
    for i in range(binned_spikes.shape[1]):
        fig = plt.figure()
        plt.plot(range(0,int(binned_spikes.shape[2])*bin_size,bin_size),expected_latent_state[:, i, :].T*len(chosen_units))
        for unit in range(len(chosen_units)):
            for j in range(spikes.shape[2]):
                if spikes[i, unit, j] > 0:
                    plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, linewidth = 0.5)
        plt.xlabel('Time post stimulus (ms)')
        plt.ylabel('Probability of HMM states')
        plt.title('Trial %i' % (i+1))
        fig.savefig('HMM_plots/%s/dig_in_%i/states_%i/Trial_%i.png' % (laser,taste_num, states_num, (i+1)))
        plt.close("all")


#  _____                _ _      _ _          _   _                       _   _                       _   
# |  __ \              | | |    | (_)        | | (_)                 /\  | | | |                     | |  
# | |__) |_ _ _ __ __ _| | | ___| |_ ______ _| |_ _  ___  _ __      /  \ | |_| |_ ___ _ __ ___  _ __ | |_ 
# |  ___/ _` | '__/ _` | | |/ _ \ | |_  / _` | __| |/ _ \| '_ \    / /\ \| __| __/ _ \ '_ ` _ \| '_ \| __|
# | |  | (_| | | | (_| | | |  __/ | |/ / (_| | |_| | (_) | | | |  / ____ \ |_| ||  __/ | | | | | |_) | |_ 
# |_|   \__,_|_|  \__,_|_|_|\___|_|_/___\__,_|\__|_|\___/|_| |_| /_/    \_\__|\__\___|_| |_| |_| .__/ \__|
#                                                                                              | |        
#                                                                                              |_|

# Cast HMM run as a function and use multiprocessing to run on multiple cores

def hmm_fit(binned_spikes,seed,num_states):
    np.random.seed(seed)
    model = DiscreteHMM.IndependentBernoulliHMM(num_states = num_states, num_emissions = binned_spikes.shape[0], 
    max_iter = 1000, threshold = 1e-9)

    # Define probabilities and pseudocounts
    p_transitions = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05) #(num_states X num_states)
    p_emissions = np.random.random(size=(num_states, binned_spikes.shape[0])) #(num_states X num_emissions)
    p_start = np.random.random(num_states) #(num_states)
    start_pseuedocounts = np.ones(num_states) #(num_states)
    transition_pseudocounts = np.abs(np.eye(num_states)*1500 - np.random.rand(num_states,num_states)*1500*0.05) #(num_states X num_states)
    
    # Emission pseudocounts : Average count of a neuron/trial, on and off in same ratio as firing probability 
    avg_firing_p = np.tile(np.mean(binned_spikes,axis = (1,2)),(num_states,1)) #(states X num_emissions)
    avg_off_p = np.tile(np.ones((1,binned_spikes.shape[0])) - np.mean(binned_spikes,axis = (1,2)), (num_states,1)) #(states X num_emissions)
    emission_pseudocounts =  np.dstack((avg_firing_p,avg_off_p))*150 #(num_states X num_emissions X 2)
    
    model.fit(data=binned_spikes, p_transitions=p_transitions, p_emissions=p_emissions, 
    p_start=p_start, transition_pseudocounts=transition_pseudocounts, emission_pseudocounts=emission_pseudocounts, 
    start_pseudocounts=start_pseuedocounts, verbose = 0)
    
    alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model.E_step()
    
    return model.log_likelihood[-1], expected_latent_state, model.converged
    
################################################

import multiprocessing as mp

num_states = range(3,8)
num_seeds = 100
n_cpu = int(sys.argv[1])

### OFF ###
for taste in range(4):
    for states in num_states:
        pool = mp.Pool(processes = n_cpu)
        results = [pool.apply_async(hmm_fit, args = (off_spikes[taste], seed, states)) for seed in range(seeds)]
        output = [p.get() for p in results]
        
        log_probs = [output[i][1] for i in range(len(output))]
        maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
        fin_out = output[maximum_pos]
    
        plot_figs(off_spikes[taste],spikes[taste],10,fin_out[1],chosen_units,taste,states,'off')
        
### ON ###
for taste in range(4):
    for states in num_states:
        pool = mp.Pool(processes = n_cpu)
        results = [pool.apply_async(hmm_fit, args = (on_spikes[taste], seed, states)) for seed in range(seeds)]
        output = [p.get() for p in results]
        
        log_probs = [output[i][1] for i in range(len(output))]
        maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
        fin_out = output[maximum_pos]
    
        plot_figs(off_spikes[taste],spikes[taste],10,fin_out[1],chosen_units,taste,states,'on')
