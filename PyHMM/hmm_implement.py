# Load spiking data
# Define model, Fit HMM
# 

# Import stuff# Impor 
import numpy as np
import DiscreteHMM
import pylab as plt

import matplotlib

import tables
import easygui
import sys
import os
import shutil # needed to remove nested directories system doesn't take care of


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

# Pull out the NSLOTS - number of CPUs allotted
#n_cpu = int(os.getenv('NSLOTS'))
n_cpu = 4 #int(sys.argv[1])

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

taste = 0
exec('spikes = hf5.root.spike_trains.dig_in_%i.spike_array[:]' % taste)
    
# Slice out the required portion of the spike array, and bin it
spikes = spikes[:, chosen_units , pre_stim - pre_stim_hmm:pre_stim + post_stim_hmm]
exec('dig_in = hf5.root.spike_trains.dig_in_%i' % taste)
laser_exists = []
try:
    laser_exists = dig_in.laser_durations[:]
except:
    pass
on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
off_trials = np.where(dig_in.laser_durations[:] == 0.0)[0]

# Bin spikes (might decrease info for fast spiking neurons)
binned_spikes = np.zeros((spikes.shape[0],spikes.shape[1], int((pre_stim_hmm + post_stim_hmm)/bin_size)))
time = []
for i in range(spikes.shape[0]):
    for j in range(spikes.shape[1]):
        for k in range(binned_spikes.shape[2]):
            if (np.sum(spikes[i, j, k*bin_size:(k+1)*bin_size]) > 0):
                binned_spikes[i,j,k] = 1 
                
off_spikes = binned_spikes[off_trials,:,:]
on_spikes = binned_spikes[on_trials,:,:]
dat_shape = (binned_spikes.shape[1],binned_spikes.shape[0],binned_spikes.shape[2])
binned_spikes_re = binned_spikes.reshape(dat_shape)
#  ______ _ _     __  __           _      _ 
# |  ____(_) |   |  \/  |         | |    | |
# | |__   _| |_  | \  / | ___   __| | ___| |
# |  __| | | __| | |\/| |/ _ \ / _` |/ _ \ |
# | |    | | |_  | |  | | (_) | (_| |  __/ |
# |_|    |_|\__| |_|  |_|\___/ \__,_|\___|_|
#

num_states = 3

model = DiscreteHMM.IndependentBernoulliHMM(num_states = num_states, num_emissions = binned_spikes.shape[1], 
        max_iter = 1000, threshold = 1e-6)

# Define probabilities and pseudocounts
p_transitions = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05) #(num_states X num_states)
p_emissions = np.random.random(size=(num_states, binned_spikes.shape[1])) #(num_states X num_emissions)
p_start = np.random.random(num_states) #(num_states)
start_pseuedocounts = np.ones(num_states) #(num_states)
transition_pseudocounts = np.abs(np.eye(num_states)*1500 - np.random.rand(num_states,num_states)*1500*0.05) #(num_states X num_states)

# Emission pseudocounts : Average count of a neuron/trial, on and off in same ratio as firing probability 
avg_firing_p = np.tile(np.mean(binned_spikes,axis = (0,2)),(num_states,1))
avg_off_p = np.tile(np.ones((1,binned_spikes.shape[1])) - np.mean(binned_spikes,axis = (0,2)), (num_states,1))
emission_pseudocounts =  np.dstack((avg_firing_p,avg_off_p))*150 #(num_states X num_emissions X 2)

model.fit(data=binned_spikes_re, p_transitions=p_transitions, p_emissions=p_emissions, 
          p_start=p_start, transition_pseudocounts=transition_pseudocounts, emission_pseudocounts=emission_pseudocounts, 
          start_pseudocounts=start_pseuedocounts, verbose = False)

alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model.E_step()
plt.plot(expected_latent_state[:, 0, :].T)
