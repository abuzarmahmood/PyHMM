# Load spiking data
# Define model, Fit HMM
# 

# Import stuff

import numpy as np
import pylab as plt
import matplotlib
import tables
import os
import shutil

os.chdir('/media/bigdata/PyHMM/PyHMM/')
import DiscreteHMM
from hmm_fit_funcs import *
from align_trials import *
from hinton import hinton
from fake_firing import raster

plt.ioff() # Prevent plots from showing


#  _                     _             _ _    _                   _       _        
# | |                   | |           (_) |  (_)                 | |     | |       
# | |     ___   __ _  __| |  ___ _ __  _| | ___ _ __   __ _    __| | __ _| |_ __ _ 
# | |    / _ \ / _` |/ _` | / __| '_ \| | |/ / | '_ \ / _` |  / _` |/ _` | __/ _` |
# | |___| (_) | (_| | (_| | \__ \ |_) | |   <| | | | | (_| | | (_| | (_| | || (_| |
# |______\___/ \__,_|\__,_| |___/ .__/|_|_|\_\_|_| |_|\__, |  \__,_|\__,_|\__\__,_|
#                               | |                    __/ |                       
#                               |_|                   |___/
#

for file in range(1,7):
    data_dir = '/media/bigdata/jian_you_data/random_ic/file_%i/' % file
    os.chdir(data_dir)
    
    
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
    #taste = int(params[5])
    start_t = int(params[6])
    end_t = int(params[7])
    bin_size = int(params[8])
    
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
        spikes[taste] = spikes[taste][:, chosen_units , start_t:end_t]
        
        
        exec('dig_in = hf5.root.spike_trains.dig_in_%i' % taste)
        laser_exists = []
        try:
            laser_exists = dig_in.laser_durations[:]
        except:
            pass
        on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
        off_trials = np.where(dig_in.laser_durations[:] == 0.0)[0]
        
        # Bin spikes (might decrease info for fast spiking neurons)
        binned_spikes = np.zeros((spikes[taste].shape[0],spikes[taste].shape[1], int((end_t - start_t)/bin_size)))
        for i in range(spikes[taste].shape[0]): # Loop over trials
            for j in range(spikes[taste].shape[1]): # Loop over neurons
                for k in range(binned_spikes.shape[2]): # Loop over time
                    if (np.sum(spikes[taste][i, j, k*bin_size:(k+1)*bin_size]) > 0):
                        binned_spikes[i,j,k] = 1
                    
    ######### For categorical HMM ########  
        # Remove multiple spikes in same time bin (for categorical HMM)
        for i in range(spikes[taste].shape[0]): # Loop over trials
            for k in range(binned_spikes.shape[2]): # Loop over time
                n_firing_units = np.where(binned_spikes[i,:,k] > 0)[0]
                if len(n_firing_units)>0:
                    binned_spikes[i,:,k] = 0
                    binned_spikes[i,np.random.choice(n_firing_units),k] = 1
        
        # Convert bernoulli trials to categorical data        
        cat_binned_spikes = np.zeros((binned_spikes.shape[0],binned_spikes.shape[2]))
        for i in range(cat_binned_spikes.shape[0]):
            for j in range(cat_binned_spikes.shape[1]):
                firing_unit = np.where(binned_spikes[i,:,j] > 0)[0]
                if firing_unit.size > 0:
                    cat_binned_spikes[i,j] = firing_unit + 1
                    
        off_spikes.append(cat_binned_spikes[off_trials,:])
        on_spikes.append(cat_binned_spikes[on_trials,:])
        
    ########################################
                    
        
        
    # =============================================================================
    #     # Data must be recast into (neurons X trials X time) before it is fit
    #     dat_shape = (binned_spikes.shape[1],binned_spikes.shape[0],binned_spikes.shape[2])
    #     binned_spikes_re = np.swapaxes(binned_spikes,0,1)
    #     off_spikes.append(binned_spikes_re[:,off_trials,:])
    #     on_spikes.append(binned_spikes_re[:,on_trials,:])
    # =============================================================================
        
    # =============================================================================
    # for taste in [0]:#range(len(off_spikes)):
    #     for trial in range(off_spikes[0].shape[1]):
    #         plt.figure()
    #         raster(off_spikes[taste][:,trial,:])
    # =============================================================================
    
    #  ______ _ _     __  __           _      _ 
    # |  ____(_) |   |  \/  |         | |    | |
    # | |__   _| |_  | \  / | ___   __| | ___| |
    # |  __| | | __| | |\/| |/ _ \ / _` |/ _ \ |
    # | |    | | |_  | |  | | (_) | (_| |  __/ |
    # |_|    |_|\__| |_|  |_|\___/ \__,_|\___|_|
    #
    
    seed_num = 100
    
    for cond_dir in ['off', 'on']:
    
        if os.path.isdir(data_dir + cond_dir):
            shutil.rmtree(data_dir + cond_dir)
        os.mkdir(data_dir + cond_dir)
        os.chdir(data_dir + cond_dir)
        
        for model_num_states in range(min_states,max_states+1):
            for taste in range(len(off_spikes)):
                
                folder_name = 'taste_%i/%i_states' % (taste,model_num_states)
                if os.path.isdir(folder_name):
                    shutil.rmtree(folder_name)
                os.makedirs(folder_name)
                
                #data = off_spikes[taste]
                exec('data = %s_spikes[taste]' % cond_dir)
                
                
                # Variational Inference HMM
                model_VI, model_MAP = hmm_cat_var_multi(data,seed_num,model_num_states,initial_conds_type = 'rand',1500,1e-6)
                
                ### MAP Outputs ###
                alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_MAP.E_step()
                # Save figures in appropriate directories
                for i in range(data.shape[0]):
                    fig = plt.figure()
                    raster(data = data[i,:],expected_latent_state = expected_latent_state[:,i,:])
                    fig.savefig(folder_name + '/' + '%i_map_%ist.png' % (i,model_num_states))
                    plt.close(fig)
                
                fig = plt.figure()
                hinton(model_MAP.p_transitions.T)
                plt.title('Log_lik = %f' %model_MAP.log_posterior[-1])
                plt.suptitle('Model converged = ' + str(model_MAP.converged))
                fig.savefig(folder_name + '/' + 'hinton_map_%ist.png' % model_num_states)
                plt.close(fig)

                # Save data in appropriate spots in HDF5 file
                node_name = '/map_hmm/%s/taste_%i/states_%i' % (cond_dir, taste, model_num_states)
                
                try:
                    hf5.remove_node(node_name,recursive =  True)
                except:
                    pass
                
                hf5.create_group('/map_hmm/%s/taste_%i' % (cond_dir, taste), 'states_%i' % model_num_states,createparents=True)
                
                hf5.create_array(node_name,'expected_latent_state',expected_latent_state)
                hf5.create_array(node_name,'log_posterior',model_MAP.log_posterior[-1])
                hf5.create_array(node_name,'p_transitions',model_MAP.p_transitions)
                hf5.create_array(node_name,'p_emissions',model_MAP.p_emissions)
                hf5.create_array(node_name,'model_converged_val',model_MAP.converged*1)
                hf5.flush()

                ### VI Outputs ###
                alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_VI.E_step()
                # Save figures in appropriate directories
                for i in range(data.shape[0]):
                    fig = plt.figure()
                    raster(data = data[i,:],expected_latent_state = expected_latent_state[:,i,:])
                    fig.savefig(folder_name + '/' + '%i_var_%ist.png' % (i,model_num_states))
                    plt.close(fig)
                
                fig = plt.figure()
                hinton(model_VI.transition_counts)
                plt.title('ELBO = %f' %model_VI.ELBO[-1])
                plt.suptitle('Model converged = ' + str(model_VI.converged))
                fig.savefig(folder_name + '/' + 'hinton_var_%ist.png' % model_num_states)
                plt.close(fig)
                
                # Save data in appropriate spots in HDF5 file
                node_name = '/var_hmm/%s/taste_%i/states_%i' % (cond_dir, taste, model_num_states)
                
                try:
                    hf5.remove_node(node_name,recursive =  True)
                except:
                    pass
                
                hf5.create_group('/var_hmm/%s/taste_%i' % (cond_dir, taste), 'states_%i' % model_num_states,createparents=True)
                
                hf5.create_array(node_name,'expected_latent_state',expected_latent_state)
                hf5.create_array(node_name,'ELBO',model_VI.ELBO[-1])
                hf5.create_array(node_name,'p_transitions',model_VI.p_transitions)
                hf5.create_array(node_name,'p_emissions',model_VI.p_emissions)
                hf5.create_array(node_name,'model_converged_val',model_VI.converged*1)
                hf5.flush()
            

    hf5.close()
