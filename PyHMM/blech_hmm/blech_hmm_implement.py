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
from hinton import hinton

from blech_hmm import *
utils.enable_gpu()

from fake_firing import raster

plt.ioff()


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
    data_dir = '/media/bigdata/jian_you_data/pomegranate_hmm/file_%i/' % file
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
    fin_binned_spikes = []
    all_off_trials = []
    all_on_trials = []
    
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
        
        all_off_trials.append(off_trials + ((taste+1)*15))
        all_on_trials.append(on_trials + ((taste+1)*15))
        
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
        
        fin_binned_spikes.append(cat_binned_spikes)            
        off_spikes.append(cat_binned_spikes[off_trials,:])
        on_spikes.append(cat_binned_spikes[on_trials,:])
        
    binned_spikes_array = np.asarray(fin_binned_spikes)
    binned_spikes_array = np.reshape(binned_spikes_array,(binned_spikes_array.shape[0]*binned_spikes_array.shape[1],binned_spikes_array.shape[2]))
    all_off_trials = np.concatenate(np.asarray(all_off_trials))
    all_on_trials = np.concatenate(np.asarray(all_on_trials))        
    ########################################
                    
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
            
            ### Taste by taste ###
            for taste in range(len(off_spikes)):
                
                folder_name = 'per_taste/taste_%i/%i_states' % (taste,model_num_states)
                if os.path.isdir(folder_name):
                    shutil.rmtree(folder_name)
                os.makedirs(folder_name)
                
                exec('trials = %s_trials' % cond_dir)
                
                
                # Pomegranate HMM from blech_clust               
                model_pom, \
                log_prob, \
                AIC, \
                BIC, \
                state_emissions,\
                state_transitions, \
                post_probs  = multinomial_hmm_implement(
                        n_states= model_num_states, 
                        threshold = 1e-9, 
                        seeds = seed_num, 
                        n_cpu = 32, 
                        binned_spikes = fin_binned_spikes[taste], 
                        off_trials = trials,
                        edge_inertia = 0, 
                        dist_inertia = 0)
                
                # Save figures in appropriate directories
                data = fin_binned_spikes[taste][trials,:]
                expected_latent_state = np.swapaxes(post_probs.T,1,2)
                for i in range(data.shape[0]):
                    fig = plt.figure()
                    raster(data = data[i,:],expected_latent_state = expected_latent_state[:,i,:])
                    fig.savefig(folder_name + '/' + '%i_pom_%ist.png' % (i,model_num_states))
                    plt.close(fig)
                
# =============================================================================
#                 plt.figure()
#                 hinton(state_transitions)
#                 plt.title('Log_lik = %f' % log_prob)
#                 plt.savefig(folder_name + '/' + 'hinton_pom_%ist.png' % model_num_states)
#                 plt.close('all')
# =============================================================================

                # Save data in appropriate spots in HDF5 file
                node_name = '/pom_hmm/%s/per_taste/taste_%i/states_%i' % (cond_dir, taste, model_num_states)
                
                try:
                    hf5.remove_node(node_name,recursive =  True)
                except:
                    pass
                
                hf5.create_group('/pom_hmm/%s/per_taste/taste_%i' % (cond_dir, taste), 'states_%i' % model_num_states,createparents=True)
                
                hf5.create_array(node_name,'expected_latent_state', expected_latent_state)
                hf5.create_array(node_name,'log_prob', log_prob)
                hf5.create_array(node_name,'p_transitions', state_transitions)
                #hf5.create_array(node_name,'p_emissions', np.asarray(state_emissions))
                hf5.flush()
                print('file %i, %s, taste %i, state %i' % (file, cond_dir, taste, model_num_states))
            
            ### All tastes together ###
            folder_name = 'all_tastes/%i_states' % (model_num_states)
            if os.path.isdir(folder_name):
                shutil.rmtree(folder_name)
            os.makedirs(folder_name)            
            
            # Pomegranate HMM from blech_clust               
            model_pom, \
            log_prob, \
            AIC, \
            BIC, \
            state_emissions,\
            state_transitions, \
            post_probs  = multinomial_hmm_implement(
                    n_states= model_num_states, 
                    threshold = 1e-9, 
                    seeds = seed_num, 
                    n_cpu = 32, 
                    binned_spikes = binned_spikes_array,
                    off_trials = all_off_trials,
                    edge_inertia = 0, 
                    dist_inertia = 0)
            
            # Save figures in appropriate directories
            data = binned_spikes_array[all_off_trials,:]
            expected_latent_state = np.swapaxes(post_probs.T,1,2)
            for i in range(data.shape[0]):
                fig = plt.figure()
                raster(data = data[i,:],expected_latent_state = expected_latent_state[:,i,:])
                fig.savefig(folder_name + '/' + '%i_pom_%ist.png' % (i,model_num_states))
                plt.close(fig)
            
# =============================================================================
#             plt.figure()
#             hinton(state_transitions)
#             plt.title('Log_lik = %f' % log_prob)
#             plt.savefig(folder_name + '/' + 'hinton_pom_%ist.png' % model_num_states)
#             plt.close('all')
# =============================================================================

            # Save data in appropriate spots in HDF5 file
            node_name = '/pom_hmm/%s/all_tastes/states_%i' % (cond_dir, model_num_states)
            
            try:
                hf5.remove_node(node_name,recursive =  True)
            except:
                pass
            
            hf5.create_group('/pom_hmm/%s/all_tastes' % cond_dir, 'states_%i' % model_num_states, createparents=True)
            
            hf5.create_array(node_name,'expected_latent_state', expected_latent_state)
            hf5.create_array(node_name,'log_prob', log_prob)
            hf5.create_array(node_name,'p_transitions', state_transitions)
            #hf5.create_array(node_name,'p_emissions', state_emissions)
            hf5.flush()
            print('file %i, %s, all_tastes, state %i' % (file, cond_dir, model_num_states))
            

    hf5.close()
