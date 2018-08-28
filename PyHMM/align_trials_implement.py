# Load spikes
# Load state probabilities
# Realign spikes around state transitions
# Output realigned spikes to hdf5 and plots


# Import stuff
import os
import numpy as np
import pylab as plt
import matplotlib
import tables
from align_trials_funcs import *

"""
 _                     _             _ _    _                   _       _        
| |                   | |           (_) |  (_)                 | |     | |       
| |     ___   __ _  __| |  ___ _ __  _| | ___ _ __   __ _    __| | __ _| |_ __ _ 
| |    / _ \ / _` |/ _` | / __| '_ \| | |/ / | '_ \ / _` |  / _` |/ _` | __/ _` |
| |___| (_) | (_| | (_| | \__ \ |_) | |   <| | | | | (_| | | (_| | (_| | || (_| |
|______\___/ \__,_|\__,_| |___/ .__/|_|_|\_\_|_| |_|\__, |  \__,_|\__,_|\__\__,_|
                              | |                    __/ |                       
                              |_|                   |___/
"""

for file in [1]:
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
    #threshold = float(params[3])
    #seeds = int(params[4])
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
        spikes[taste] = spikes[taste][:, chosen_units ,:]
        spikes[taste] = np.swapaxes(spikes[taste],0,1)
        
        
        exec('dig_in = hf5.root.spike_trains.dig_in_%i' % taste)
        laser_exists = []
        try:
            laser_exists = dig_in.laser_durations[:]
        except:
            pass
        on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
        off_trials = np.where(dig_in.laser_durations[:] == 0.0)[0]
                    
        off_spikes.append(spikes[taste][:,off_trials,:])
        on_spikes.append(spikes[taste][:,on_trials,:])

"""
  _                     _   _____           _           _     _ _ _ _   _           
 | |                   | | |  __ \         | |         | |   (_) (_) | (_)          
 | |     ___   __ _  __| | | |__) | __ ___ | |__   __ _| |__  _| |_| |_ _  ___  ___ 
 | |    / _ \ / _` |/ _` | |  ___/ '__/ _ \| '_ \ / _` | '_ \| | | | __| |/ _ \/ __|
 | |___| (_) | (_| | (_| | | |   | | | (_) | |_) | (_| | |_) | | | | |_| |  __/\__ \
 |______\___/ \__,_|\__,_| |_|   |_|  \___/|_.__/ \__,_|_.__/|_|_|_|\__|_|\___||___/
"""                                                                                    

    for cond_dir in ['off']:
            
        var_aligned = []
        for model_num_states in [3]: #range(min_states,max_states+1):
            for taste in [1,2,3]: #range(len(off_spikes)):
                
                exec('data = %s_spikes[taste]' % cond_dir)
                
                map_hmm_node = '/map_hmm/%s/taste_%i/%i_states/expected_latent_state' % (cond_dir, taste, model_num_states)
                var_hmm_node = '/var_hmm/%s/taste_%i/%i_states/expected_latent_state' % (cond_dir, taste, model_num_states)
                
                map_probs = hf5.get_node(map_hmm_node)[:]
                var_probs = hf5.get_node(var_hmm_node)[:]
                
                firing = calc_firing_rate(data,250,25)
                firing_re = np.reshape(firing,(int(firing.size/firing.shape[2]),firing.shape[2]))
                
                transitions = calc_trans(var_probs, range(80,150), 0.6, start_t, bin_size)

# =============================================================================
#                 firing2 = firing
#                 max_val = np.max(firing)*1.2
#                 for i in range(firing2.shape[1]):
#                     firing2[:,i,int(transitions[i]/25)] = max_val
#                 firing2_re = np.reshape(firing2,(int(firing2.size/firing2.shape[2]),firing2.shape[2]))
#                 
#                 plt.imshow(firing2_re)
# =============================================================================
                
                firing_aligned, firing_unaligned = ber_align_trials(firing, transitions, 0, 25, 2000)
                firing_aligned_re = np.reshape(firing_aligned,(int(firing_aligned.size/firing_aligned.shape[2]),firing_aligned.shape[2]))
                firing_unaligned_re = np.reshape(firing_unaligned,(int(firing_unaligned.size/firing_unaligned.shape[2]),firing_unaligned.shape[2]))
                

                plt.subplot(121)
                plt.imshow(firing_aligned_re)
                plt.subplot(122)
                plt.imshow(firing_unaligned_re)
                
                for i in range(firing_aligned_re.shape[0]):
                    firing_aligned_re[i,:] = firing_aligned_re[i,:]/np.max(firing_aligned_re[i,:])
                    firing_unaligned_re[i,:] = firing_unaligned_re[i,:]/np.max(firing_unaligned_re[i,:])
                
                
                ax1 = plt.subplot(121)
                plt.plot(np.nansum(firing_aligned_re,axis = 0))
                plt.subplot(122, sharex = ax1, sharey = ax1)
                plt.plot(np.nansum(firing_unaligned_re,axis = 0))
"""
           _ _               _______   _       _     
     /\   | (_)             |__   __| (_)     | |    
    /  \  | |_  __ _ _ __      | |_ __ _  __ _| |___ 
   / /\ \ | | |/ _` | '_ \     | | '__| |/ _` | / __|
  / ____ \| | | (_| | | | |    | | |  | | (_| | \__ \
 /_/    \_\_|_|\__, |_| |_|    |_|_|  |_|\__,_|_|___/
                __/ |                                
               |___/
"""