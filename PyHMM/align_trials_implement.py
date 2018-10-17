# Load spikes
# Load state probabilities
# Realign spikes around state transitions
# Output realigned spikes to hdf5 and plots


# Import stuff
import os
import shutil
import numpy as np
import pylab as plt
import matplotlib
import tables
from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr
import matplotlib.gridspec as gridspec

os.chdir('/media/bigdata/PyHMM/PyHMM')
from align_trials_funcs import *
from hinton import hinton

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data

plt.ioff()


# _                     _             _ _    _                   _       _        
#| |                   | |           (_) |  (_)                 | |     | |       
#| |     ___   __ _  __| |  ___ _ __  _| | ___ _ __   __ _    __| | __ _| |_ __ _ 
#| |    / _ \ / _` |/ _` | / __| '_ \| | |/ / | '_ \ / _` |  / _` |/ _` | __/ _` |
#| |___| (_) | (_| | (_| | \__ \ |_) | |   <| | | | | (_| | | (_| | (_| | || (_| |
#|______\___/ \__,_|\__,_| |___/ .__/|_|_|\_\_|_| |_|\__, |  \__,_|\__,_|\__\__,_|
#                              | |                    __/ |                       
#                              |_|                   |___/
#
palatability_ranks = np.asarray([3,4,1,2])

for file in range(1,7):
    data_dir = '/media/bigdata/jian_you_data/des_ic/file_%i/' % file
    plot_dir = data_dir + 'plots/'
    os.chdir(data_dir)
    
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)
    
    
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
    # For alignment
    spikes = []
    off_spikes = []
    on_spikes = []
    all_off_trials = []
    all_on_trials = []
    

    
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
        
        all_off_trials.append(off_trials + taste*len(off_trials)*2)
        all_on_trials.append(on_trials + taste*len(on_trials)*2)
                    
        off_spikes.append(spikes[taste][:,off_trials,:])
        on_spikes.append(spikes[taste][:,on_trials,:])
        
    all_off_trials = np.concatenate(np.asarray(all_off_trials))
    all_on_trials = np.concatenate(np.asarray(all_on_trials))
    
    all_on_firing_aligned = np.zeros((spikes[0].shape[0],all_off_trials.size,int(2000/25)))
    all_on_firing_unaligned = np.zeros((spikes[0].shape[0],all_off_trials.size,int(2000/25)))
    all_off_firing_aligned = np.zeros((spikes[0].shape[0],all_off_trials.size,int(2000/25)))
    all_off_firing_unaligned = np.zeros((spikes[0].shape[0],all_off_trials.size,int(2000/25)))
        


#           _ _               _______   _       _     
#     /\   | (_)             |__   __| (_)     | |    
#    /  \  | |_  __ _ _ __      | |_ __ _  __ _| |___ 
#   / /\ \ | | |/ _` | '_ \     | | '__| |/ _` | / __|
#  / ____ \| | | (_| | | | |    | | |  | | (_| | \__ \
# /_/    \_\_|_|\__, |_| |_|    |_|_|  |_|\__,_|_|___/
#                __/ |                                
#               |___/
#

    for model_num_states in range(min_states,max_states+1):
        for cond_dir in ['off','on']:
            
            os.chdir(data_dir + cond_dir)
            for taste in range(len(off_spikes)):
                
                
                folder_name = 'taste_%i/%i_states' % (taste,model_num_states)
                
                exec('data = %s_spikes[taste]' % cond_dir)
                
                var_hmm_node = '/map_hmm/%s/taste_%i/%i_states/expected_latent_state' % (cond_dir, taste, model_num_states)
                var_probs = hf5.get_node(var_hmm_node)[:]
                p_emissions_node = '/map_hmm/%s/taste_%i/%i_states/p_emissions' % (cond_dir, taste, model_num_states)
                p_emissions = hf5.get_node(p_emissions_node)[:]
                
                # Convert spikes to firing rate
                firing = calc_firing_rate(data,250,25)
                
                # Get transition times from HMM fits
                transitions, flags = calc_transitions(
                        var_probs, 
                        800,
                        2000,
                        0.5,
                        start_t, 
                        bin_size,
                        delta = 0.1)
                
                # Use HMM fits to align firing rate
                firing_aligned, firing_unaligned = ber_align_trials(
                        firing, 
                        transitions, 
                        0, 
                        25, 
                        2000)

                exec('all_%s_firing_aligned[:,(palatability_ranks[taste]-1)*firing_aligned.shape[1]: (palatability_ranks[taste])*firing_aligned.shape[1],:] = firing_aligned' % cond_dir)
                exec('all_%s_firing_unaligned[:,(palatability_ranks[taste]-1)*firing_aligned.shape[1]: (palatability_ranks[taste])*firing_aligned.shape[1],:] = firing_unaligned' % cond_dir)
            
                # Plot all aligned and unaligned trials
                firing_aligned_re = np.reshape(firing_aligned,(int(firing_aligned.size/firing_aligned.shape[2]),firing_aligned.shape[2]))
                firing_unaligned_re = np.reshape(firing_unaligned,(int(firing_unaligned.size/firing_unaligned.shape[2]),firing_unaligned.shape[2]))
                
                fig = plt.figure()
                plt.subplot(121)
                plt.imshow(firing_aligned_re)
                plt.subplot(122)
                plt.imshow(firing_unaligned_re)
                fig.savefig(folder_name + '/' + 'var_%ist_all.png' % (model_num_states))
                plt.close(fig)
                
                # Convert mean 'change' caused by alignment
                mean_aligned = np.mean(firing_aligned,axis=1)
                mean_unaligned = np.mean(firing_unaligned,axis=1)
                
                for i in range(mean_aligned.shape[0]):
                    mean_aligned[i,:] = preprocessing.scale(mean_aligned[i,:])
                    mean_unaligned[i,:] = preprocessing.scale(mean_unaligned[i,:])

                # Make sure direction of change in firing rate is uniform
                mid = int(mean_aligned.shape[1]/2)
                nrn_flip = np.sum(mean_aligned[:,range(0,mid)],axis=1) > 0
                
                mean_aligned[nrn_flip,:] = mean_aligned[nrn_flip,:]*-1
                mean_unaligned[nrn_flip,:] = mean_unaligned[nrn_flip,:]*-1
                
                fig = plt.figure()
                plt.subplot(311)
                plt.imshow(mean_aligned)
                plt.subplot(312)
                plt.imshow(mean_unaligned)
                plt.subplot(313)
                plt.plot(np.sum(mean_aligned,axis=0),label = 'Aligned')
                plt.plot(np.sum(mean_unaligned,axis=0), label = 'Unaligned')
                plt.legend()
                fig.savefig(folder_name + '/' + 'var_%ist_mean.png' % (model_num_states))
                plt.close(fig)
                
                # Plot neuron emissions
                fig = plt.figure()
                hinton(p_emissions)
                fig.savefig(folder_name + '/' + 'var_emissions_%ist.png' % model_num_states)
                plt.close(fig)
                
               print('file%i, %s, model%i, taste%i' %(file,cond_dir,model_num_states,taste))
#   _____      _            _       _          _____                
#  / ____|    | |          | |     | |        / ____|               
# | |     __ _| | ___ _   _| | __ _| |_ ___  | |     ___  _ __ _ __ 
# | |    / _` | |/ __| | | | |/ _` | __/ _ \ | |    / _ \| '__| '__|
# | |___| (_| | | (__| |_| | | (_| | ||  __/ | |___| (_) | |  | |   
#  \_____\__,_|_|\___|\__,_|_|\__,_|\__\___|  \_____\___/|_|  |_|   
#
# Start -> tastes x neurons x trials x time
# For every time bin, correlate firing across all trials of a neuron with taste ranks -> neurons x time
                # e.g. 15 data points for every palatability rank if 15 trials per taste
# Average ABSOLUTE correlations across neurons -> time
# Repeat for both ON and OFF conditions of every model state
            
            palatability_vec = np.reshape(np.tile(palatability_ranks[:,np.newaxis],15),len(all_off_trials))
            exec('spearman_%s_%i_aligned = np.zeros((all_off_firing_aligned.shape[0],all_off_firing_aligned.shape[2]))' % (cond_dir, model_num_states))
            exec('spearman_%s_%i_unaligned = np.zeros((all_off_firing_aligned.shape[0],all_off_firing_aligned.shape[2]))' % (cond_dir, model_num_states))
            for nrn in range(all_off_firing_aligned.shape[0]):
                for time in range(all_off_firing_aligned.shape[2]):
                    exec('spearman_%s_%i_aligned[nrn,time] = spearmanr(all_%s_firing_aligned[nrn,:,time],palatability_vec)[0]' % (cond_dir, model_num_states, cond_dir))
                    exec('spearman_%s_%i_unaligned[nrn,time] = spearmanr(all_%s_firing_unaligned[nrn,:,time],palatability_vec)[0]' % (cond_dir, model_num_states, cond_dir))

            fig = plt.figure()
            exec("plt.plot(np.mean(spearman_%s_%i_unaligned**2,axis=0),label='un')" % (cond_dir, model_num_states))
            exec("plt.plot(np.mean(spearman_%s_%i_aligned**2,axis=0),label='al')" % (cond_dir, model_num_states))
            plt.legend()
            plt.show()
            exec("fig.savefig(plot_dir+ 'spearman_%s_%i.png')" % (cond_dir, model_num_states))
            plt.close(fig)
            print('correlation: %s, %i' % (cond_dir, model_num_states))

        for i in range(all_off_firing_aligned.shape[0]):
            fig = pic_plot(all_off_firing_aligned,all_off_firing_unaligned,i,palatability_ranks)
            fig.savefig(plot_dir + 'state%i_nrn%i_map.png' % (model_num_states, i))
            plt.close(fig)
    
    hf5.close()
    
def error_plot(data):
    """
    y is a matrix (trials x time)
    """
    plt.errorbar(x = range(data.shape[1]), y = np.mean(data,axis=0),
                 yerr = np.std(data,axis = 0))
    plt.show()

def pic_plot(a,b,ind, palatability_ranks):
    """
    a and b are 3D arrays (neurons x trials x time)
    """

    fig = plt.figure()
    gs = gridspec.GridSpec(nrows = 7, ncols = 3)
    
    fig.add_subplot(gs[2:-1,0])
    plt.imshow(preprocessing.scale(a[ind,:,:],axis=1), interpolation='nearest', aspect='auto')
    plt.title('aligned')
    
    fig.add_subplot(gs[2:-1,1])
    plt.imshow(preprocessing.scale(b[ind,:,:],axis=1), interpolation='nearest', aspect='auto')
    plt.title('unaligned')
    
    fig.add_subplot(gs[2,2])
    plt.plot(np.mean(a[ind,0:15,:],axis = 0))
    plt.plot(np.mean(b[ind,0:15,:],axis = 0))
    plt.title('All tastes')
    fig.add_subplot(gs[3,2])
    plt.plot(np.mean(a[ind,15:30,:],axis = 0))
    plt.plot(np.mean(b[ind,15:30,:],axis = 0))
    fig.add_subplot(gs[4,2])
    plt.plot(np.mean(a[ind,30:45,:],axis = 0))
    plt.plot(np.mean(b[ind,30:45,:],axis = 0))
    fig.add_subplot(gs[5,2])
    plt.plot(np.mean(a[ind,45:60,:],axis = 0))
    plt.plot(np.mean(b[ind,45:60,:],axis = 0))
    
    a2 = preprocessing.scale(a[ind,:,:],axis=1)
    b2 = preprocessing.scale(b[ind,:,:],axis=1)
    mid = int(a2.shape[1]/2)
    nrn_flip = np.sum(a[ind,:,range(0,mid)].T,axis=1) > np.sum(a[ind,:,range(mid,a.shape[2])].T,axis=1)
    a2[nrn_flip,:] = a2[nrn_flip,:]*-1
    b2[nrn_flip,:] = b2[nrn_flip,:]*-1
    
    fig.add_subplot(gs[-1,0])
    plt.plot(np.mean(a2,axis=0))
    plt.plot(np.mean(b2,axis=0))
    plt.title('Mean firing adjusted')
    
    fig.add_subplot(gs[-1,1])
    plt.plot(np.mean(a[ind,:,:],axis=0))
    plt.plot(np.mean(b[ind,:,:],axis=0))
    plt.title('Mean firing')
    
    fig.add_subplot(gs[1,0])
    palatability_vec = np.reshape(np.tile(palatability_ranks[:,np.newaxis],15),60)
    spearman_a,spearman_b = np.zeros(a.shape[2]),np.zeros(a.shape[2])
    spearman_p_a, spearman_p_b = np.zeros(a.shape[2]),np.zeros(a.shape[2])
    for time in range(a.shape[2]):
        spearman_a[time] = spearmanr(a[ind,:,time],palatability_vec)[0]
        spearman_b[time] = spearmanr(b[ind,:,time],palatability_vec)[0]
        spearman_p_a[time] = spearmanr(a[ind,:,time],palatability_vec)[1]
        spearman_p_b[time] = spearmanr(b[ind,:,time],palatability_vec)[1]
    plt.plot(spearman_a)
    plt.plot(spearman_b)
    plt.title('Spearman_rho')
    plt.axhline()
    
    
    fig.add_subplot(gs[0,0])
    plt.plot(spearman_p_a)
    plt.plot(spearman_p_b)
    plt.title('Spearman_p')
    
    fig.add_subplot(gs[1,1])
    pearson_a,pearson_b = np.zeros(a.shape[2]),np.zeros(a.shape[2])
    pearson_p_a, pearson_p_b = np.zeros(a.shape[2]),np.zeros(a.shape[2])
    for time in range(a.shape[2]):
        pearson_a[time] = pearsonr(a[ind,:,time],palatability_vec)[0]
        pearson_b[time] = pearsonr(b[ind,:,time],palatability_vec)[0]
        pearson_p_a[time] = pearsonr(a[ind,:,time],palatability_vec)[1]
        pearson_p_b[time] = pearsonr(b[ind,:,time],palatability_vec)[1]
    plt.plot(pearson_a)
    plt.plot(pearson_b)
    plt.title('Pearson_rho')
    plt.axhline()
    
    fig.add_subplot(gs[0,1])
    plt.plot(pearson_p_a)
    plt.plot(pearson_p_b)
    plt.title('Pearson_p')
    
    for i in fig.axes:
        i.get_xaxis().set_visible(False)
        i.get_yaxis().set_visible(False)
    
    return fig
##########################