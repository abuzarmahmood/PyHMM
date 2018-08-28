#           _ _               _______   _       _     
#     /\   | (_)             |__   __| (_)     | |    
#    /  \  | |_  __ _ _ __      | |_ __ _  __ _| |___ 
#   / /\ \ | | |/ _` | '_ \     | | '__| |/ _` | / __|
#  / ____ \| | | (_| | | | |    | | |  | | (_| | \__ \
# /_/    \_\_|_|\__, |_| |_|    |_|_|  |_|\__,_|_|___/
#                __/ |                                
#               |___/  

import numpy as np

# Detecting states
# - Use time range to find dominant state during a period
# - Use threshold to find when that state becomes significant during a trial (regardless of given time range)
# - Snip firing of neuron around that trial for a given window length

# Find dominant state during range
# First emergence of dominant state (check slope to make sure)
    # This ensures that dominant state transition is not picked at 0

def calc_trans(expected_latent_state,time_range,probability_thresh, start_t, bin_size):
    """
    Calculates transition points in relation to origin spikes sequence
    : expected_latent_state: states x trials x time
    : time_range: Range of indices for calculating dominant state
    : probability_tresh: Threshold to mark state transition
    : start_t: Start time for data USED TO FIT HMM
    : bin_size: Bin size for data USED TO FIT HMM
    Return -> Times at which transitions occur for every trial
    """
    
    transition_points = []
    dom_state = np.argmax(np.sum(np.sum(expected_latent_state,axis=1)[:,time_range],axis = 1))
    
    for trial in range(expected_latent_state.shape[1]):
        this_transition = np.where(expected_latent_state[dom_state,trial,:] > probability_thresh)[0][0]
        if this_transition > 0:    
            transition_points.append(this_transition)
        else:
            transition_points.append(-1)
    
    transition_times = np.zeros(len(transitions))        
    for i in range(transition_times.size):
        if transition_points[i] > 0:
            transition_times[i] = (transition_points[i]*bin_size) + start_t
        else:
            transition_times[i] = -1
            
    return transition_times

def ber_align_trials(data, transition_times, start_t_data, bin_size_data, window_size):
    """
    : data: neurons X trials X time
        : Original spike timing data (without any snipping)
    : transition_times: absolute time for transitions as calculates by calc_trans
    : start_t_data: Start time of data FED TO FUNCTION
    : bin_size_data: Bin size of data FED TO FUNCTION
    : window_size: Window DIAMETER around transition points (in milliseconds)
    Return -> Realigned data with same dims as input data (but cropped in time)
    """
    min_transition_time = min(transition_times[transition_times > 0])
    if  min_transition_time - window_size/2 - start_t_data < 0 :
        raise ValueError('Data start time too late to handle window size\nCan support window of %i' % int((min_transition_time - start_t_data)/2))
    
    window_bins = int(window_size/bin_size_data)
    aligned_data = np.zeros((data.shape[0],data.shape[1],window_bins))
    for trial in range(aligned_data.shape[1]):
        if transition_times[trial] > 0:
            # Convert transition time to bins in given data
            transition = int((transition_times[trial] - start_t_data) / bin_size_data)
            aligned_data[:,trial,:] = data[:,trial,transition - int(window_bins/2) : transition + int(window_bins/2)]
    
    mean_transition_bin = int(np.mean(transition_times)/bin_size_data)
    unaligned_data = data[:,:,(mean_transition_bin - int(window_bins/2)):(mean_transition_bin + int(window_bins/2))]
    
    return aligned_data, unaligned_data

def cat_align_trials(data, transition_times, start_t_data, bin_size_data, window_size):
    """
    Convert categorical data to spike trains and feed into ber_align_trials
    : data: trials x time
    """
    
    spikes = np.zeros((np.max(data).astype('int') + 1,data.shape[0] ,data.shape[1])) # categorical data has no firing too
    for trial in range(spikes.shape[1]):
        for time in range(spikes.shape[2]):
            spikes[data[trial,time].astype('int'),trial,time] = 1
     
    # Remove no-spiking category from data
    spikes = spikes[1::,:,:]       
    aligned_data = ber_align_trials(spikes,expected_latent_state,time_range,probability_thresh,window_radius)
            
    return aligned_data

def calc_firing_rate(data, window_size, step_size):
    # Takes in either bernoulli data (neurons x trials x time)
    # Returns an array (neurons X trials X time) reduced by binning in time
    
    firing = np.zeros((data.shape[0],data.shape[1],int((data.shape[2] - window_size)/step_size + 1)))    
    for nrn in range(firing.shape[0]):    
        for trial in range(firing.shape[1]):
            for time in range(firing.shape[2]):
                firing[nrn,trial,time] = np.mean(data[nrn,trial,step_size*time:step_size*time + window_size])
    return firing

def mean_slope(data,window_radius):
    # Calculates slope for all points in data in windows dictated by window radius
    # Output is vector of slopes same size as data
        # Points on edges with incomplete windows are returned with NaN slopes
    slopes = np.zeros(data.shape)
    
    for time in range(data.size):
        if time < window_radius or time > data.size - window_radius:
            slopes[time] = np.nan
        else:
            temp_data = data[range(time - window_radius,time + window_radius)]
            slopes[time] = np.polyfit(range(temp_data.size), temp_data, deg=1)[0]
    slopes[~np.isnan(slopes)] = slopes[~np.isnan(slopes)] / np.max(np.abs(slopes[~np.isnan(slopes)]))
            
    return slopes
            