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

# Expected latent state = state x trials x time
# Find dominant state during range

def align_trials(data,expected_latent_state,transition_points,time_range,probability_tresh,window_radius):
    dom_state = np.argmax(np.sum(np.sum(expected_latent_state,axis=1)[:,time_range],axis = 1))
    for trial in range(expected_latent_state.shape[1]):
        try:
            this_transition = np.where(expected_latent_state[dom_state,trial,:] > probability_thresh)[0][0]
            if ((this_transition + window_radius) > expected_latent_state.shape[2]) or ((this_transition - window_radius) < 0):
                this_transition = -1
            transition_points.append(this_transition)
        except:
            transition_points.append(-1)
            
    aligned_data = np.zeros((data.shape[0],data.shape[1],window_radius*2))
    for trial in range(aligned_data.shape[1]):
        if transition_points[trial] > 0:
            aligned_data[:,trial,:] = data[:,trial,transition_points[trial]-window_radius:transition_points[trial]+window_radius]
            
    return aligned_data