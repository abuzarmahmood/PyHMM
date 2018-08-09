#  _    _ __  __ __  __   _______            ______                           _      
# | |  | |  \/  |  \/  | |__   __|          |  ____|                         | |     
# | |__| | \  / | \  / |    | | ___  _   _  | |__  __  ____ _ _ __ ___  _ __ | | ___ 
# |  __  | |\/| | |\/| |    | |/ _ \| | | | |  __| \ \/ / _` | '_ ` _ \| '_ \| |/ _ \
# | |  | | |  | | |  | |    | | (_) | |_| | | |____ >  < (_| | | | | | | |_) | |  __/
# |_|  |_|_|  |_|_|  |_|    |_|\___/ \__, | |______/_/\_\__,_|_| |_| |_| .__/|_|\___|
#                                     __/ |                            | |           
#                                    |___/                             |_|
#
## From 
## https://github.com/narendramukherjee/PyHMM
## https://github.com/narendramukherjee/PyHMM/blob/master/PyHMM/examples_variational_discrete.ipynb

## Independent Bernoulli Examples

# Import stuff
import numpy as np
from hinton import hinton
from hmm_fit_funcs import *
from scipy.interpolate import spline

#%matplotlib inline
import pylab as plt
import pickle
import os


#  ______    _          _____        _        
# |  ____|  | |        |  __ \      | |       
# | |__ __ _| | _____  | |  | | __ _| |_ __ _ 
# |  __/ _` | |/ / _ \ | |  | |/ _` | __/ _` |
# | | | (_| |   <  __/ | |__| | (_| | || (_| |
# |_|  \__,_|_|\_\___| |_____/ \__,_|\__\__,_|
#
                                             
# Bernoulli trials with 2 states
# Transitions random with some min durations
# Randomize state transition times

### Set neuron and trial numbers using data dimensions ###
data = np.zeros((10, 15, 300)) # neurons x trials x time
num_states = 4
state_order = np.asarray([0,1,2,3]) # use one order for all trials; len = num_transitions + 1

ceil_p = 0.1 # Maximum firing probability -> Make shit sparse   
jitter_t = 10 # Jitter between transition times for neurons on same trial
min_duration = 30 # Min time of 1st transition & time b/w transitions & time of 2nd transition before end

p = np.random.rand(num_states, data.shape[0])*ceil_p # states x neuron

t = np.zeros((data.shape[1], len(state_order)-1)) # trials x num of transitions (2) 
for trial in range(t.shape[0]):
    #count = 0
    first_trans, last_trans, middle_trans = [1,1,1]
    while (first_trans or last_trans or middle_trans):
        t[trial,:] = (np.random.rand(1,t.shape[1]) * data.shape[2])
        
        first_trans = (t[trial,0] < min_duration) # Make sure first transition is after min_duration
        last_trans = (t[trial,-1] + min_duration > data.shape[2]) # Make sure last transition is min_duration before the end
        middle_trans = np.sum(t[trial,1:] - t[trial,0:-1] < min_duration)  # Make sure there is a distance of min_duration between all intermediate transitions
   
        #count +=1
    print(trial)

t = np.repeat(t[:, :, np.newaxis], data.shape[0], axis=2) # trials x num of transitions (2) x neurons
t = t + np.random.random(t.shape)*jitter_t
t = t.astype('int')

# For every trial, for every neuron, walk through time
# If time has passed a transition, update transition count and use transitions count
# to index from the appropriate state in state order
for trial in range(data.shape[1]):
    for neuron in range(data.shape[0]):
        trans_count = 0 # To keep track of transitions
        for time in range(data.shape[2]):
            try:
                if time < t[trial,trans_count, neuron]:
                    data[neuron, trial, time] = np.random.binomial(1, p[state_order[trans_count],neuron])
                else:
                    trans_count += 1
                    data[neuron, trial, time] = np.random.binomial(1, p[state_order[trans_count],neuron])
            except: # Lazy programming -_-
                if trans_count >= t.shape[1]:
                    data[neuron, trial, time] = np.random.binomial(1, p[state_order[trans_count],neuron])
    
# Raster plot
def raster(data,trans_times,expected_latent_state=None):
    # Take two 2D arrays: 
        # data : neurons x time
        # trans_times : num_transition x neurons
        # expected_latent_state: states x time
    if expected_latent_state is not None:
        plt.plot(expected_latent_state.T*data.shape[0])
    for unit in range(data.shape[0]):
        for time in range(data.shape[1]):
            if data[unit, time] > 0:
                plt.vlines(time, unit, unit + 0.5, linewidth = 0.5)
    mean_trans = np.mean(trans_times, axis = 1)
    for transition in range(len(mean_trans)):
        plt.vlines(mean_trans[transition], 0, data.shape[0],colors = 'r', linewidth = 1)
    plt.xlabel('Time post stimulus (ms)')
    plt.ylabel('Neuron')

# Look at raster for fake data
for i in range(data.shape[1]):
    plt.figure(i)
    raster(data[:,i,:],t[i,:,:])

#  ______ _ _     _    _ __  __ __  __ 
# |  ____(_) |   | |  | |  \/  |  \/  |
# | |__   _| |_  | |__| | \  / | \  / |
# |  __| | | __| |  __  | |\/| | |\/| |
# | |    | | |_  | |  | | |  | | |  | |
# |_|    |_|\__| |_|  |_|_|  |_|_|  |_|
#        

# MAP fit with correct number of states in model
# =============================================================================
# model_MAP_c = hmm_map_fit_multi(data,30,2)
# alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_MAP_c.E_step()
# 
# for i in range(data.shape[1]):
#     plt.figure(i)
#     raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])
#     plt.savefig('%i_MAP_2st.png' % i)
#     plt.close(i)
# 
# plt.figure()
# hinton(model_MAP_c.p_transitions.T)
# plt.savefig('hinton_MAP_2st.png')
# plt.close()
# =============================================================================

for state_num in range(7,8):
    folder_name = 'var_state_comparison/%i_states' % state_num
    os.mkdir(folder_name)
    # MAP fit with excess states in model
    model_MAP = hmm_map_fit_multi(data,30,state_num)
    alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_MAP.E_step()
    
# =============================================================================
#     for i in range(data.shape[1]):
#         plt.figure(i)
#         raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])
#         plt.savefig('%i_MAP_7st.png' % i)
#         plt.close(i)
#         
#     plt.figure()
#     hinton(model_MAP.p_transitions.T)
#     
#     plt.savefig('hinton_MAP_7st.png')
#     plt.close()
# =============================================================================
    
    # Variational Inference HMM fit with excess states in model
    model_VI = hmm_var_fit_multi(data,model_MAP,30,state_num)
    alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_VI.E_step()
    
    for i in range(data.shape[1]):
        plt.figure()
        raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])
        plt.savefig(folder_name + '/' + '%i_var_%ist.png' % (i,state_num))
        plt.close(i)
    
    plt.figure()
    hinton(model_VI.transition_counts)
    plt.title('ELBO = %f' %model_VI.ELBO[-1])
    plt.savefig(folder_name + '/' + 'hinton_var_%ist.png' % state_num)
    plt.close()


#           _ _                                  _      _____ _               _    
#     /\   | (_)                                | |    / ____| |             | |   
#    /  \  | |_  __ _ _ __  _ __ ___   ___ _ __ | |_  | |    | |__   ___  ___| | __
#   / /\ \ | | |/ _` | '_ \| '_ ` _ \ / _ \ '_ \| __| | |    | '_ \ / _ \/ __| |/ /
#  / ____ \| | | (_| | | | | | | | | |  __/ | | | |_  | |____| | | |  __/ (__|   < 
# /_/    \_\_|_|\__, |_| |_|_| |_| |_|\___|_| |_|\__|  \_____|_| |_|\___|\___|_|\_\
#                __/ |                                                             
#               |___/
#
# Make plots before and after alignment
# Unaligned firing
all_series_unaligned = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))
sum_firing_unaligned = (np.sum(all_series_unaligned,axis=0))

# Detecting states
# Look at when each state goes above a certain probability
# Slope around that point should be positive

# Take cumulative sum of state probability
# Every timepoint is a n-dim vector (n = number of states) of SLOPE of cumsum
# Cluster using k-means

for i in range(data.shape[1]):
    plt.figure()
    raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])
    plt.figure()
    plt.plot(np.cumsum(expected_latent_state[:,i,:],axis=1).T)