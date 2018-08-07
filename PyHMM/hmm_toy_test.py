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

#%matplotlib inline
import pylab as plt


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
data = np.zeros((10, 20, 300)) # neurons x trials x time

ceil_p = 0.1 # Maximum firing probability -> Make shit sparse
p = np.random.rand(2, data.shape[0])*ceil_p # states x neuron
# Don't need to normalize across neurons
# =============================================================================
# for state in range(p.shape[0]):
#     p[state,:] = p[state,:]/np.sum(p,axis = 1)[state]
# =============================================================================
    
jitter_t = 20 # Jitter between transition times for neurons on same trial
min_duration = 70 # Min time of 1st transition & time b/w transitions & time of 2nd transition before end
t = np.zeros((data.shape[1], 2)) # trials x num of transitions (2) 
for trial in range(t.shape[0]):
    while ((t[trial,0] < min_duration) or (t[trial,1] < t[trial,0] + min_duration) or (t[trial,1] + min_duration > data.shape[2] ) ):
        t[trial,:] = (np.random.rand(1,t.shape[1]) * data.shape[2])
t = np.repeat(t[:, :, np.newaxis], data.shape[0], axis=2) # trials x num of transitions (2) x neurons
t = t + np.random.random(t.shape)*jitter_t
t = t.astype('int')
#t = np.reshape(t,(t.shape[0],t.shape[2],t.shape[1])) # trials x neurons x transitions

state_order = np.asarray([0,1,0]) # use one order for all trials; len = num_transitions + 1

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

for i in range(10):#data.shape[1]):
    plt.figure(i)
    raster(data[:,i,:],t[i,:,:])
#  ______ _ _     _    _ __  __ __  __ 
# |  ____(_) |   | |  | |  \/  |  \/  |
# | |__   _| |_  | |__| | \  / | \  / |
# |  __| | | __| |  __  | |\/| | |\/| |
# | |    | | |_  | |  | | |  | | |  | |
# |_|    |_|\__| |_|  |_|_|  |_|_|  |_|
#        

# MAP Estimate

# from hmm_fit import hmm_fit
#from hmm_fit_multi import hmm_fit_multi

from hmm_fit_funcs import *

model_MAP = hmm_map_fit_multi(data,30,7)


#expected_latent_state = model_MAP[1]
alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_MAP.E_step()

####### Check
for i in range(10):#data.shape[1]):
    plt.figure(i)
    raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])

plt.figure()
hinton(model_MAP.p_transitions.T)

# Variational Inference HMM
model_VI = hmm_var_fit_multi(data, model_MAP, 30, 7)
alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_VI.E_step()

####### Check
for i in range(10):#data.shape[1]):
    plt.figure(i)
    raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])
plt.figure()
hinton(model_VI.transition_counts)