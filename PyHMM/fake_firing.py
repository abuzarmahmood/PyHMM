#  ______    _          _____        _        
# |  ____|  | |        |  __ \      | |       
# | |__ __ _| | _____  | |  | | __ _| |_ __ _ 
# |  __/ _` | |/ / _ \ | |  | |/ _` | __/ _` |
# | | | (_| |   <  __/ | |__| | (_| | || (_| |
# |_|  \__,_|_|\_\___| |_____/ \__,_|\__\__,_|
#

# Import stuff
import numpy as np
import pylab as plt

                                             
# Bernoulli trials with arbitrary numbers of states and transitions (upto constraints)
# Firing probability random for each neuron (upto a ceiling)
# Transitions random with some min durations
# Jitter between individual neuron state transtiions
    
def fake_firing(nrns,trials,length,num_states,state_order,ceil_p,jitter_t,min_duration):
    # nrns = number of neurons (emissions)
    # trials = number of trials
    # length = number of bins for data
    # state_order = LIST: fixed order for states
    # ceil_p = maximum firing probability
    # jitter_t = max amount of jitter between neurons for a state transition
    # min_duration =    time between start and first transition
    #                   time between final state and end
    #                   time between intermediate state transitions
    
    
    # Returns data array, transition times, emission probabilities
    data = np.zeros((nrns, trials, length)) # neurons x trials x time
    
    # Emission probabilities of neurons for every state
    p = np.random.rand(num_states, data.shape[0])*ceil_p # states x neuron
    
    # Transition times for every transition over every trial
    t = np.zeros((data.shape[1], len(state_order)-1)) # trials x num of transitions (2) 
    for trial in range(t.shape[0]):
        first_trans, last_trans, middle_trans = [1,1,1]
        while (first_trans or last_trans or middle_trans):
            t[trial,:] = (np.random.rand(1,t.shape[1]) * data.shape[2])
            
            first_trans = (t[trial,0] < min_duration) # Make sure first transition is after min_duration
            last_trans = (t[trial,-1] + min_duration > data.shape[2]) # Make sure last transition is min_duration before the end
            middle_trans = np.sum(t[trial,1:] - t[trial,0:-1] < min_duration)  # Make sure there is a distance of min_duration between all intermediate transitions
       
        print(trial)
    
    t = np.repeat(t[:, :, np.newaxis], data.shape[0], axis=2) # trials x num of transitions x neurons
    t = t + np.random.uniform(-1,1,t.shape)*jitter_t # Add jitter to individual neuron transitions
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
    return data, t, p


# Raster plot
def raster(data,trans_times=None,expected_latent_state=None):
    # Take two 2D arrays: 
        # data : neurons x time
        # trans_times : num_transition x neurons
        # expected_latent_state: states x time
    # Red lines indicate mean transition times
    # Yellow ticks indicate individual neuron transition times
    
    # Plot state probability
    if expected_latent_state is not None:
        plt.plot(expected_latent_state.T*data.shape[0])
        
    # Plot individual neuron transition times
    if trans_times is not None:
        for unit in range(data.shape[0]):
            for transition in range(trans_times.shape[0]):
                plt.vlines(trans_times[transition,unit], unit, unit+0.5, linewidth = 3, color = 'y')
    
    # Plot spikes         
    for unit in range(data.shape[0]):
        for time in range(data.shape[1]):
            if data[unit, time] > 0:
                plt.vlines(time, unit, unit + 0.5, linewidth = 0.5)
    
    # Plot mean transition times         
    if trans_times is not None:
        mean_trans = np.mean(trans_times, axis = 1)
        for transition in range(len(mean_trans)):
            plt.vlines(mean_trans[transition], 0, data.shape[0],colors = 'r', linewidth = 1)
            
    plt.xlabel('Time post stimulus (ms)')
    plt.ylabel('Neuron')