# Based on narendramukherjee/PyHMM
# https://github.com/narendramukherjee/PyHMM

import numpy as np
import DiscreteHMM
import variationalHMM
import multiprocessing as mp

"""
   _____      _                        _           _ 
  / ____|    | |                      (_)         | |
 | |     __ _| |_ ___  __ _  ___  _ __ _  ___ __ _| |
 | |    / _` | __/ _ \/ _` |/ _ \| '__| |/ __/ _` | |
 | |___| (_| | ||  __/ (_| | (_) | |  | | (_| (_| | |
  \_____\__,_|\__\___|\__, |\___/|_|  |_|\___\__,_|_|
                       __/ |                         
                      |___/

: Data
    : Only one neuron fires in every time bin
    : In every time bin, a spike is indicated by the index of the firing neuron
        and there is a separate category for no spiking

: The firing of every neuron is linked to every other neuron since emissions
    are mutually exclusive

"""

#######
# MAP #
#######

def hmm_cat_map(binned_spikes, seed, num_states, initial_conds_type = 'des', 
                max_iter = 1500, threshold = 1e-4):
    """
    Categorical Maximum a-posteriori HMM
    Priors are defined as pseudocounts (since these HMM are discrete)    
    Initial conditions and pseudocounts can either be 'des' (designed) or 'rand' (random)
    In all cases:
        p_emissions (random), p_start (random) are initialized in the same manner
    des:
        p_transitions: Almost diagonal matrix with some off-diagonal values
        transition_pseudocounts: Same structure as p_transitions with transitions values 
            on the order of time (multiplied by a random number)
        emission_pseudocounts: Avg firing of every neuron during 1 trial (multiplied by a random number)
    rand:
        All initial probabilties and pseudocounts are random
        
    : binned_spikes: trials x time
    : seed: To control randomization
    : num_states: Number of states for the model
    : initial_conds_type: Described above
    : max_iter: Maximum number of iterations to fit the model for
    : threshold: Change in cost at which fitting stops
    
    Returns -> model as generated in DiscreteHMM.CategoricalHMM
        
    """
    
    np.random.seed(seed)
    n_emissions = np.max(binned_spikes).astype('int') + 1 # 0-indexing
    
    model = DiscreteHMM.CategoricalHMM(
            num_states = num_states, 
            num_emissions = n_emissions, 
            max_iter = max_iter, 
            threshold =  threshold)
    
    # Desgined initial conditions
    if initial_conds_type == 'des':
        # Define probabilities
        p_transitions = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05) #(num_states X num_states)
        p_emissions = np.random.random(size=(num_states, n_emissions)) #(num_states X num_emissions)
        p_start = np.random.random(num_states) #(num_states)
        
        # Define pseudocounts
        # Pseudocounts multiplied by random number to add variance to different runs
        # Pseudocounts are on the order of numbers for a single trial
        start_pseuedocounts = np.ones(num_states)*np.random.random() #(num_states)
        transition_pseudocounts = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05)*binned_spikes.shape[1]*np.random.random() #(num_states X num_states)
        
        # Emission pseudocounts : Average count of a neuron/trial
        avg_firing = np.zeros(n_emissions)
        for i in range(avg_firing.size):
            avg_firing[i] = np.sum(binned_spikes == i) #(states X num_emissions)
        emission_pseudocounts =  np.tile(avg_firing/binned_spikes.shape[0], (num_states,1))*np.random.random() #(num_states X num_emissions)
    
    # Random initial conditions, all between 0 and 1
    elif initial_conds_type == 'rand':
        p_transitions = np.random.rand(num_states,num_states) #(num_states X num_states)
        p_emissions = np.random.random(size=(num_states, n_emissions)) #(num_states X num_emissions)
        p_start = np.random.random(num_states) #(num_states)
        start_pseuedocounts = np.random.random(num_states) #(num_states)
        transition_pseudocounts = np.random.rand(num_states,num_states) #(num_states X num_states)
        emission_pseudocounts = np.random.random(size=(num_states, n_emissions)) #(num_states X num_emissions)
    
    model.fit(
            data=binned_spikes, 
            p_transitions=p_transitions, 
            p_emissions=p_emissions, 
            p_start=p_start,
            transition_pseudocounts=transition_pseudocounts, 
            emission_pseudocounts=emission_pseudocounts, 
            start_pseudocounts=start_pseuedocounts,
            verbose = 0)
    
    return model

def hmm_cat_map_multi(binned_spikes, num_seeds, num_states, initial_conds_type, 
                      max_iter, threshold, n_cpu = mp.cpu_count()):
    """
    Runs multiple fits of the Categorial MAP HMM
    
    : params: Same as hmm_cat_map except
    : num_seeds: Used to generate sequential seeds from 1 -> num_seeds
    
    Returns -> Model with highest log-posterior probability from all seeds run
    """
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_cat_map, args = (binned_spikes, seed, 
        num_states, initial_conds_type, max_iter, threshold)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()  
    
    log_probs = [output[i].log_posterior[-1] for i in range(len(output))]
    maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    fin_out = output[maximum_pos]
    
    return fin_out

###############
# Variational #
###############                    

def hmm_cat_var(binned_spikes, seed, num_states, initial_conds_type, 
                max_iter = 1500,threshold = 1e-4):
    """
    Variational Inference Categorical HMM
    The model uses a MAP estimate as it's initial conditions
    
    : params: Same as hmm_cat_map
    """    
    initial_model = hmm_cat_map(
            binned_spikes,
            seed,num_states,
            initial_conds_type, 
            max_iter, 
            threshold)
    
    model_VI = variationalHMM.CategoricalHMM(
            num_states = num_states, 
            num_emissions = np.max(binned_spikes).astype('int') + 1, # 0-indexing, 
            max_iter = max_iter, 
            threshold = threshold)
    
    # Since the initial model only outputs probabilities, the pseudocounts are scaled
    # by the size of the data to bring them onto the same order as those used in the MAP HMM.
    model_VI.fit(
            data = binned_spikes, 
            transition_hyperprior = 1, 
            emission_hyperprior = 1, 
            start_hyperprior = 1, 
            initial_transition_counts = binned_spikes.shape[1]*initial_model.p_transitions, 
            initial_emission_counts = binned_spikes.shape[1]*initial_model.p_emissions,
            initial_start_counts = num_states*initial_model.p_start, 
            update_hyperpriors = True, 
            update_hyperpriors_iter = 10,
            verbose = 0)
    
    return model_VI, initial_model

def hmm_cat_var_multi(binned_spikes, num_seeds,num_states, initial_conds_type, 
                      max_iter, threshold, n_cpu = mp.cpu_count()):
    """
    Runs multiple fits of the Categorial Variational Inference HMM
    
    : params: Same as hmm_cat_map_multi
    
    Returns -> VI model with highest ELBO and MAP model with highest
        log-posterior from all seeds run
    """
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_cat_var, args = (binned_spikes, seed, 
        num_states,initial_conds_type, max_iter, threshold)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()  
    
    # Best VI model
    elbo = [output[i][0].ELBO[-1] for i in range(len(output))]
    maximum_VI_pos = np.where(elbo == np.max(elbo))[0][0]
    fin_VI_out = output[maximum_VI_pos]
    
    # Best MAP model
    log_probs = [output[i][1].log_posterior[-1] for i in range(len(output))]
    maximum_MAP_pos = np.where(log_probs == np.max(log_probs))[0][0]
    fin_MAP_out = output[maximum_MAP_pos]    
    
    return fin_VI_out[0], fin_MAP_out[1] 

"""
  ____                              _ _ _ 
 |  _ \                            | | (_)
 | |_) | ___ _ __ _ __   ___  _   _| | |_ 
 |  _ < / _ \ '__| '_ \ / _ \| | | | | | |
 | |_) |  __/ |  | | | | (_) | |_| | | | |
 |____/ \___|_|  |_| |_|\___/ \__,_|_|_|_|

: Data structure
    : Shape: neurons x trials x time

: The firing of every neuron is independent from every other neuron.
"""

#######
# MAP #
#######

def hmm_ber_map(binned_spikes,seed,num_states,max_iter = 1500,threshold = 1e-4):
    np.random.seed(seed)
    model = DiscreteHMM.IndependentBernoulliHMM(
            num_states = num_states, 
            num_emissions = binned_spikes.shape[0], 
            max_iter = max_iter, 
            threshold = threshold)

    # Define probabilities
    p_transitions = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05) #(num_states X num_states)
    p_emissions = np.random.random(size=(num_states, binned_spikes.shape[0])) #(num_states X num_emissions)
    p_start = np.random.random(num_states) #(num_states)
    
    # Define pseudocounts on the order of values in a single trial
    start_pseuedocounts = np.ones(num_states) #(num_states)
    transition_pseudocounts = np.abs(np.eye(num_states)*binned_spikes.shape[2] - np.random.rand(num_states,num_states)*binned_spikes.shape[2]*0.05) #(num_states X num_states)
    
    # Emission pseudocounts : Average count of a neuron/trial, on and off in same ratio as firing probability 
    mean_on_counts = np.sum(binned_spikes,axis = (1,2))/binned_spikes.shape[1]
    mean_off_counts = np.sum(1 - binned_spikes,axis = (1,2))/binned_spikes.shape[1]
    all_on_counts = np.tile(mean_on_counts,(num_states,1)) #(states X num_emissions)
    all_off_counts = np.tile(mean_off_counts, (num_states,1)) #(states X num_emissions)
    emission_pseudocounts =  np.dstack((all_on_counts,all_off_counts)) #(num_states X num_emissions X 2)
    
    model.fit(
            data=binned_spikes, 
            p_transitions=p_transitions, 
            p_emissions=p_emissions, 
            p_start=p_start, 
            transition_pseudocounts=transition_pseudocounts*np.random.random(), 
            emission_pseudocounts=emission_pseudocounts*np.random.random(), 
            start_pseudocounts=start_pseuedocounts*np.random.random(), 
            verbose = 0)
    
    return model


def hmm_ber_map_multi(binned_spikes,num_seeds,num_states,n_cpu = mp.cpu_count()):
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_ber_map, args = (binned_spikes, seed, num_states)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()  
    
    log_probs = [output[i].log_posterior[-1] for i in range(len(output))]
    maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    fin_out = output[maximum_pos]
    return fin_out
   
###############
# Variational #
###############

def hmm_ber_var(binned_spikes,seed,num_states, max_iter = 1500, threshold = 1e-4):
    
    initial_model = hmm_ber_map(binned_spikes,seed,num_states)
    
    model_VI = variationalHMM.IndependentBernoulliHMM(
            num_states = num_states, 
            num_emissions = binned_spikes.shape[0], 
            max_iter = max_iter, 
            threshold = threshold)

    # Define probabilities and pseudocounts
    p_emissions_bernoulli = np.zeros((model_VI.num_states, binned_spikes.shape[0], 2))
    p_emissions_bernoulli[:, :, 0] = initial_model.p_emissions
    p_emissions_bernoulli[:, :, 1] = 1.0 - initial_model.p_emissions
    
    model_VI.fit(
            data = binned_spikes, 
            transition_hyperprior = 1, 
            emission_hyperprior = 1, 
            start_hyperprior = 1, 
            initial_transition_counts = binned_spikes.shape[2]*initial_model.p_transitions, 
            initial_emission_counts = binned_spikes.shape[2]*p_emissions_bernoulli,
            initial_start_counts = binned_spikes.shape[0]*initial_model.p_start, 
            update_hyperpriors = True, 
            update_hyperpriors_iter = 10,
            verbose = False)
    
    return model_VI, initial_model 

def hmm_ber_var_multi(binned_spikes,num_seeds,num_states,n_cpu = mp.cpu_count()):
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_ber_var, args = (binned_spikes, seed, num_states)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()  
    
    # Best VI model
    elbo = [output[i][0].ELBO[-1] for i in range(len(output))]
    maximum_VI_pos = np.where(elbo == np.max(elbo))[0][0]
    fin_VI_out = output[maximum_VI_pos]
    
    # Best MAP model
    log_probs = [output[i][1].log_posterior[-1] for i in range(len(output))]
    maximum_MAP_pos = np.where(log_probs == np.max(log_probs))[0][0]
    fin_MAP_out = output[maximum_MAP_pos]    
    
    return fin_VI_out[0], fin_MAP_out[1]                                                    
