import numpy as np
import DiscreteHMM
import variationalHMM
import multiprocessing as mp

#  __  __          _____  
# |  \/  |   /\   |  __ \ 
# | \  / |  /  \  | |__) |
# | |\/| | / /\ \ |  ___/ 
# | |  | |/ ____ \| |     
# |_|  |_/_/    \_\_|     
#

###############
# CATEGORICAL #
###############

def hmm_cat_map(binned_spikes,seed,num_states):
    
    np.random.seed(seed)
    n_emissions = np.unique(binned_spikes).size
    
    model = DiscreteHMM.CategoricalHMM(
            num_states = num_states, 
            num_emissions = n_emissions, 
            max_iter = 2000, 
            threshold = 1e-6)
    
    # Desgined initial conditions
# =============================================================================
#     # Define probabilities
#     p_transitions = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05) #(num_states X num_states)
#     p_emissions = np.random.random(size=(num_states, n_emissions)) #(num_states X num_emissions)
#     p_start = np.random.random(num_states) #(num_states)
#     
#     # Define pseudocounts
#     # Pseudocounts multiplied by random number to add variance to different runs
#     # Pseudocounts are on the order of numbers for a single trial
#     start_pseuedocounts = np.ones(num_states) #(num_states)
#     transition_pseudocounts = np.abs(np.eye(num_states)*binned_spikes.shape[1] - np.random.rand(num_states,num_states)*binned_spikes.shape[1]*0.05) #(num_states X num_states)
#     
#     # Emission pseudocounts : Average count of a neuron/trial
#     avg_firing = np.zeros(n_emissions)
#     for i in range(avg_firing.size):
#         avg_firing[i] = np.sum(binned_spikes == i) #(states X num_emissions)
#     emission_pseudocounts =  np.tile(avg_firing/binned_spikes.shape[0], (num_states,1)) #(num_states X num_emissions)
#     
# =============================================================================
    
    # Random initial conditions
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
# =============================================================================
#             transition_pseudocounts=transition_pseudocounts*np.random.random(), 
#             emission_pseudocounts=emission_pseudocounts*np.random.random(), 
#             start_pseudocounts=start_pseuedocounts*np.random.random(), 
# =============================================================================
            verbose = 0)
    
    return model

def hmm_cat_map_multi(binned_spikes,num_seeds,num_states,n_cpu = mp.cpu_count()):
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_cat_map, args = (binned_spikes, seed, num_states)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()  
    
    log_probs = [output[i].log_posterior[-1] for i in range(len(output))]
    maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    fin_out = output[maximum_pos]
    return fin_out

#############
# BERNOULLI #
#############                    

def hmm_ber_map(binned_spikes,seed,num_states):
    np.random.seed(seed)
    model = DiscreteHMM.IndependentBernoulliHMM(
            num_states = num_states, 
            num_emissions = binned_spikes.shape[0], 
            max_iter = 2000, 
            threshold = 1e-6)

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


# __      __        _       _   _                   _ 
# \ \    / /       (_)     | | (_)                 | |
#  \ \  / /_ _ _ __ _  __ _| |_ _  ___  _ __   __ _| |
#   \ \/ / _` | '__| |/ _` | __| |/ _ \| '_ \ / _` | |
#    \  / (_| | |  | | (_| | |_| | (_) | | | | (_| | |
#     \/ \__,_|_|  |_|\__,_|\__|_|\___/|_| |_|\__,_|_|
#



# binned_trial = neurons x trials x time

###############
# CATEGORICAL #
###############

def hmm_cat_var(binned_spikes,seed,num_states):
    
    initial_model = hmm_cat_map(binned_spikes,seed,num_states)
    
    model_VI = variationalHMM.CategoricalHMM(
            num_states = num_states, 
            num_emissions = np.unique(binned_spikes).size, 
            max_iter = 2000, 
            threshold = 1e-6)
  
    model_VI.fit(
            data = binned_spikes, 
            transition_hyperprior = 1, 
            emission_hyperprior = 1, 
            start_hyperprior = 1, 
            initial_transition_counts = binned_spikes.shape[1]*initial_model.p_transitions, 
            initial_emission_counts = binned_spikes.shape[1]*initial_model.p_emissions,
            initial_start_counts = np.unique(binned_spikes).size*initial_model.p_start, 
            update_hyperpriors = True, 
            update_hyperpriors_iter = 10,
            verbose = 0)
    
    return model_VI, initial_model # This way it can evaluate both best VI model and best MAP
                                    # model in the same run

def hmm_cat_var_multi(binned_spikes,num_seeds,num_states,n_cpu = mp.cpu_count()):
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_cat_var, args = (binned_spikes, seed, num_states)) for seed in range(num_seeds)]
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
   
#############
# BERNOULLI #
############# 

def hmm_ber_var(binned_spikes,seed,num_states):
    
    initial_model = hmm_ber_map(binned_spikes,seed,num_states)
    
    model_VI = variationalHMM.IndependentBernoulliHMM(
            num_states = num_states, 
            num_emissions = binned_spikes.shape[0], 
            max_iter = 2000, 
            threshold = 1e-6)

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
