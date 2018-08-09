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
                    

def hmm_map_fit(binned_spikes,seed,num_states):
    np.random.seed(seed)
    model = DiscreteHMM.IndependentBernoulliHMM(num_states = num_states, num_emissions = binned_spikes.shape[0], 
    max_iter = 1000, threshold = 1e-9)

    # Define probabilities and pseudocounts
    p_transitions = np.abs(np.eye(num_states) - np.random.rand(num_states,num_states)*0.05) #(num_states X num_states)
    p_emissions = np.random.random(size=(num_states, binned_spikes.shape[0])) #(num_states X num_emissions)
    p_start = np.random.random(num_states) #(num_states)
    start_pseuedocounts = np.ones(num_states) #(num_states)
    transition_pseudocounts = np.abs(np.eye(num_states)*1500 - np.random.rand(num_states,num_states)*1500*0.05) #(num_states X num_states)
    
    # Emission pseudocounts : Average count of a neuron/trial, on and off in same ratio as firing probability 
    avg_firing_p = np.tile(np.mean(binned_spikes,axis = (1,2)),(num_states,1)) #(states X num_emissions)
    avg_off_p = np.tile(np.ones((1,binned_spikes.shape[0])) - np.mean(binned_spikes,axis = (1,2)), (num_states,1)) #(states X num_emissions)
    emission_pseudocounts =  np.dstack((avg_firing_p,avg_off_p))*150 #(num_states X num_emissions X 2)
    
    model.fit(data=binned_spikes, p_transitions=p_transitions, p_emissions=p_emissions, 
    p_start=p_start, transition_pseudocounts=transition_pseudocounts, emission_pseudocounts=emission_pseudocounts, 
    start_pseudocounts=start_pseuedocounts, verbose = 0)
    
    return model


def hmm_map_fit_multi(binned_spikes,num_seeds,num_states,n_cpu = mp.cpu_count()):
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_map_fit, args = (binned_spikes, seed, num_states)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()  
    
    log_probs = [output[i].log_likelihood[-1] for i in range(len(output))]
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
# intial_model = output model from hmm_map_fit
def hmm_var_fit(binned_spikes,initial_model,seed,num_states):
    
    np.random.seed(seed)
    model_VI = variationalHMM.IndependentBernoulliHMM(num_states = num_states, num_emissions = binned_spikes.shape[0], 
    max_iter = 1000, threshold = 1e-9)

    # Define probabilities and pseudocounts
    p_emissions_bernoulli = np.zeros((model_VI.num_states, binned_spikes.shape[0], 2))
    p_emissions_bernoulli[:, :, 0] = initial_model.p_emissions
    p_emissions_bernoulli[:, :, 1] = 1.0 - initial_model.p_emissions
    
    model_VI.fit(data = binned_spikes, transition_hyperprior = 1, emission_hyperprior = 1, start_hyperprior = 1, 
          initial_transition_counts = 30*initial_model.p_transitions, initial_emission_counts = 30*p_emissions_bernoulli,
            initial_start_counts = 2*initial_model.p_start, update_hyperpriors = True, update_hyperpriors_iter = 1,
            verbose = False)

    return model_VI

def hmm_var_fit_multi(binned_spikes,initial_model,num_seeds,num_states,n_cpu = mp.cpu_count()):
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_var_fit, args = (binned_spikes, initial_model, seed, num_states)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()  
    
    elbo = [output[i].ELBO[-1] for i in range(len(output))]
    maximum_pos = np.where(elbo == np.max(elbo))[0][0]
    fin_out = output[maximum_pos]
    return fin_out                                                   
