# Cast HMM run as a function and use multiprocessing to run on multiple cores
# Discrete HMMs
import numpy as np
import DiscreteHMM
import multiprocessing as mp

def hmm_fit(binned_spikes,seed,num_states):
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
    
    alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model.E_step()
    converged_val = model.converged
    fin_log_lik = model.log_likelihood[-1]
    p_transitions = model.p_transitions
    
    del model  
    return fin_log_lik, expected_latent_state, p_transitions, converged_val

def hmm_fit_multi(binned_spikes,num_seeds,num_states,n_cpu):
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(hmm_fit, args = (binned_spikes, seed, num_states)) for seed in range(num_seeds)]
    output = [p.get() for p in results]
    #pool.close()
    #pool.join()  
    
    log_probs = [output[i][1] for i in range(len(output))]
    maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    fin_out = output[maximum_pos]
    return fin_out
