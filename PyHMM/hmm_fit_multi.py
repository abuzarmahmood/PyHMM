# Cast HMM run as a function and use multiprocessing to run on multiple cores
# Discrete HMMs
import numpy as np
import DiscreteHMM
from hmm_implement_discrete import hmm_fit
import multiprocessing as mp

def hmm_fit_multi(binned_spikes,num_seeds,min_states,max_states,n_cpu):
    num_states = range(min_states,max_states+1)
    for states in num_states:
        pool = mp.Pool(processes = n_cpu)
        results = [pool.apply_async(hmm_fit, args = (binned_spikes, seed, states)) for seed in range(num_seeds)]
        output = [p.get() for p in results]
        pool.close()
        pool.join()  
        
        log_probs = [output[i][1] for i in range(len(output))]
        maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
        fin_out = output[maximum_pos]
        return fin_out