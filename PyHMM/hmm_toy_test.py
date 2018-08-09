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

# Import stuff
import numpy as np
from hinton import hinton
from hmm_fit_funcs import *
from fake_firing import *
from scipy.interpolate import spline

#%matplotlib inline
import pylab as plt
import pickle
import os

#  ______ _ _     _    _ __  __ __  __ 
# |  ____(_) |   | |  | |  \/  |  \/  |
# | |__   _| |_  | |__| | \  / | \  / |
# |  __| | | __| |  __  | |\/| | |\/| |
# | |    | | |_  | |  | | |  | | |  | |
# |_|    |_|\__| |_|  |_|_|  |_|_|  |_|
#        

nrns = 10
trials = 15
length = 300
data_num_states = 3
state_order = np.asarray([0,1,0,2]) # use one order for all trials; len = num_transitions + 1

ceil_p = 0.1 # Maximum firing probability -> Make shit sparse   
jitter_t = 30 # Jitter from mean transition times for neurons on same trial
min_duration = 50 # Min time of 1st transition & time b/w transitions & time of 2nd transition before end

out = fake_firing(nrns=nrns,trials=trials,length=length,num_states=data_num_states, \
               state_order=state_order,ceil_p=ceil_p,jitter_t=jitter_t,min_duration=min_duration)
data = out[0]
t = out[1]
p = out[2]

# Look at raster for fake data
for i in range(data.shape[1]):
    plt.figure(i)
    raster(data[:,i,:],t[i,:,:])
    
###
for model_num_states in range(2,8):
    folder_name = 'var_state_comparison/2_state_data/%i_states' % model_num_state
    os.makedirs(folder_name)
    
    # MAP HMM 
    model_MAP = hmm_map_fit_multi(data,30,model_num_states)

    # Variational Inference HMM
    model_VI = hmm_var_fit_multi(data,model_MAP,30,model_num_states)
    alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_VI.E_step()
    
    for i in range(data.shape[1]):
        plt.figure()
        raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])
        plt.savefig(folder_name + '/' + '%i_var_%ist.png' % (i,model_num_states))
        plt.close(i)
    
    plt.figure()
    hinton(model_VI.transition_counts)
    plt.title('ELBO = %f' %model_VI.ELBO[-1])
    plt.savefig(folder_name + '/' + 'hinton_var_%ist.png' % model_num_states)
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