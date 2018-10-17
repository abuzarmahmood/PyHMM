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
import os
import tables
os.chdir('/media/bigdata/PyHMM/PyHMM/')
import numpy as np
from hinton import hinton
from hmm_fit_funcs import *
from fake_firing import *

#%matplotlib inline
import pylab as plt
plt.ioff()
import shutil
import pickle



#  ______ _ _     _    _ __  __ __  __ 
# |  ____(_) |   | |  | |  \/  |  \/  |
# | |__   _| |_  | |__| | \  / | \  / |
# |  __| | | __| |  __  | |\/| | |\/| |
# | |    | | |_  | |  | | |  | | |  | |
# |_|    |_|\__| |_|  |_|_|  |_|_|  |_|
#
jitter_p_vec = np.arange(0,0.05,0.01)
filename_vec = ['file%i' % (name) for name in range(len(jitter_p_vec))]

for this_file in range(len(filename_vec)):

    filename = filename_vec[this_file]
    jitter_p = jitter_p_vec[this_file]

    #filename = 'file5'
    data_dir = '/media/bigdata/PyHMM/PyHMM/fake_data_tests/new_tests/' + filename
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    os.chdir(data_dir)

    nrns = 10
    trials = 15
    length = 300
    state_order = np.asarray([0,1,2]) # use one order for all trials; len = num_transitions + 1
    palatability_state = 1
    ceil_p = 0.05 # Maximum firing probability -> Make shit sparse   
    jitter_t = 10 # Jitter from mean transition times for neurons on same trial
    #jitter_p = 0.1
    min_duration = 80 # Min time of 1st transition & time b/w transitions & time of 2nd transition before end
    seed_num = 30

    data, t, p, all_p, scaling = make_fake_file(
            filename = filename, 
            nrns = nrns,
            trials = trials,
            length = length,
            state_order = state_order,
            palatability_state = palatability_state,
            ceil_p = ceil_p,
            jitter_t = jitter_t,
            jitter_p = jitter_p,
            jitter_p_type = 'abs',
            min_duration = min_duration,
            data_type = 'cat')
    

# =============================================================================
#     data, t, p, all_p, scaling = fake_cat_firing(
#             nrns = nrns,
#             trials = trials,
#             length = length,
#             state_order = state_order,
#             palatability_state = palatability_state,
#             ceil_p = ceil_p,
#             jitter_t = jitter_t,
#             jitter_p = jitter_p,
#             jitter_p_type = 'abs',
#             min_duration = min_duration)    
# =============================================================================
    
    p_plot = prob_plot(all_p)
    p_plot.savefig(filename + 'p_plot.png')
    plt.close(p_plot)

    # Combining multiple datasets
    # =============================================================================
    # data1, t1, p1, scaling1 = fake_cat_firing(
    #                                 nrns = nrns,
    #                                 trials = trials,
    #                                 length = length,
    #                                 state_order = state_order,
    #                                 palatability_state = palatability_state,
    #                                 ceil_p = ceil_p,
    #                                 jitter_t = jitter_t,
    #                                 jitter_p = jitter_p,
    #                                 min_duration = min_duration)
    # 
    # data2, t2, p2, scaling2 = fake_cat_firing(
    #                                 nrns = nrns,
    #                                 trials = trials,
    #                                 length = length,
    #                                 state_order = state_order,
    #                                 palatability_state = palatability_state,
    #                                 ceil_p = ceil_p,
    #                                 jitter_t = jitter_t,
    #                                 jitter_p = jitter_p,
    #                                 min_duration = min_duration)
    # 
    # new_data = np.empty(data1[0].shape)
    # new_data[:10,:] = data1[0][:10,:]
    # new_data[10:15,:] = data2[0][10:15,:]
    # 
    # new_p = [p1[0],p2[0]]
    # new_t = np.empty(t2[0].shape)
    # new_t[:10,:,:] = t1[0][:10,:,:]
    # new_t[10:15,:,:] = t2[0][10:15,:,:]
    # 
    # data = [new_data]
    # p = [new_p]
    # t = [new_t]
    # =============================================================================
    # =============================================================================
    # # Look at raster for fake data
    # for i in range(data[0].shape[0]):
    #     plt.figure(i)
    #     raster(data[0][i,:],t[0][i,:,:])
    # =============================================================================

    ###
    for model_num_states in range(3,6):
        for taste in [0]:#range(len(data)):
            plot_dir = 'plots/taste%i/states%i' % (taste, model_num_states)
            if os.path.exists(data_dir + '/' + plot_dir):
                shutil.rmtree(data_dir + '/' + plot_dir)
            os.makedirs(data_dir + '/' + plot_dir)

            # Variational Inference HMM
            model_VI, model_MAP = hmm_cat_var_multi(
                    data[taste],
                    seed_num,
                    model_num_states,
                    initial_conds_type = 'rand',
                    max_iter = 1500,
                    threshold = 1e-6)

            ### MAP Outputs ###
            alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_MAP.E_step()
            
            # Output to hdf5 file
            hf5 = tables.open_file(filename+'.h5',mode="r+")
            hf5.create_array('/spike_trains/dig_in_%i/expected_prob_MAP' % (taste), 'state%i' % model_num_states, expected_latent_state, createparents=True)

            
            for i in range(data[taste].shape[0]):
                plt.figure()
                raster(data[taste][i,:],t[taste][i,:,:],expected_latent_state[:,i,:])
                plt.savefig(plot_dir + '/' + '%i_map_%ist.png' % (i,model_num_states))
                plt.close(i)

            plt.figure()
            hinton(model_MAP.p_transitions)
            plt.title('log_post = %f' %model_MAP.log_posterior[-1])
            plt.suptitle('Model converged = ' + str(model_MAP.converged))
            plt.savefig(plot_dir + '/' + 'hinton_map_%ist.png' % model_num_states)
            plt.close()

            ### VI Outputs ###
            alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_VI.E_step()
            
            hf5.create_array('/spike_trains/dig_in_%i/expected_prob_VI' % (taste), 'state%i' % model_num_states, expected_latent_state, createparents=True)
            hf5.flush()
            hf5.close()
            
            # Save figures in appropriate directories
            for i in range(data[taste].shape[0]):
                fig = plt.figure()
                raster(data[taste][i,:],t[taste][i,:,:],expected_latent_state[:,i,:])
                fig.savefig(plot_dir + '/' + '%i_var_%ist.png' % (i,model_num_states))
                plt.close(fig)

            fig = plt.figure()
            hinton(model_VI.transition_counts)
            plt.title('ELBO = %f' %model_VI.ELBO[-1])
            plt.suptitle('Model converged = ' + str(model_VI.converged))
            fig.savefig(plot_dir + '/' + 'hinton_var_%ist.png' % model_num_states)
            plt.close(fig)
                    
    with open('%s_params.pkl' % filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([data, p, t], f)
    # =============================================================================
    # #           _ _                                  _      _____ _               _    
    # #     /\   | (_)                                | |    / ____| |             | |   
    # #    /  \  | |_  __ _ _ __  _ __ ___   ___ _ __ | |_  | |    | |__   ___  ___| | __
    # #   / /\ \ | | |/ _` | '_ \| '_ ` _ \ / _ \ '_ \| __| | |    | '_ \ / _ \/ __| |/ /
    # #  / ____ \| | | (_| | | | | | | | | |  __/ | | | |_  | |____| | | |  __/ (__|   < 
    # # /_/    \_\_|_|\__, |_| |_|_| |_| |_|\___|_| |_|\__|  \_____|_| |_|\___|\___|_|\_\
            # #                __/ |                                                             
    # #               |___/
    # #
    # # Make plots before and after alignment
    # # Unaligned firing
    # all_series_unaligned = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))
    # sum_firing_unaligned = (np.sum(all_series_unaligned,axis=0))
    # 
    # # Detecting states
    # # Look at when each state goes above a certain probability
    # # Slope around that point should be positive
    # 
    # # Take cumulative sum of state probability
    # # Every timepoint is a n-dim vector (n = number of states) of SLOPE of cumsum
    # # Cluster using k-means
    # 
    # for i in range(data.shape[1]):
    #     plt.figure()
    #     raster(data[:,i,:],t[i,:,:],expected_latent_state[:,i,:])
    #     plt.figure()
    #     plt.plot(np.cumsum(expected_latent_state[:,i,:],axis=1).T)
    # =============================================================================
