#  _    _ __  __ __  __    _____                    _       _       
# | |  | |  \/  |  \/  |  / ____|                  | |     | |      
# | |__| | \  / | \  / | | |     ___  _ __ _ __ ___| | __ _| |_ ___ 
# |  __  | |\/| | |\/| | | |    / _ \| '__| '__/ _ \ |/ _` | __/ _ \
# | |  | | |  | | |  | | | |___| (_) | |  | | |  __/ | (_| | ||  __/
# |_|  |_|_|  |_|_|  |_|  \_____\___/|_|  |_|  \___|_|\__,_|\__\___|
#

# Ingredients list:
    # Spikes as a list of tastes
    # On and off trials as a list the same length as spikes
    # expected_latent_state as length of tastes
    # Break aligned and unaligned trials into on and off groups
    # Taste palatability rank
    

import numpy as np
from scipy.stats import spearmanr, pearsonr
from align_trials_funcs import *

class hmm_correlate:
    """
    Helper class to streamline alignment and correlation of firing data
    given HMM fits
    """
    
    def __init__(self, spikes, off_trials, on_trials, expected_latent_state,
                 palatability_ranks):
        """
        All input variables are lists the size of tastes in the same order
        : spikes: Raw data extracted from hdf5 files (taste)(neurons x trials x time)
        : on/off trials: on and off trials for every taste
        : expected_latent_state: Probabilities from HMM fit (taste)(states x trials x time)
        : palatability_ranks: Ranking of tastes from the BAT rig
        """

        self.spikes = spikes
        self.off_trials = off_trials
        self.on_trials = on_trials
        self.expected_latent_state = expected_latent_state
        self.palatability_ranks = palatability_ranks
        
        self.plot_dir = ''
        self.firing_rate = []
        self.transitions = []
        self.flags = []
        self.aligned_data = []
        self.unaligned_data = []
        self.correlations = {'off_al':None,'on_al':None,'off_un':None,'on_un':None}
    
    def calc_firing_rate(self, window_size, step_size):
        """
        Calculates firing rate by moving window smoothing of spikes
        
        : data: neurons x trials x time
        : window_size: Size of moving window IN MS
        : step_size: IN MS
        
        # Returns an array nested in a list (trials)(neurons X trials X time) reduced by binning in time
        """
        all_spikes = self.spikes
        firing_rate = []
        for data in all_spikes:
            firing = np.zeros((data.shape[0],data.shape[1],int((data.shape[2] - window_size)/step_size + 1)))    
            for nrn in range(firing.shape[0]):    
                for trial in range(firing.shape[1]):
                    for time in range(firing.shape[2]):
                        firing[nrn,trial,time] = np.mean(data[nrn,trial,step_size*time:step_size*time + window_size])
            firing_rate.append(firing)
        self.firing_rate = firing_rate
        
    def calc_transitions(self, min_prob_t, max_prob_t,
                     probability_thresh, start_t, bin_size, delta = 0.2):
        """
        Calculates transition points in relation to origin spikes sequence
        Will look for closest point to threshold probability
            If prob moves THROUGH threshold, it will find closes point AND first transition
            Otherwise it will just return closes point
            
        : expected_latent_state: states x trials x time
        : min_prob_t, max_prob_t : Min and Max time for calculating dominant state in MS
        : probability_tresh: Threshold to mark state transition
        : start_t: Start time for data USED TO FIT HMM
        : bin_size: Bin size for data USED TO FIT HMM
        : delta: Dictates range of transition
            e.g. delta = 0.1 will look for probability change from 0-0.1 to 0.9-1
            
        Return -> Times at which transitions occur for every trial
            Flags: 0 -> No transition, 1 -> Transition occured
        """
        all_transitions = []
        all_flags = []        
        all_expected_latent = self.expected_latent_state
        time_range = np.arange(int(min_prob_t/bin_size),int(max_prob_t/bin_size))
        
        for expected_latent_state in all_expected_latent:
            transition_points = []
            flags = []
            
            # Detect the dominant state for which transitions will be caluclated
            dom_state = np.argmax(np.sum(np.sum(expected_latent_state,axis=1)[:,time_range],axis = 1))
            
            # If probability of a state has values both below_range and above_range,
            #   we can say that a transition happened.
            # In this case it passed through the threshold value
            # If transition did not occur, detect when the state was closest to transitions
            #   i.e. where the probability reached its minimum
            for trial in range(expected_latent_state.shape[1]):
                above_range = expected_latent_state[dom_state,trial,:] > 1 - delta
                below_range = expected_latent_state[dom_state,trial,:] < delta      
                if np.sum(above_range) and np.sum(below_range) > 0:
                    this_transition = np.where(np.abs(np.diff((expected_latent_state[dom_state,trial,:] > probability_thresh)*1)))[0][0]
                    flags.append(1)
                else:
                    this_transition = np.argmin(np.abs(expected_latent_state[dom_state,trial,:] - probability_thresh))
                    flags.append(0)
                transition_points.append(this_transition)
            
            # Convert transition bins to absolute transition times in ms
            transition_times = np.zeros(len(transition_points))        
            for i in range(transition_times.size):
                    transition_times[i] = (transition_points[i]*bin_size) + start_t
                    
            all_transitions.append(transition_times)
            all_flags.append(flags)
        
        self.transitions = all_transitions
        self.flags = all_flags


    def ber_align_trials(self,start_t_data, bin_size_data, 
                     window_size, use_flags = False):
        """
        : data: neurons X trials X time
            : Original data (without any snipping)
        : transition_times: absolute time for transitions as calculates by calc_trans
        : start_t_data: Start time of data FED TO FUNCTION
        : bin_size_data: Bin size of data FED TO FUNCTION
        : window_size: Window DIAMETER around transition points (in milliseconds)
        : use_flags: Use flags created by calc_trans
            If true, only trials with 'proper' transitions will be aligned
        
        Return -> Realigned data with same dims as input data (but cropped in time)
        
        Note: Mean transition times are used for the center of the unaligned data
            to provide a good control. In this case, the data might already have a transition
            across ALL the trials and the HMM is not detecting trial to trial variations. 
        """
        all_transition_times = self.transitions
        all_flags = self.flags
        all_data = self.firing_rate
        
        all_aligned_data = []
        all_unaligned_data = []
        
        for i in range(len(all_data)):
            transition_times = all_transition_times[i]
            flags = all_flags[i]
            data = all_data[i]
            
            
        # If snipping window is greater than supplied data, throw error
            min_transition_time = min(transition_times[transition_times > 0])
            if  min_transition_time - window_size/2 - start_t_data < 0 :
                raise ValueError('Data start time too late to handle window size\nCan support window of %i' % int((min_transition_time - start_t_data)/2))
            
            window_bins = int(window_size/bin_size_data)
            
            # If flags are supplied, take out trials where transitions didn't happen
            if use_flags:
                if flags == None:
                    raise ValueError('use_flags selected but no flags supplied')
                else:
                    aligned_data = np.zeros((data.shape[0],np.sum(flags),window_bins))
                    transition_times = transition_times[flags]
            else:
                aligned_data = np.zeros((data.shape[0],data.shape[1],window_bins))
            
            # For all trials/transitions, go through data and crop-out windows  
            for trial in range(aligned_data.shape[1]):
                # Convert transition time to bins in given data
                transition = int((transition_times[trial] - start_t_data) / bin_size_data)
                aligned_data[:,trial,:] = data[:,trial,transition - int(window_bins/2) : transition + int(window_bins/2)]
            
            # Provide unaligned data for same window size around the average of transition times
            mean_transition_bin = int(np.mean(transition_times)/bin_size_data)
            unaligned_data = data[:,:,(mean_transition_bin - int(window_bins/2)):(mean_transition_bin + int(window_bins/2))]
            
            all_aligned_data.append(aligned_data)
            all_unaligned_data.append(unaligned_data)
            
            self.aligned_data = all_aligned_data
            self.unaligned_data = all_unaligned_data
        
    def calc_correlations(self):      
        off_aligned = []
        off_unaligned = []
        on_aligned = []
        on_unaligned = []
        for taste in range(len(self.aligned_data)):
            off_aligned.append(self.aligned_data[taste][:,self.off_trials[taste],:])
            off_unaligned.append(self.unaligned_data[taste][:,self.off_trials[taste],:])
            on_aligned.append(self.aligned_data[taste][:,self.on_trials[taste],:])
            on_unaligned.append(self.unaligned_data[taste][:,self.on_trials[taste],:])
        
        def list2square(x) :
            trial_num = x[0].shape[1]
            temp = np.zeros((x[0].shape[0],len(x)*trial_num,x[0].shape[2]))
            for i in range(len(x)):
                temp[:,i*trial_num : (i+1)*trial_num,:] = x[i]
            return temp
                
        off_aligned = list2square(off_aligned)
        on_aligned = list2square(on_aligned)
        off_unaligned = list2square(off_unaligned)
        on_unaligned = list2square(on_unaligned)
        
        palatability_vec = np.tile(self.palatability_ranks[:,np.newaxis],int(off_aligned.shape[1]/self.palatability_ranks.size)).flatten()
        
        
        cor_off_aligned = np.zeros((off_aligned.shape[0],off_aligned.shape[2]))
        cor_on_aligned = np.zeros((off_aligned.shape[0],off_aligned.shape[2]))
        cor_off_unaligned = np.zeros((off_aligned.shape[0],off_aligned.shape[2]))
        cor_on_unaligned = np.zeros((off_aligned.shape[0],off_aligned.shape[2]))
        
        for nrn in range(off_aligned.shape[0]):
            for time in range(off_aligned.shape[2]):
                cor_off_aligned[nrn,time] = spearmanr(off_aligned[nrn,:,time],palatability_vec)[0]
                cor_on_aligned[nrn,time] = spearmanr(on_aligned[nrn,:,time],palatability_vec)[0]
                cor_off_unaligned[nrn,time] = spearmanr(off_unaligned[nrn,:,time],palatability_vec)[0]
                cor_on_unaligned[nrn,time] = spearmanr(on_unaligned[nrn,:,time],palatability_vec)[0]
        
        self.correlations['off_al'] = cor_off_aligned
        self.correlations['on_al'] = cor_on_aligned
        self.correlations['off_un'] = cor_off_unaligned
        self.correlations['on_un'] = cor_on_unaligned