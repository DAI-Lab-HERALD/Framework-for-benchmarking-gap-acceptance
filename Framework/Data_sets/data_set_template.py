import pandas as pd
import numpy as np
import os

class data_set_template():
    def __init__(self, model_class_to_path, num_samples_path_pred):
        if not all([hasattr(self, attr) for attr in ['create_path_samples']]):
            raise AttributeError("The raw data has not been loaded yet")
        
        # Find path where preprocessed path can be found
        if os.name == 'nt':
            self.path = '/'.join(os.path.dirname(__file__).split('\\')[:-1])
        else:
            self.path = '/'.join(os.path.dirname(__file__).split('/')[:-1])
        test_file = self.path + '/Results/Data/' + self.get_name() + '-processed_paths.npy'
        if os.path.isfile(test_file):
            [self.Path, 
             self.T, 
             self.Domain_old, 
             self.path_names, 
             self.num_samples] = np.load(test_file, allow_pickle = True)            
        else:
            self.create_path_samples()
            
            if not all([hasattr(self, attr) for attr in ['Path', 'T', 'Domain_old', 'num_samples']]):
                raise AttributeError("The preprocessing has failed, data is missing")
            
            # check some of the aspect to see if pre_process worked
            if not isinstance(self.Path, pd.core.frame.DataFrame):
                raise TypeError("Paths should be saved in a pandas data frame")
            if len(self.Path) != self.num_samples:
                raise TypeError("Path des not have right number of sampels")
                
            self.path_names = self.Path.columns
            
            path_names_test = [name[2:] for name in self.path_names[:4]]
            
            if not path_names_test == ['ego_x', 'ego_y', 'tar_x', 'tar_y']:
                raise AttributeError("Not all required elements are in Path")
            
            if not isinstance(self.T, np.ndarray):
                raise TypeError("Time points should be saved in a numpy array")
            if len(self.T) != self.num_samples:
                raise TypeError("Time points des not have right number of sampels")
                
            if not isinstance(self.Domain_old,  pd.core.frame.DataFrame):
                raise TypeError("Time points should be saved in a numpy array")
            if len(self.Domain_old) != self.num_samples:
                raise TypeError("Time points des not have right number of sampels")
                
                
            
            for i in range(self.num_samples):            
                # check if time input consists out of tuples
                if not isinstance(self.T[i], tuple):
                    raise TypeError("A time point samples is expected to be a tuple.") 
                
                
                test_length = len(self.T[i])
                for j in range(len(self.path_names)):
                    # check if time input consists out of tuples
                    if not isinstance(self.Path.iloc[i,j], tuple):
                        raise TypeError("Path is expected to be consisting of tuples.") 
                    # test if input tuples have right length  
                    length = len(self.Path.iloc[i,j])
                    if test_length != length:
                        raise TypeError("Path sample does not have a matching number of timesteps")  
                        
            # save the results
            save_data = np.array([self.Path, 
                                  self.T, 
                                  self.Domain_old,
                                  self.path_names,
                                  self.num_samples], object)
            np.save(test_file, save_data)

        self.data_loaded = False
        
        if model_class_to_path == None:
            self.model_class_to_path = None
        elif model_class_to_path != None and model_class_to_path.get_output_type_class() != 'path':
            raise TypeError("The model used for output type transformation towards path has to be a path model.")
        else:
            self.model_class_to_path = model_class_to_path
        
        
        if type(num_samples_path_pred) != type(0):
            raise TypeError("num_samples_path_pred should be an integer")

        self.num_samples_path_pred = max(1, num_samples_path_pred)
        
                
    def reset(self):
        self.data_loaded = False
        
    
    def extract_time_points(self):
        # here for analyizing the dataset
        self.data_time_file = (self.path + 
                               '/Results/Data/' + 
                               self.get_name() + 
                               '-time_points.npy')
        
        if os.path.isfile(self.data_time_file):
            [self.accepted_id,
             self.accepted_tchat,
             self.accepted_ts,
             self.accepted_tchat_ts,
             self.accepted_ta,
             self.accepted_tchat_ta,
             self.accepted_tcrit,
             self.rejected_id,
             self.rejected_tchat,
             self.rejected_ts,
             self.rejected_tchat_ts,
             self.rejected_tc,
             self.rejected_tchat_tc,
             self.rejected_tcrit, _] = np.load(self.data_time_file, allow_pickle = True)
        
        else:
            self.accepted_id = []
            self.accepted_tchat = []
            self.accepted_ts = []
            self.accepted_tchat_ts = []
            self.accepted_ta = []
            self.accepted_tchat_ta = []
            self.accepted_tcrit = []
            
            self.rejected_id = []
            self.rejected_tchat = []
            self.rejected_ts = []
            self.rejected_tchat_ts = []
            self.rejected_tc = []
            self.rejected_tchat_tc = []
            self.rejected_tcrit = []
            
            for i_sample in range(self.num_samples):
                if np.mod(i_sample,100) == 0:
                    print('path ' + str(i_sample).rjust(len(str(self.num_samples))) + '/{} divided'.format(self.num_samples))
                
                path = self.Path.iloc[i_sample]
                domain = self.Domain_old.iloc[i_sample]
                t = np.array(self.T[i_sample])
                
                [undecided, accepted, rejected, critical, 
                 undecided_2_accepted, undecided_2_rejected, not_critical_2_critical,
                 tchat] = self.position_states_in_path(path, t, domain)
                
                try:
                    ind_decided = np.where(undecided[:-1] & (accepted[1:] | rejected[1:]))[0][0] + 1
                    possible_ind_s = np.where(undecided[:ind_decided])[0]
                    try:
                        ind_s = possible_ind_s[np.where(possible_ind_s[:-1] + 1 != possible_ind_s[1:])[0][-1] + 1]
                    except:
                        ind_s = possible_ind_s[0]
                except:
                    # there never was as starting point here in the first place
                    continue
                
                ts = t[ind_s]
                tchat_ts = tchat[ind_s]
                
                feasible = (t > ts)
                
                try:
                    ind_crit_2 = np.where(critical & feasible)[0][0]
                    ind_crit_1 = ind_crit_2 - 1
                    if critical[ind_crit_1]:
                        tcrit = t[ind_crit_1]
                    else:
                        fac_crit = np.abs(not_critical_2_critical[ind_crit_2]) / (np.abs(not_critical_2_critical[ind_crit_2]) + 
                                                                             np.abs(not_critical_2_critical[ind_crit_1]))
                        tcrit = t[ind_crit_2] * fac_crit + t[ind_crit_1] * (1 - fac_crit)
                except:
                    tcrit = t.max() + 1000
                
                if accepted[ind_decided]:
                    # gap is accepted
                    ind_decided_2 = ind_decided
                    ind_decided_1 = ind_decided_2 - 1
                    fac_decided = np.abs(undecided_2_accepted[ind_decided_2]) / (np.abs(undecided_2_accepted[ind_decided_2]) + 
                                                                                 np.abs(undecided_2_accepted[ind_decided_1]))
                    ta = t[ind_decided_2] * fac_decided + t[ind_decided_1] * (1 - fac_decided)
                    tchat_ta = tchat[ind_decided_2] * fac_decided + tchat[ind_decided_1] * (1 - fac_decided)
                    
                    self.accepted_id.append(i_sample)
                    self.accepted_tchat.append(tchat)
                    self.accepted_ts.append(ts)
                    self.accepted_tchat_ts.append(tchat_ts)
                    self.accepted_ta.append(ta)
                    self.accepted_tchat_ta.append(tchat_ta)
                    self.accepted_tcrit.append(tcrit)
    
                elif rejected[ind_decided]:
                    # gap is rejected
                    ind_decided_2 = ind_decided
                    ind_decided_1 = ind_decided_2 - 1
                    fac_decided = np.abs(undecided_2_rejected[ind_decided_2]) / (np.abs(undecided_2_rejected[ind_decided_2]) + 
                                                                                 np.abs(undecided_2_rejected[ind_decided_1]))
                    tc = t[ind_decided_2] * fac_decided + t[ind_decided_1] * (1 - fac_decided)
                    tchat_tc = tchat[ind_decided_2] * fac_decided + tchat[ind_decided_1] * (1 - fac_decided)
                    
                    
                    self.rejected_id.append(i_sample)
                    self.rejected_tchat.append(tchat)
                    self.rejected_ts.append(ts)
                    self.rejected_tchat_ts.append(tchat_ts)
                    self.rejected_tc.append(tc)
                    self.rejected_tchat_tc.append(tchat_tc)
                    self.rejected_tcrit.append(tcrit)
                    
                else:
                    raise TypeError("neither accepted or rejected despite prior filtering")
                
            self.accepted_id        = np.array(self.accepted_id)
            self.accepted_tchat     = np.array(self.accepted_tchat)
            self.accepted_ts        = np.array(self.accepted_ts)
            self.accepted_tchat_ts  = np.array(self.accepted_tchat_ts)
            self.accepted_ta        = np.array(self.accepted_ta)
            self.accepted_tchat_ta  = np.array(self.accepted_tchat_ta)
            self.accepted_tcrit     = np.array(self.accepted_tcrit)
            
            self.rejected_id        = np.array(self.rejected_id)
            self.rejected_tchat     = np.array(self.rejected_tchat)
            self.rejected_ts        = np.array(self.rejected_ts)
            self.rejected_tchat_ts  = np.array(self.rejected_tchat_ts)
            self.rejected_tc        = np.array(self.rejected_tc) 
            self.rejected_tchat_tc  = np.array(self.rejected_tchat_tc)
            self.rejected_tcrit     = np.array(self.rejected_tcrit)   
            
            save_data_time = np.array([self.accepted_id,
                                       self.accepted_tchat,
                                       self.accepted_ts,
                                       self.accepted_tchat_ts,
                                       self.accepted_ta,
                                       self.accepted_tchat_ta,
                                       self.accepted_tcrit,
                                       self.rejected_id,
                                       self.rejected_tchat,
                                       self.rejected_ts,
                                       self.rejected_tchat_ts,
                                       self.rejected_tc,
                                       self.rejected_tchat_tc,
                                       self.rejected_tcrit, 0], object) #0 is there to avoid some numpy load and save errros
            np.save(self.data_time_file, save_data_time)
        
    
    def determine_dtc_boundary(self):
        self.data_dtc_bound_file = (self.path + 
                                    '/Results/Data/' + 
                                    self.get_name() + 
                                    '-dtc_bound.npy')
        
        if os.path.isfile(self.data_dtc_bound_file):
            self.dtc_boundary = np.load(self.data_dtc_bound_file, allow_pickle = True)[0]
        else:
            accepted_dts = (self.accepted_tchat_ts - self.accepted_ts)[:,np.newaxis]
            accepted_dta = (self.accepted_tchat_ta - self.accepted_ta)[:,np.newaxis]   
    
            rejected_dts = (self.rejected_tchat_ts- self.rejected_ts)[:,np.newaxis]
            
            # samples with accepted_dta > dts_boundary and accepted_dts < dts_boundary will not be included
            # samples with rejected_dts < dts_boundary will be rejected
            dtc_boundaries = np.linspace(0, 20, 201)[np.newaxis,:]
            
            use_accepted = (dtc_boundaries < accepted_dts) & (dtc_boundaries > accepted_dta)
            num_accepted = use_accepted.sum(axis = 0)
    
            use_rejected = dtc_boundaries < rejected_dts 
            num_rejected = use_rejected.sum(axis = 0)
    
            num = np.minimum(num_accepted, num_rejected)
    
            self.dtc_boundary = dtc_boundaries[0, np.argmax(num)]
            np.save(self.data_dtc_bound_file, np.array([self.dtc_boundary]))
        
        print('For predictions on dataset ' + self.get_name() + ' at gaps with fixed sizes, a size of {} s was chosen'.format(self.dtc_boundary))
    
    def get_data(self, t0_type, dt, num_timesteps_in, exclude_post_crit):
        '''
        Parameters
        ----------
        t0_type : string
            The timepoint at which a prediction should be made: 
            'start':    takes as starting point t the first time the ego vehicle offered a gap 
                        to the target vehicle. One then uses t0 = t_S + dt * (num_timesteps_in - 1).
            'recog':    takes as starting point t the first time the target vehicle became aware of
                        the ego vehicle. One then uses t0 = t_R + dt * (num_timesteps_in - 1). 
            'col_m':    takes t0 = min( t : t_C(t) - t = m * dt)
                        
        dt : float
            The difference between time steps.
        num_timesteps_in : int
            The number of timestpes for input. The first timepoint is then t0 - dt * (num_timesteps_in - 1).
        exclude_decided : bool
            If true, cases with t_A < t_0 are excluded from the dataset, as they force the model to 
            learn to detect acceptance maneuvers and not to predict them
            

        Returns
        -------
        input and output data
    
        '''
        self.extract_time_points()
        
        self.dt = dt
        
        num_accepted = len(self.accepted_id)
        num_rejected = len(self.rejected_id)
        
        num_samples = num_accepted + num_rejected
        
        # check if same data set has already been done in the same way
        self.data_file = (self.path + 
                          '/Results/Data/' + 
                          self.get_name() + 
                          '-io_paths(' + 
                          't0=' + t0_type + 
                          '_dt={}'.format(self.dt) +
                          '_nI={}'.format(num_timesteps_in) + 
                          '_EC' * exclude_post_crit +  
                          ').npy')
        
        if os.path.isfile(self.data_file):
            
            
            [self.Input_path, 
             self.Input_T, 
             self.Output_path, 
             self.Output_T, 
             self.Output_A, 
             self.Output_T_E, 
             self.Domain, _] = np.load(self.data_file, allow_pickle = True)
            
        else:
            
            if t0_type[:3] == 'col':
                self.determine_dtc_boundary()
            
            # prepare empty information, where this 
            Input_path = []
            
            Output_path = []
            
            Input_T = []
            
            Output_T = []
            
            Output_A = []
            
            Output_T_E = []
            
            Domain = []
            
            
            for i in range(num_samples):
                # print progress
                if np.mod(i,100) == 0:
                    print('path ' + str(i).rjust(len(str(num_samples))) + '/{} divided'.format(num_samples))
                
                if i < num_accepted:
                    # accepted cases
                    i_accepted = i
                    i_path = self.accepted_id[i_accepted]
                    
                    tchat = self.accepted_tchat[i_accepted]
                    
                    tstart = self.accepted_ts[i_accepted]
                    tendd = self.accepted_ta[i_accepted]
                    tendp = self.accepted_tchat_ta[i_accepted]
                    tcrit = self.accepted_tcrit[i_accepted]
                    
                    output_A = True
                    output_T_E = tendd
   
                else:
                    # rejected cases
                    i_rejected = i - num_accepted
                    i_path = self.rejected_id[i_rejected]
                    
                    tchat = self.rejected_tchat[i_rejected]
                    
                    tstart = self.rejected_ts[i_rejected]
                    tendd = self.rejected_tc[i_rejected]
                    tendp = tendd
                    tcrit = self.rejected_tcrit[i_rejected]
                    
                    output_A = False
                    output_T_E = tendd
                
                # load old data
                path = self.Path.iloc[i_path]
                t = np.array(self.T[i_path])
                domain = self.Domain_old.iloc[i_path]
                # Needed for later recovery of path data
                domain['Path_ID'] = i_path 
                
                ## extract prediction point
                if t0_type[:5] == 'start':
                    t0 = tstart
                elif t0_type[:3] == 'col':
                    dtc = tchat - t
                    try:
                        ind_0 = np.where((dtc < self.dtc_boundary) & (t >= tstart))[0][0]
                        t0 = t[ind_0] - (self.dtc_boundary - dtc[ind_0])
                    except:
                        t0 = tendd + self.dt
                elif t0_type[:4] == 'crit':
                    t0 = tcrit - 0.01
                else:
                    raise TypeError("This type of starting point is not defined")
                    
                # weaken start point if allowed and necessary 
                if t0_type[-6:] != 'strict':
                    t0 = max(t0, np.min(t) + (num_timesteps_in - 1) * self.dt)
                
                ## perform necessary checks:
                # guarantee sufficient input data
                if t0 < np.min(t) + (num_timesteps_in - 1) * self.dt:
                    continue
                # guarantee position is valid
                if t0 < tstart:
                    continue
                # guarantee no made decision
                if t0 >= tendd:
                    continue
                # guarantee useful prediction
                if exclude_post_crit and t0 >= tcrit:
                    continue
                # guarantee enough output data
                num_timesteps_out = int(np.ceil((tendp - t0)/self.dt))
                if t0 + num_timesteps_out * self.dt > np.max(t) or num_timesteps_out <= 1:
                    # relax for excepted, as long as t_a is still part of the data
                    if output_A:
                        num_timesteps_out = int(np.floor((np.max(t) - t0)/self.dt))
                        if t0 + num_timesteps_out * self.dt < tendd or num_timesteps_out <= 1:
                            continue
                    else:
                        continue
                    
                ## build new path data
                # create time points
                input_T = t0 + np.arange(1 - num_timesteps_in, 1) * self.dt
                output_T = t0 + np.arange(1, num_timesteps_out + 1) * self.dt
                
                if np.max(output_T) > np.max(t):
                    # not enough future time data to get output data
                    continue
                
                # prepare empty pandas series for path
                input_path = pd.Series(np.empty(len(self.path_names), object), index = self.path_names)
                output_path = pd.Series(np.empty(2, object), index = self.path_names[2:4])
                
                # fill in input paths
                for index in input_path.index:
                    input_path[index] = np.interp(input_T + 1e-5, # + 1e-5 is necessary to avoid nan for min(input_T == min(t))
                                                  t, 
                                                  path[index], 
                                                  left = np.nan, 
                                                  right = np.nan)
                    if index[2:] in ['ego_x', 'tar_x'] and np.isnan(input_path[index]).any():
                        print('strange')
                     
                
                # fill in putput path
                for index in output_path.index:
                    output_path[index] = np.interp(output_T - 1e-5, # - 1e-5 is necessary to avoid nan for min(output_T == max(t))
                                                   t, 
                                                   path[index], 
                                                   left = np.nan, 
                                                   right = np.nan)
                    if index[2:] == 'tar_x' and np.isnan(output_path[index]).any():
                        print('strange')
                
                # reset time origin
                input_T = input_T - t0
                output_T = output_T - t0 
                output_T_E = output_T_E - t0
                
                ## save results
                Input_path.append(input_path)
                Output_path.append(output_path)
                Input_T.append(input_T)
                Output_T.append(output_T)
                Output_A.append(output_A)
                Output_T_E.append(output_T_E)
                Domain.append(domain)
            
            
            
            self.Input_path = pd.DataFrame(Input_path)
            self.Output_path = pd.DataFrame(Output_path)
            self.Input_T = np.array(Input_T+[np.random.rand(0)], object)[:-1]
            self.Output_T =  np.array(Output_T+[np.random.rand(0)], object)[:-1]
            self.Output_A = np.array(Output_A, bool)
            self.Output_T_E = np.array(Output_T_E, float)
            self.Domain = pd.DataFrame(Domain).reset_index(drop = True)
            
            
            
            self.fill_empty_input_path()
            
                
            
            save_data = np.array([self.Input_path, 
                                  self.Input_T, 
                                  self.Output_path, 
                                  self.Output_T, 
                                  self.Output_A, 
                                  self.Output_T_E, 
                                  self.Domain, 0], object) #0 is there to avoid some numpy load and save errros
            np.save(self.data_file, save_data)
             
        print("")
        print("")
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print("")    
        
        print("The dataset " + self.get_name() + 
        ' (with t0 = ' + t0_type + 
        ', dt = {}'.format(self.dt) +
        ' and nI = {})'.format(num_timesteps_in))
        print("includes {}/{} possible samples:".format(len(self.Output_A), num_samples))
        
        print("{}/{} accepted gaps and {}/{} rejected gaps".format(self.Output_A.sum(),
                                                                   num_accepted,
                                                                   len(self.Output_A) - self.Output_A.sum(),
                                                                   num_rejected))
        print("")
        print("---------------------------------------------------------------")
        print("")
        
        abort = False
        if 5 > min(self.Output_A.sum(), len(self.Output_A) - self.Output_A.sum()):
            print("An unbalanced dataset has nothing to learn from")
            print("")
            abort = True
            
        if len(self.Output_A) < 100:
            print("There are not enough samples for a reasonable training")
            print("")
            abort = True
            
        if abort:
            return None
        
        
        self.data_loaded = True
        
        return [self.Input_path, 
                self.Input_T, 
                self.Output_path, 
                self.Output_T, 
                self.Output_A, 
                self.Output_T_E, 
                self.Domain, 
                self.data_file]
    
    def path_to_binary_and_time(self, Output_path_pred, Output_T_pred, Domain, model_save_file, save_results = True):
        
        if save_results:
            test_file = (model_save_file[:-4] + 
                         '_prediction' +
                         '_T2bt' +  
                         '.npy')
            
            if os.path.isfile(test_file):
                [Output_A_pred, 
                 Output_T_E_pred, _] = np.load(test_file, allow_pickle = True)
            else:
                t_A_quantile = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                # Different probable predictions are saved in a list:
                # 1 sample is a pandas series
                # iterrate over test samples
                # Prepare binary decision as scalar value
                Output_A_pred = np.empty(len(Output_T_pred), 'float64')
                # Prepare action time (in case of positive decision) as quantile
                Output_T_E_pred = np.empty(len(Output_T_pred), object)
                
                for (i_sample, output_path_pred) in Output_path_pred.iterrows():
                    # Number of discrete paths that represent probability distribution
                    number_poss = None
                    domain = Domain.iloc[i_sample]
                    
                    output_path_i = output_path_pred.copy()
                    # each sample.index now consits not of a tuple of length n_O, 
                    # but instead of a list of number_poss tuples of length n_O
                    for index in output_path_pred.index: # index here is tar_x and tar_y
                        if number_poss == None:
                            number_poss = len(output_path_pred[index])
                        else:
                            if number_poss != len(output_path_pred[index]):
                                raise TypeError("Not all states have the same number of discrete samples")
                                
                    output_a = np.zeros(number_poss)
                    output_t_e = np.zeros(number_poss)
                    for i_poss in range(number_poss):
                        for index in output_path_pred.index:
                            output_path_i[index] = output_path_pred[index][i_poss]
                        
                        # get a and t_a for each discrete path
                        output_a[i_poss], output_t_e[i_poss] = self.path_to_binary_and_time_sample(output_path_i, 
                                                                                                   Output_T_pred[i_sample], 
                                                                                                   domain) 
                        
                    Output_A_pred[i_sample] = output_a.mean()
                    # Create quantile distribution over T_E, only for accepted cases
                    Output_T_E_pred[i_sample] = np.nanquantile(output_t_e, t_A_quantile)
                
                save_data = np.array([Output_A_pred, Output_T_E_pred, 0], object)
                np.save(test_file, save_data)
        else:
            t_A_quantile = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # Different probable predictions are saved in a list:
            # 1 sample is a pandas series
            # iterrate over test samples
            # Prepare binary decision as scalar value
            Output_A_pred = np.empty(len(Output_T_pred), 'float64')
            # Prepare action time (in case of positive decision) as quantile
            Output_T_E_pred = np.empty(len(Output_T_pred), object)
            
            for (i_sample, output_path_pred) in Output_path_pred.iterrows():
                # Number of discrete paths that represent probability distribution
                number_poss = None
                domain = Domain.iloc[i_sample]
                
                output_path_i = output_path_pred.copy()
                # each sample.index now consits not of a tuple of length n_O, 
                # but instead of a list of number_poss tuples of length n_O
                for index in output_path_pred.index: # index here is tar_x and tar_y
                    if number_poss == None:
                        number_poss = len(output_path_pred[index])
                    else:
                        if number_poss != len(output_path_pred[index]):
                            raise TypeError("Not all states have the same number of discrete samples")
                            
                output_a = np.zeros(number_poss)
                output_t_e = np.zeros(number_poss)
                for i_poss in range(number_poss):
                    for index in output_path_pred.index:
                        output_path_i[index] = output_path_pred[index][i_poss]
                    
                    # get a and t_a for each discrete path
                    output_a[i_poss], output_t_e[i_poss] = self.path_to_binary_and_time_sample(output_path_i, 
                                                                                               Output_T_pred[i_sample],
                                                                                               domain) 
                    
                Output_A_pred[i_sample] = output_a.mean()
                # Create quantile distribution over T_E, only for accepted cases
                Output_T_E_pred[i_sample] = np.nanquantile(output_t_e, t_A_quantile)

        return Output_A_pred, Output_T_E_pred
    
    def train_path_accepted(self):
        if not self.data_loaded:
            raise AttributeError("Input and Output data has not yet been specified")
        
        
        test_file_accepted = self.data_file[:-4] + '-T2p(a).npy'
          
        self.path_model_accepted = self.model_class_to_path(Input_path_train = self.Input_path.iloc[self.Output_A],
                                                            Input_T_train = self.Input_T[self.Output_A],
                                                            Output_path_train = self.Output_path.iloc[self.Output_A], 
                                                            Output_T_train = self.Output_T[self.Output_A], 
                                                            Output_A_train = self.Output_A[self.Output_A], 
                                                            Output_T_E_train = self.Output_T_E[self.Output_A], 
                                                            Domain_train = self.Domain.iloc[self.Output_A], 
                                                            splitter_save_file = test_file_accepted,
                                                            num_samples_path_pred = self.num_samples_path_pred)
        self.path_model_accepted.train()
        
    
    def train_path_rejected(self):
        if not self.data_loaded:
            raise AttributeError("Input and Output data has not yet been specified")
        
        test_file_rejected = self.data_file[:-4] + '-T2p(r).npy'
        
        Output_An = np.invert(self.Output_A)
        self.path_model_rejected = self.model_class_to_path(Input_path_train = self.Input_path.iloc[Output_An],
                                                            Input_T_train = self.Input_T[Output_An],
                                                            Output_path_train = self.Output_path.iloc[Output_An], 
                                                            Output_T_train = self.Output_T[Output_An], 
                                                            Output_A_train = self.Output_A[Output_An], 
                                                            Output_T_E_train = self.Output_T_E[Output_An], 
                                                            Domain_train = self.Domain.iloc[Output_An], 
                                                            splitter_save_file = test_file_rejected,
                                                            num_samples_path_pred = self.num_samples_path_pred)   
        
        self.path_model_rejected.train()
        
        
    def binary_to_time(self, Output_A_pred, Domain, model_save_file, splitter_save_file):
        test_file_outer = (model_save_file[:-4] + 
                           '_prediction' +
                           '_T2t' +  
                           '.npy')
        if os.path.isfile(test_file_outer):
            [Output_T_E_pred, _] = np.load(test_file_outer, allow_pickle = True)
        else:
            test_file_inner_accepted = (self.data_file[:-4] + 
                                        '-T2p(a)' +   
                                        '_prediction' +
                                        '.npy')
            
            if os.path.isfile(test_file_inner_accepted):
                [Output_path_apred_accepted, _] = np.load(test_file_inner_accepted, allow_pickle = True)
            else:
                self.train_path_accepted()
                
                [Output_path_apred_accepted] = self.path_model_accepted.predict(Input_path_test = self.Input_path, 
                                                                                Input_T_test = self.Input_T,
                                                                                Output_T_test = self.Output_T,
                                                                                save_results = False)
                for i_sample in range(len(Output_path_apred_accepted)):
                    # get path predictions
                    output_path_pred_accepted = Output_path_apred_accepted.iloc[i_sample]
                    
                    # get time_step predictions
                    number_poss = None
                    # check number of predictions
                    for index in output_path_pred_accepted.index: # index here is tar_x and tar_y
                        if number_poss == None:
                            number_poss = len(output_path_pred_accepted[index])
                        else:
                            if number_poss != len(output_path_pred_accepted[index]):
                                raise TypeError("Not all rejected states have the same number of discrete samples")
                
                save_data_inner_accepted = np.array([Output_path_apred_accepted, 0], object)
                np.save(test_file_inner_accepted, save_data_inner_accepted)
                
            [_, Test_index, _, _, _] = np.load(splitter_save_file, allow_pickle = True)
            Output_path_pred_accepted = Output_path_apred_accepted.iloc[Test_index].reset_index(drop = True)
            Output_T_pred_accepted = self.Output_T[Test_index]
                
            t_A_quantile = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # Different probable predictions are saved in a list:
            # 1 sample is a pandas series
            # iterrate over test samples
            # Prepare binary decision as scalar value
            
            # Prepare action time (in case of positive decision) as quantile
            Output_T_E_pred = np.empty(len(Output_T_pred_accepted), object)
            
            for (i_sample, output_path_pred) in Output_path_pred_accepted.iterrows():
                # Number of discrete paths that represent probability distribution
                
                number_poss = len(output_path_pred[output_path_pred.index[0]])
                output_t_e = np.zeros(number_poss)
                output_path_i = output_path_pred.copy()
                domain = Domain.iloc[i_sample]
                for i_poss in range(number_poss):
                    for index in output_path_pred.index:
                        output_path_i[index] = output_path_pred[index][i_poss]
                    
                    # get a and t_a for each discrete path
                    _, output_t_e[i_poss] = self.path_to_binary_and_time_sample(output_path_i, 
                                                                                Output_T_pred_accepted[i_sample], 
                                                                                domain) 
                    
                Output_T_E_pred[i_sample] = np.nanquantile(output_t_e, t_A_quantile)
            save_data_outer = np.array([Output_T_E_pred, 0], object)
            np.save(test_file_outer, save_data_outer)
        return Output_T_E_pred
    
    
    
    def binary_and_time_to_path(self, Output_A_pred, Output_T_E_pred, Domain, model_save_file, splitter_save_file):
        # check if this has already been performed
        test_file_outer = (model_save_file[:-4] + 
                           '_prediction' +
                           '_T2p' +  
                           '.npy')
        if os.path.isfile(test_file_outer):
            [Output_path_pred, _] = np.load(test_file_outer, allow_pickle = True)
        else:
            # check if the specific testing case set has already been predicted
            test_file_inner_rejected = (self.data_file[:-4] + 
                                        '-T2p(r)' +   
                                        '_prediction' +
                                        '.npy')
            
            
            test_file_inner_accepted = (self.data_file[:-4] + 
                                        '-T2p(a)' +   
                                        '_prediction' +
                                        '.npy')
            
            
            if os.path.isfile(test_file_inner_accepted):
                [Output_path_apred_accepted, _] = np.load(test_file_inner_accepted, allow_pickle = True)
            else:
                self.train_path_accepted()
                
                # accepted
                [Output_path_apred_accepted] = self.path_model_accepted.predict(Input_path_test = self.Input_path, 
                                                                                Input_T_test = self.Input_T,
                                                                                Output_T_test = self.Output_T,
                                                                                save_results = False)
                for i_sample in range(len(Output_path_apred_accepted)):
                    # get path predictions
                    output_path_pred_accepted = Output_path_apred_accepted.iloc[i_sample]
                    # get time_step predictions
                    number_poss = len(output_path_pred_accepted[output_path_pred_accepted.index[0]])
                    # check number of predictions
                    for index in output_path_pred_accepted.index: # index here is tar_x and tar_y
                        if number_poss != len(output_path_pred_accepted[index]):
                            raise TypeError("Not all accepted states have the same number of discrete samples")
                            
                save_data_inner_accepted = np.array([Output_path_apred_accepted, 0], object)
                np.save(test_file_inner_accepted, save_data_inner_accepted)
            
            
            if os.path.isfile(test_file_inner_rejected):
                [Output_path_apred_rejected, _] = np.load(test_file_inner_rejected, allow_pickle = True)
            else:
                self.train_path_rejected()
                
                # predict paths assuming a rejected gap
                [Output_path_apred_rejected] = self.path_model_rejected.predict(Input_path_test = self.Input_path, 
                                                                                Input_T_test = self.Input_T,
                                                                                Output_T_test = self.Output_T,
                                                                                save_results = False)
                
                for i_sample in range(len(Output_path_apred_rejected)):
                    # get path predictions
                    output_path_pred_rejected = Output_path_apred_rejected.iloc[i_sample]
                    output_path_pred_accepted = Output_path_apred_accepted.iloc[i_sample]
                    
                    # get time_step predictions
                    output_t_pred = self.Output_T[i_sample]
                    number_poss = len(output_path_pred_accepted[output_path_pred_accepted.index[0]])
                    # check number of predictions
                    for index in output_path_pred_rejected.index: # index here is tar_x and tar_y
                        if number_poss != len(output_path_pred_rejected[index]):
                            raise TypeError("Not all accepted states have the same number of discrete samples")
                            
                save_data_inner_rejected = np.array([Output_path_apred_rejected, 0], object)
                np.save(test_file_inner_rejected, save_data_inner_rejected)
                
                
            [_, Test_index, _, _, _] = np.load(splitter_save_file, allow_pickle = True)
            Output_path_pred_accepted = Output_path_apred_accepted.iloc[Test_index]
            Output_path_pred_rejected = Output_path_apred_rejected.iloc[Test_index]
            Output_T_pred = self.Output_T[Test_index]   
            Output_path_pred = pd.DataFrame(np.empty((len(Output_path_pred_accepted),2), object), columns = self.path_names[2:4])
            
            # Go through each sample:
            for i_sample in range(len(Output_path_pred_accepted)):
                # get path predictions
                output_path_pred_rejected = Output_path_pred_rejected.iloc[i_sample]
                output_path_pred_accepted = Output_path_pred_accepted.iloc[i_sample]
                # This should be sorted
                output_t_e_pred = np.array(Output_T_E_pred[i_sample])
                # get time_step predictions
                output_t_pred = Output_T_pred[i_sample]
                domain = Domain.iloc[i_sample]
                
                
                # For rejected cases: check if those are really rejected, to avoid a potential source of error 
                # that might come from the path prediction error
                # Get number of possible paths taken from each prediction model
                number_poss = len(output_path_pred_rejected[output_path_pred_rejected.index[0]])
                number_poss_rejected = int(number_poss - round(number_poss * Output_A_pred[i_sample]))
                output_path_i_rejected = output_path_pred_rejected.copy()     
                output_a_rejected = np.zeros(number_poss, bool)
                for i_poss in range(number_poss):
                    for index in output_path_pred_rejected.index:
                        output_path_i_rejected[index] = output_path_pred_rejected[index][i_poss]
                    output_a_rejected[i_poss], _ = self.path_to_binary_and_time_sample(output_path_i_rejected, 
                                                                                       output_t_pred, 
                                                                                       domain) 
                
                # only use random choice here
                possible_cases_rejected = np.arange(number_poss)[np.invert(output_a_rejected)]
                if len(possible_cases_rejected) == 0:
                    random_choice_rejected = np.random.choice(np.arange(number_poss), number_poss_rejected)
                elif len(possible_cases_rejected) >= number_poss_rejected:
                    random_choice_rejected = np.random.choice(possible_cases_rejected,
                                                              number_poss_rejected, replace = False)
                else: # if not enough useful cases exist, allow repetition
                    random_choice_rejected = np.random.choice(possible_cases_rejected, 
                                                              number_poss_rejected, replace = True)
                
                # Add usable cases to final result
                for index in output_path_pred_rejected.index:
                    Output_path_pred.iloc[i_sample][index] = np.zeros((number_poss, len(output_path_pred_rejected[index][0])))
                    Output_path_pred.iloc[i_sample][index][:number_poss_rejected] = output_path_pred_rejected[index][random_choice_rejected]
                
            
                # For accepted cases: check if those are really rejected, to avoid a potential source of error 
                # that might come from the path prediction error, also save t_A for later allignment
                number_poss_accepted = number_poss - number_poss_rejected 
                output_path_i_accepted = output_path_pred_rejected.copy()     
                output_a_accepted = np.zeros(number_poss, bool)
                output_t_e_accepted = np.zeros(number_poss)
                for i_poss in range(number_poss):
                    for index in output_path_pred_accepted.index:
                        output_path_i_accepted[index] = output_path_pred_accepted[index][i_poss]
                    [output_a_accepted[i_poss], 
                     output_t_e_accepted[i_poss]] = self.path_to_binary_and_time_sample(output_path_i_accepted, 
                                                                                        output_t_pred, 
                                                                                        domain) 
                
                possible_cases_accepted = np.arange(number_poss)[output_a_accepted]
                possible_cases_accepted_t_e = np.clip(output_t_e_accepted[output_a_accepted], None, 1e5)
                temp =  np.hstack((output_t_e_pred, [1e6]))[np.newaxis, :]
                possible_cases_bracket = np.argmax(possible_cases_accepted_t_e[:,np.newaxis] < temp, axis = 1)
                # get counts of each bin, the - 1 is there to offset the addition of the np.arange
                possible_cases_counts = np.bincount(np.concatenate((possible_cases_bracket, np.arange(temp.shape[1])))) - 1
                weights = 1/(1e-5 + possible_cases_counts[possible_cases_bracket])
                weights = weights / weights.sum()
                if len(possible_cases_accepted) == 0:
                    random_choice_accepted = np.random.choice(np.arange(number_poss), number_poss_accepted)
                else:
                    random_choice_accepted = np.random.choice(possible_cases_accepted, number_poss_accepted, p=weights)
                
                # Add usable cases to final result
                for index in output_path_pred_rejected.index:
                    Output_path_pred.iloc[i_sample][index][number_poss_rejected:] = output_path_pred_accepted[index][random_choice_accepted]
                    Output_path_pred.iloc[i_sample][index] = Output_path_pred.iloc[i_sample][index].astype('float32')
                
            save_data_outer = np.array([Output_path_pred, 0], object)
            np.save(test_file_outer, save_data_outer)
        return Output_path_pred



    def get_name(self):
        return self.__str__()[1:-1].split('.')[0]
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                         Data-set dependend functions                              ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    
    def create_path_samples(self): # Data should be there after initialization already
        raise AttributeError('Has to be overridden in actual data-set class')
        # Loads self.Data from wherever it is saved
        # Extract samples from data
        # each sample should then contain either one accepted or rejected gap 
        # creates:
            # self.Path -           a pandas array of samples, with each column corresponding to
            #                       a certain position of a vehicle
            # self.T -              the timesteps at which the path data was sampled
            # self.Domain_old -     a pandas array with a number of columns which allow
            #                       to calssifiy different samples in different domains
            # self.num_Samples -    the number of path samples
        
    def position_states_in_path(self, path, t, domain):
        # takes the path input and determines at each timesteps as what the situation can be classified,
        # which might either be undecided, turned, passed or none of the three
        # at each timestep, only one of these possibilities can be true
        # these arrays are then returned as, together with the estimated time of
        # closing the gap, tchat
        # tchat should be larger than t if the gap is still open, but negative otherwise
        raise AttributeError('Has to be overridden in actual data-set class')
        # return undecided, turned, passed, critical, tchat
     
    def fill_empty_input_path(self):
        # There to fill up np.nan places in the input paths
        # (needed for better moddeling and calculating similarity)
        raise AttributeError('Has to be overridden in actual data-set class')

     
    def path_to_binary_and_time_sample(self, output_path, output_t, domain):
        # takes a single output path and corresponding timepoints 
        # and determines if by max(output_t) a gap was accepted or not.
        # if the gap was accepted, the time of acceptance output_t_e is 
        # also provided, which is otherwise set to None
        raise AttributeError('Has to be overridden in actual data-set class')
        # return output_a, output_t_e
        
    def provide_map_drawing(self, domain):
        # Provides the line information needed for the drawing of the predicted paths
        # and the corresponding analysis
        raise AttributeError("Has to be overridden in actual data-set class")
        # return lines_solid, lines_dashed

    
    
        
    
    
    
