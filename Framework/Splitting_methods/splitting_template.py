import pandas as pd
import numpy as np
import os
import scipy as sp




class splitting_template():
       
    def split_data(self, Input_path, Input_T, 
                   Output_path, Output_T, 
                   Output_A, Output_T_E, 
                   Domain, data_set_save_file):
        
        path = os.path.dirname(data_set_save_file)[:-4]
        file = os.path.basename(data_set_save_file)[:-4]
        
        self.split_file = (path +
                           'Splitting/' + 
                           file +
                           '-' + 
                           self.get_name() + 
                           '.npy')
        
        if os.path.isfile(self.split_file):
            [self.Train_index, 
             self.Test_index,
             sim_all,
             Sim_any, _] = np.load(self.split_file, allow_pickle = True)
        else:
            self.split_data_method(Input_path, Input_T, 
                                   Output_path, Output_T, 
                                   Output_A, Output_T_E, Domain)
             
            # check if split was successful
            if not all([hasattr(self, attr) for attr in ['Train_index', 'Test_index']]):
                 raise AttributeError("The splitting into train and test set was unsuccesful")
            
            np.random.shuffle(self.Train_index)
            np.random.shuffle(self.Test_index)
            # Determine similarity
            sim_all, Sim_any = self.get_similarity_method(Input_path, Input_T, 
                                                          Output_path, Output_T, 
                                                          Output_A, Output_T_E, Domain)
            
            save_data = np.array([self.Train_index, 
                                  self.Test_index,
                                  sim_all,
                                  Sim_any, 0], object) #0 is there to avoid some numpy load and save errros
            np.save(self.split_file, save_data)
 
        return self.Train_index, self.Test_index, sim_all, Sim_any, self.split_file

        
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                      Splitting method dependend functions                         ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    
    
    def __init__(self):
        # Important: Only internal matters, no additional inputs possible
        raise AttributeError('Has to be overridden in actual method.')
        

    def split_data_method(self, Input_path, Input_T, 
                          Output_path, Output_T, 
                          Output_A, Output_T_E, Domain):
        # this function takes the given input and then creates a 
        # split according to a desied method
        # creates:
            # self.Train_index -    A 1D numpy including the samples IDs of the training set
            # self.Test_index -     A 1D numpy including the samples IDs of the test set
        raise AttributeError('Has to be overridden in actual method.')
        
    
    def get_similarity_method(self, Input_path, Input_T, 
                              Output_path, Output_T, 
                              Output_A, Output_T_E, Domain):
        
        # num_samples = len(Input_path)
        # num_dims = 2
        # num_vehicles = int(len(Input_path.columns) / num_dims)
        # num_timesteps = len(Input_path.iloc[0,0])
        # Input_path_array = np.zeros((num_samples, num_vehicles, num_timesteps, num_dims))
        # for i_sample in range(num_samples):
        #     for i_vehicle in range(num_vehicles):
        #         for i_dims in range(num_dims):
        #             i_dv = i_vehicle * num_dims + i_dims
        #             Input_path_array[i_sample, i_vehicle, :, i_dims] = Input_path.iloc[i_sample, i_dv]
        
        # Input_path_array = Input_path_array.astype('float32')
        # # it would be nice if rotation and eigenvectors could be used here, but this is likely to expensive
        # Weights_dim = (Input_path_array.ptp(axis = 2, keepdims = True)).max(0, keepdims = True) ** (-1)

        # try:
        #     Diff = Input_path_array[self.Test_index, np.newaxis,:] - Input_path_array[np.newaxis, self.Train_index,:]
        #     # get weights distance over dimensions
        #     Dist = np.linalg.norm(Diff * Weights_dim[np.newaxis], axis = 4)
        #     # get mean dist over time
        #     Dist = Dist.mean(3)
        # except np.core._exceptions._ArrayMemoryError:
        #     Dist = np.zeros((len(self.Test_index, self.Train_index, num_vehicles)), 'float32')
        #     for i_sample_test in self.Test_index:
        #         for i_sample_train in self.Train_index:
        #             diff = Input_path_array[i_sample_test] - Input_path_array[i_sample_test]
        #             dist = np.linalg.norm(diff * Weights_dim[0], axis = 2)
        #             Dist[i_sample_test, i_sample_train, :] = dist.mean(1)
        
        # # This is a fucking problem right now, maybe weigh difference indeed by influence on final outcome
        # Weights_vehicle = np.array([4,1,1,1,1]).reshape(1,1,-1)
        # Weights_vehicle = Weights_vehicle / Weights_vehicle.sum()
        
        # factor = 1
        
        # Similarity = np.sum(Weights_vehicle / (1 + factor * Dist), axis = 2)
        
        # partition = 0.01
        
        # num_partition = int(partition * len(self.Train_index))
        
        # Confidence = (Output_A[self.Test_index,np.newaxis] == Output_A[np.argsort(- Similarity, axis = 1)[:,:num_partition]]).mean(1)
        
        # Similarity_accept_to_accept = Similarity[Output_A[self.Test_index]][:,Output_A[self.Train_index]]
        
        # Similarity_accept_to_reject = Similarity[Output_A[self.Test_index]][:,np.invert(Output_A[self.Train_index])]
        
        # Similarity_reject_to_accept = Similarity[np.invert(Output_A[self.Test_index])][:,Output_A[self.Train_index]]
        
        # Similarity_reject_to_reject = Similarity[np.invert(Output_A[self.Test_index])][:,np.invert(Output_A[self.Train_index])]
        
        # Similarity_accept = Similarity_accept_to_accept.max(1) * (1 - )
        
        sim_all = 1
        Sim_any = np.ones(len(self.Test_index))
        return sim_all, Sim_any
        
    
    def get_name(self):
        # Returns a string with the name of the class
        # IMPORTANT: No '-' in the name
        raise AttributeError('Has to be overridden in actual method')
        
        
    
        



