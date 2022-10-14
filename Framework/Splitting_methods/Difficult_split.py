import pandas as pd
import numpy as np
import os
import importlib
from splitting_template import splitting_template



class Difficult_split(splitting_template):
    
    def __init__(self):
        self.test_part = 0.2
        
        
    def split_data_method(self, Input_path, Input_T, 
                          Output_path, Output_T, 
                          Output_A, Output_T_E, Domain):
        
        ### REdo
        logit_model_module = importlib.import_module('logit_theofilatos')
        logit_model_class = getattr(logit_model_module, 'logit_theofilatos')
        
        logit_model = logit_model_class(Input_path_train = Input_path,
                                        Input_T_train = Input_T,
                                        Output_path_train = Output_path, 
                                        Output_T_train = Output_T, 
                                        Output_A_train = Output_A, 
                                        Output_T_E_train = Output_T_E, 
                                        Domain_train = Domain, 
                                        splitter_save_file = None)
        
        logit_model.train()
        
        output = logit_model.predict(Input_path_test = Input_path, 
                                     Input_T_test = Input_T,
                                     Output_T_test = Output_T,
                                     num_samples_path_pred = 1,
                                     save_results = False) # Does not matter, is not a path predicter
        
        Prob = output[0]
        
        Index_accepted = np.where(Output_A)[0]
        Index_rejected = np.where(np.invert(Output_A))[0]
        
        Prob_accepted = Prob[Index_accepted]
        Prob_rejected = Prob[Index_rejected]
        
        num_test_accepted = int(self.test_part * len(Index_accepted))
        num_test_rejected = int(self.test_part * len(Index_rejected))
        
        Sorted_accepted = np.argsort(Prob_accepted)
        Sorted_rejected = np.argsort(- Prob_rejected)
        
        Index_accepted_test = Index_accepted[Sorted_accepted][:num_test_accepted]
        Index_accepted_train = Index_accepted[Sorted_accepted][num_test_accepted:]
        
        Index_rejected_test = Index_rejected[Sorted_rejected][:num_test_rejected]
        Index_rejected_train = Index_rejected[Sorted_rejected][num_test_rejected:]
        
        self.Train_index= np.sort(np.concatenate((Index_accepted_train, Index_rejected_train), axis = 0))
        self.Test_index= np.sort(np.concatenate((Index_accepted_test, Index_rejected_test), axis = 0))
        
    def get_name(self):
        return 'difficult_split'
        
        
    
        



