import pandas as pd
import numpy as np
from splitting_template import splitting_template





class Random_split_1(splitting_template):
    
    def __init__(self):
        self.test_part = 0.2
        
        
    def split_data_method(self, Input_path, Input_T, 
                          Output_path, Output_T, 
                          Output_A, Output_T_E, Domain):
        
        Index = np.arange(len(Output_A))
        Index_accepted = Index[Output_A]
        Index_rejected = Index[np.invert(Output_A)]
        
        num_train_accepted = int((1 - self.test_part) * len(Index_accepted))
        num_train_rejected = int((1 - self.test_part) * len(Index_rejected))
        
        np.random.seed(1)
        np.random.shuffle(Index_accepted)
        np.random.shuffle(Index_rejected)
        
        Index_accepted_train = Index_accepted[:num_train_accepted]
        Index_accepted_test = Index_accepted[num_train_accepted:]
        
        Index_rejected_train = Index_rejected[:num_train_rejected]
        Index_rejected_test = Index_rejected[num_train_rejected:]
        
        self.Train_index= np.sort(np.concatenate((Index_accepted_train, Index_rejected_train), axis = 0))
        self.Test_index= np.sort(np.concatenate((Index_accepted_test, Index_rejected_test), axis = 0))
    
    def get_name(self):
        return 'random_split_1'
        
        
        
    
        



