import pandas as pd
import numpy as np
from splitting_template import splitting_template





class No_split(splitting_template):
    
    def __init__(self):
        pass
        
        
    def split_data_method(self, Input_path, Input_T, 
                          Output_path, Output_T, 
                          Output_A, Output_T_E, Domain):
        self.Train_index = np.arange(len(Input_path))
        self.Test_index = np.arange(len(Input_path))
        
    
    
    def get_name(self):
        return 'no_split'
        
        
        
    
        



