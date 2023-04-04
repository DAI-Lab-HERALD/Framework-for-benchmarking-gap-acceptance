import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class FDE(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Error = 0
        for i_sample in range(len(self.Output_path_pred)):
            (num_poss, _) = self.Output_path_pred.iloc[i_sample, 0].shape
            diff = np.zeros((num_poss, ))
            for j in range(len(self.Output_path.columns)):
                diff += (self.Output_path_pred.iloc[i_sample, j][:,-1] -
                         self.Output_path.iloc[i_sample, j][np.newaxis,-1]) ** 2
            diff = np.sqrt(diff)   
            
            Error += np.mean(diff)
        return [Error / len(self.Output_path_pred)]
    
    
    def get_output_type_class(self = None):
        return 'path'
    
    def get_t0_type():
        return 'col'
    
    def get_opt_goal():
        return 'minimize'
        
    def main_result_idx():
        return 0
    
    def get_name(self):
        return 'FDE'
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False