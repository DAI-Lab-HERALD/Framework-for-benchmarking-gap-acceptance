import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Useless_brake(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Threshold = np.min(self.Output_A_pred[self.Output_A])
        Result = np.mean(self.Output_A_pred[np.invert(self.Output_A)] < Threshold)
        return [Result]
    
    
    def get_output_type_class(self = None):
        return 'binary'
    
    def get_t0_type():
        return 'crit'
    
    def get_opt_goal():
        return 'maximize'
        
    def main_result_idx():
        return 0
    
    def get_name(self):
        return 'brake'
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False