import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Accuracy(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        decision_steps = 51
        Threshold = np.linspace(0, 1, decision_steps)
        Accuracy = np.zeros(decision_steps)
        for i, threshold in enumerate(Threshold): 
            Output_A_thres = self.Output_A_pred > threshold
            
            Accuracy[i] = np.mean(self.Output_A == Output_A_thres)
        
        return [np.max(Accuracy)]
    
    
    def get_output_type_class(self = None):
        return 'binary'
    
    def get_t0_type():
        return 'start'
    
    def get_opt_goal():
        return 'maximize'
        
    def main_result_idx():
        return 0
    
    def get_name(self):
        return 'accuracy'
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False