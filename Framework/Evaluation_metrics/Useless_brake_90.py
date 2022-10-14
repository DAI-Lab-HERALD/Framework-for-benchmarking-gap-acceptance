import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from Useless_brake import Useless_brake

class Useless_brake_90(Useless_brake):
    def evaluate_prediction_method(self):
        Threshold = np.quantile(self.Output_A_pred[self.Output_A], 0.1)
        Result = np.mean(self.Output_A_pred[np.invert(self.Output_A)] < Threshold)
        return [Result]
    
    def get_name(self):
        return 'brake_90'