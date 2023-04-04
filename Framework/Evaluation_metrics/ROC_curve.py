import numpy as np
import pandas as pd
import os
from evaluation_template import evaluation_template 
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

class ROC_curve(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self, plot = False):
        decision_steps = 101
        Threshold = np.linspace(0, 1, decision_steps)
        TPR = np.zeros(decision_steps + 2)
        FPR = np.zeros(decision_steps + 2)
        for i, threshold in enumerate(Threshold): 
            ii = i + 1
            Output_A_thres = self.Output_A_pred > threshold
            TP = np.sum(Output_A_thres[self.Output_A == True ] == True)
            FP = np.sum(Output_A_thres[self.Output_A == False] == True)
            FN = np.sum(Output_A_thres[self.Output_A == True ] == False)
            TN = np.sum(Output_A_thres[self.Output_A == False] == False)
            
            TPR[ii] = TP / (TP+FN)
            FPR[ii] = FP / (TN+FP)
        
        #For AUC, we can expect that FPR is always declining
        
                
        FPR[0] = 1
        TPR[0] = 1
        
        auc = np.sum((FPR[:-1] - FPR[1:]) * 0.5 * (TPR[:-1] + TPR[1:]))
        return [FPR, TPR, auc]
    
    def create_plot(self, results, test_file):
        file = os.path.basename(test_file)[:-4]
        [data_set, data_set_values, split_method, model, _] = file.split('-')
        
        split_method = split_method[:-6] # no, random, critical
        data_set_values = data_set_values[9:-1].replace('_', ', ')
        model = model.replace('_', ' ')
        
        [FPR, TPR, auc] = results
        
        plt.figure(figsize = (7,7))
        plt.plot([0,1], [0,1], '--k')
        plt.plot(FPR, TPR, 'r')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.gca().set_aspect('equal', adjustable='box') 
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        
        title = (r'\textbf{ROC curve} with AUC=' + '{:0.3f}'.format(auc) + 
                 r'\\Data set: ' + data_set + 
                 r' with ' + data_set_values + 
                 r'\\Split: ' + split_method + r' splitting between testing and trainig set' + 
                 r'\\Model: ' + model)
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
        
    
    def get_output_type_class(self = None):
        return 'binary' 
    
    def get_t0_type():
        return 'start'
    
    def get_opt_goal():
        return 'maximize'
        
    def main_result_idx():
        return 2
    
    def get_name(self):
        return 'roc_curve'
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True