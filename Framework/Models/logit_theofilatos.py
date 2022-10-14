import numpy as np
import pandas as pd
from model_template import model_template
from sklearn.linear_model import LogisticRegression as LR

class logit_theofilatos(model_template):
    
    def setup_method(self):
        self.timesteps = max([len(T) for T in self.Input_T_train])
        
        self.Parameter = np.zeros(len(self.input_names_train) * self.timesteps + 1)

    def train_method(self, l2_regulization = 0.01):
        # Multiple timesteps have to be flattened
        X_help = self.Input_path_train.to_numpy()
        X = np.ones([X_help.shape[0], X_help.shape[1], self.timesteps]) * np.nan
        for i in range(len(X)):
            for j in range(X_help.shape[1]):
                X[i, j, (self.timesteps - len(X_help[i,j])):] =  X_help[i,j]
        X = X.reshape(X.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        self.mean = np.nanmean(X, axis = 0, keepdims = True)
        X = X - self.mean
        X[np.isnan(X)] = 0
        
        # get parameters of linear model
        Log_reg = LR(C = 1/(l2_regulization+1e-7), max_iter = 100000)
        param = Log_reg.fit(X, self.Output_A_train)
        self.Parameter = np.concatenate((param.coef_[0], param.intercept_))
        
        self.weights_saved = [self.mean, self.Parameter]
        
        
    def load_method(self, l2_regulization = 0):
        [self.mean, self.Parameter] = self.weights_saved
        
    def predict_method(self):
        # Multiple timesteps have to be flattened
        X_help = self.Input_path_test.to_numpy()
        X = np.ones([X_help.shape[0], X_help.shape[1], self.timesteps])*np.nan
        for i in range(len(X)):
            for j in range(X_help.shape[1]):
                X[i, j, max(0,(self.timesteps - len(X_help[i,j]))):] =  X_help[i,j][max(0,(len(X_help[i,j]) - self.timesteps)):]
        X = X.reshape(X.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        X = X - self.mean
        # set missing values to zero
        X[np.isnan(X)] = 0
        
        
        Linear = np.sum(np.array(X * self.Parameter[:-1]), axis=1) + self.Parameter[-1]
        #prevent overflow
        Prob = np.zeros_like(Linear)
        pos = Linear > 0
        neg = np.invert(pos)
        Prob[pos] = 1 / (1 + np.exp(-Linear[pos]))
        Prob[neg] = np.exp(Linear[neg]) / (np.exp(Linear[neg]) + 1)
        return [Prob]

    def check_input_names_method(self, names, train = True):
        if all(names == self.input_names_train):
            return True
        else:
            return False
     
    
    def get_output_type_class():
        # Logit model only produces binary outputs
        return 'binary'
    
    def get_name(self):
        return 'logit'
        

        
        
        
        
        
    
        
        
        