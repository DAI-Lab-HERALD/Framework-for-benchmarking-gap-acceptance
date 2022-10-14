import numpy as np
import pandas as pd
import os
import importlib
from model_template import model_template
from sklearn.linear_model import LogisticRegression as LR

class meta_khelfa(model_template):
    
    def setup_method(self):
        self.timesteps = max([len(T) for T in self.Input_T_train])
        

    def train_method(self, l2_regulization = 0.01):
        
        data_set_name = os.path.basename(self.model_file)[:-4].split('-')[0]
        data_set_class = getattr(importlib.import_module(data_set_name), data_set_name)
        data_set = data_set_class(model_class_to_path = None, 
                                  num_samples_path_pred = self.num_samples_path_pred)
        data_set.dt = float(os.path.basename(self.model_file)[:-4].split('-')[1]
                            .split('dt=')[1].split('_')[0])
        
        
        
        
        
        
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
        
        
        path = os.path.dirname(self.model_file)
        file = os.path.basename(self.model_file)[:-9]
        
        self.model_param_files = [f[:-4].split('-')[-1] for f in os.listdir(path) 
                                  if f[:len(file)] == file and 'prediction' not in f and self.get_name() not in f]
        # for each model, load model
        
        model_avail = [f[:-3] for f in os.listdir(path[:-14] + 'Models/') 
                       if f[-3:] == '.py' and f not in ['model_template.py', 'meta_khelfa.py']]
            
        
        model_avail_name = [getattr(importlib.import_module(model), model).get_name(None) 
                            for model in model_avail]
        
        # Load model by model:
        for model_param_file in self.model_param_files:
            try:
                i = model_avail_name.index(model_param_file)
            except:
                raise AttributeError("Somehow, a trained model has no model.py file")
                
            model = model_avail[i]
            
            model_class = getattr(importlib.import_module(model), model)
            model_type = model_class.get_output_type_class()
            
            model = model_class(self.Input_path_train,
                                self.Input_T_train,
                                self.Output_path_train, 
                                self.Output_T_train, 
                                self.Output_A_train, 
                                self.Output_T_E_train, 
                                self.Domain_train, 
                                path[:-6] + 'Splitting/' + file + '.npy',
                                num_samples_path_pred = self.num_samples_path_pred)
            # Load trained model
            _ = model.train()
            output = model.predict(Input_path_test = self.Input_path_train, 
                                   Input_T_test = self.Input_T_train,
                                   Output_T_test = self.Output_T_train,
                                   save_results = False)
            
            if model_type == 'path':
                [Output_path_pred] = output
                [Output_A_pred, _] = data_set.path_to_binary_and_time(Output_path_pred, 
                                                                      self.Output_T_train, 
                                                                      self.Domain_train, 
                                                                      model_save_file = None,
                                                                      save_results = False)
            elif model_type == 'binary':
                [Output_A_pred] = output
            else:
                [Output_A_pred, _] = output
                
            X = np.concatenate((X, Output_A_pred[:,np.newaxis]), 1)
            

        # get parameters of linear model
        Log_reg = LR(C = 1/(l2_regulization+1e-7), max_iter = 1000000)
        param = Log_reg.fit(X, self.Output_A_train)
        self.Parameter = np.concatenate((param.coef_[0], param.intercept_))
        
        self.weights_saved = [self.mean, self.Parameter, self.model_param_files]
        
        
    def load_method(self, l2_regulization = 0):
        [self.mean, self.Parameter, self.model_param_files] = self.weights_saved
        
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
        
        path = os.path.dirname(self.file_name)
        file = os.path.basename(self.file_name)
        
        model_pred_files = [f[:-4].split('-')[-1] for f in os.listdir(path) if f[:len(file)] == file and 'prediction.npy' in f]
        # for each model, load model
        
        assert all([mod + '_prediction' in model_pred_files for mod in self.model_param_files]), "Some models included in meta heuristic not found"
        
        model_avail = [f[:-3] for f in os.listdir(path[:-14] + 'Models/') 
                       if f[-3:] == '.py' and f not in ['model_template.py', 'meta_khelfa.py']]
            
        
        model_avail_name = [getattr(importlib.import_module(model), model).get_name(None) 
                            for model in model_avail]
        
        # Load model by model:
        for model_param_file in self.model_param_files:
            try:
                i = model_avail_name.index(model_param_file)
            except:
                raise AttributeError("Somehow, a trained model has no model.py file")
                
            model_type = getattr(importlib.import_module(model_avail[i]), model_avail[i]).get_output_type_class()
            
            if model_type == 'path':
                model_pred_file = (self.file_name + '-' + model_param_file + '_prediction_T2bt.npy')
                output = list(np.load(model_pred_file, allow_pickle = True)[:-1])
                [Output_A_pred, _] = output
                
            elif model_type == 'binary':
                model_pred_file = (self.file_name + '-' + model_param_file + '_prediction.npy')
                output = list(np.load(model_pred_file, allow_pickle = True)[:-1])
                [Output_A_pred] = output
            else:
                model_pred_file = (self.file_name + '-' + model_param_file + '_prediction.npy')
                output = list(np.load(model_pred_file, allow_pickle = True)[:-1])
                [Output_A_pred, _] = output
                
            X = np.concatenate((X, Output_A_pred[:,np.newaxis]), 1)
        
        
        
        
        
        
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
        return 'meta'
        

        
        
        
        
        
    
        
        
        