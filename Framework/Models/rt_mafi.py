import numpy as np
import pandas as pd
from model_template import model_template
from sklearn.ensemble import RandomForestRegressor as RF

class rt_mafi(model_template):
    
    def setup_method(self):
        self.timesteps = max([len(T) for T in self.Input_T_train])
        
        self.model = None

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
        
        # get parameters of random_forest_model
        Feature_ratio = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
        Number_features = np.unique(np.round(np.array(Feature_ratio) * X.shape[1])).astype(int)
        Number_features = Number_features[Number_features > 0]
        
        Number_trees = [2,5,10,20,50,100]
        best_loss = None
        
        print("Random forest")
        print("start training:")
        for number_features in Number_features:
            for number_trees in Number_trees:
                print("Train with {} random trees using {}/{} features"
                      .format(number_trees, number_features, X.shape[1]))
                
                test_model = RF(n_estimators = number_trees, max_features = number_features)
                test_loss = np.zeros(10)
                
                length_fold = int(np.floor(len(X)/10))
                length = 10 * length_fold
                Index = np.arange(length)
                np.random.shuffle(Index)
                
                for i in range(10):
                    Index_test = Index[i * length_fold:(i + 1) * length_fold]
                    Index_train = np.concatenate((Index[:i * length_fold], Index[(i + 1) * length_fold:]))
                    
                    test_model = test_model.fit(X[Index_train],self.Output_A_train[Index_train])
                    
                    Output_test = self.Output_A_train[Index_test]
                    Output_test_pred = test_model.predict(X[Index_test])
                    
                    # Get output value
                    test_loss[i] = np.mean((Output_test-Output_test_pred)**2)
                    
                # Do ten fold crossvariation
                test_loss = np.mean(test_loss)
                
                if best_loss == None or test_loss < best_loss:
                    best_loss = test_loss
                    self.model = test_model
                    self.number_features = number_features
                    self.number_trees = number_trees
                    
                print("MSE: {:0.3f} (current best: {:0.3f})".format(test_loss,best_loss))
        self.model = RF(n_estimators = self.number_trees, max_features = self.number_features)
        self.model.fit(X, self.Output_A_train) 
        
        self.weights_saved = [self.mean, self.number_features, self.number_trees]
        
        
    def load_method(self, l2_regulization = 0):
        [self.mean, self.number_features, self.number_trees] = self.weights_saved
        
        self.model = RF(n_estimators = self.number_trees, max_features = self.number_features)
        
        X_help = self.Input_path_train.to_numpy()
        X = np.ones([X_help.shape[0], X_help.shape[1], self.timesteps]) * np.nan
        for i in range(len(X)):
            for j in range(X_help.shape[1]):
                X[i, j, (self.timesteps - len(X_help[i,j])):] =  X_help[i,j]
        X = X.reshape(X.shape[0], -1)
        X = X - self.mean
        self.model.fit(X, self.Output_A_train) 
        # set parameters
        
    def predict_method(self):
        # Multiple timesteps have to be flattened
        X_help = self.Input_path_test.to_numpy()
        X = np.ones([X_help.shape[0], X_help.shape[1], self.timesteps])*np.nan
        for i in range(len(X)):
            for j in range(X_help.shape[1]):
                X[i, j, max(0,(self.timesteps - len(X_help[i,j]))):] =  X_help[i,j][max(0,(len(X_help[i,j]) - self.timesteps)):]
        X = X.reshape(X.shape[0], -1)
        X = X - self.mean
        # set missing values to zero
        X[np.isnan(X)] = 0
        
        # make prediction
        Prob = self.model.predict(X)
        
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
        return 'random_forest'
        

        
        
        
        
        
    
        
        
        