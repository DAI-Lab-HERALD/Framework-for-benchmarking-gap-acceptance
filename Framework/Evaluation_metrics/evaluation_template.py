import pandas as pd
import numpy as np
import os


class evaluation_template():
    def __init__(self, Output_path, Output_T, Output_A, Output_T_E, Domain, splitter_save_file):
        # check if output type is an pandas array
        if not isinstance(Output_path, pd.core.frame.DataFrame):
            raise TypeError("Output path is expected to be a pandas dataframe.")
        
        # check if output has two columns (x and y position)
        name_test = [name[2:] for name in Output_path.columns]
        if not name_test == ['tar_x', 'tar_y']:
            raise TypeError("Output path does not fulfill requirements.")
        
        # check if number smaples matches
        self.num_samples = len(Output_path)
            
        # check if Output_T_train is a numpy vector
        if not (isinstance(Output_T, np.ndarray) and len(Output_T.shape) == 1):
            raise TypeError("Output_T is expected to be a numpy vector.")
        
        # check if the number of samples matches
        if not len(Output_T) == self.num_samples:
            raise TypeError("Output_T is expected to include same number of samples as Input.")
        
        # check if number timesteps matches
        for i in range(self.num_samples):                
            # check if time input consists out of tuples
            if not isinstance(Output_T[i], np.ndarray):
                raise TypeError("Output_T is expected to be consisting of arrays.") 
            
            test_length = len(Output_T[i])
            for j in range(2):
                # check if time input consists out of tuples
                if not isinstance(Output_path.iloc[i,j], np.ndarray):
                    raise TypeError("Output path is expected to be consisting of arrays.") 
                # test if input tuples have right length  
                length = len(Output_path.iloc[i,j])
                if test_length != length:
                    raise TypeError("Output path does not have a matching number of timesteps")    
               
        if not (isinstance(Output_A, np.ndarray) and len(Output_A.shape) == 1):
            raise TypeError("Output action is expected to be a numpy vector.")       
        # check if the number of samples matches
        if not len(Output_A) == self.num_samples:
            raise TypeError("Output action is expected to include same number of samples as Input.")
        
        # check if Output_T_train is a numpy vector
        if not (isinstance(Output_T_E, np.ndarray) and len(Output_T_E.shape) == 1):
            raise TypeError("Output_T_E is expected to be a numpy vector.")
        # check if number smaples matches
        if not len(Output_T_E) == self.num_samples:
            raise TypeError("Output_T_E is expected to include same number of samples as Input.") 
        # check if Output_T_train has only scalars
        for i in range(self.num_samples):
            if not np.isscalar(Output_T_E[i]):
                raise TypeError("For binary ouput, time is expected to be a scalar.")
            Output_T_E[i] = float(Output_T_E[i])
            
        self.Output_path = Output_path
        self.Output_T = Output_T
        self.Output_A = Output_A
        self.Output_T_E = Output_T_E
        self.Domain = Domain
        
        if self.requires_preprocessing():
            path = os.path.dirname(splitter_save_file)[:-6]
            file = os.path.basename(splitter_save_file)[:-4]
           
            test_file = (path +
                         'Metrics/' + 
                         file +
                         '-' + 
                         self.get_name() + 
                         '_weights' +
                         '.npy')
            
            if os.path.isfile(test_file):
                self.weights_saved = list(np.load(test_file, allow_pickle = True)[:-1])  
            else:
                self.setup_method() # output needs to be a list of components
                
                save_data = np.array(self.weights_saved + [0], object) #0 is there to avoid some numpy load and save errros
                np.save(test_file, save_data)
        
        
        
    def evaluate_prediction(self, Output_pred, model_save_file, create_plot_if_possible = False):
        # check output_data_train           
        if self.get_output_type_class() == 'path':
            [Output_path_pred] = Output_pred
            
            if not isinstance(Output_path_pred, pd.core.frame.DataFrame):
                raise TypeError("Output is expected to be a pandas dataframe.")
            
            if not len(Output_path_pred) == self.num_samples:
                raise TypeError("Predicted output has to have the same number of samples as true output.")
            
            
            
            # check if number timesteps matches
            for i in range(self.num_samples):                
                test_length = len(self.Output_T[i])
                
                for j in range(2):
                    # check if time input consists out of tuples
                    if not isinstance(Output_path_pred.iloc[i,j], np.ndarray):
                        raise TypeError("Output path prediciction is expected to be consisting of arrays.")
                    if j == 0:
                        number_poss = len(Output_path_pred.iloc[i,j])
                    else:
                        if number_poss != len(Output_path_pred.iloc[i,j]):
                            raise TypeError("Not all states have the same number of discrete samples")
                            
                for j in range(2):
                    for k in range(number_poss):
                        # test if input tuples have right length  
                        length = len(Output_path_pred.iloc[i,j][k])
                        if test_length != length:
                            raise TypeError("Output does not have a matching number of timesteps")    
            
            self.Output_path_pred = Output_path_pred
            
            
        elif self.get_output_type_class() == 'binary_and_time':
            [Output_A_pred, Output_T_E_pred] = Output_pred
            # check if output type is a numpy vector
            if not (isinstance(Output_A_pred, np.ndarray) and len(Output_A_pred.shape) == 1):
                raise TypeError("Predicted output decision is expected to be a numpy vector.")       
            # check if number smaples matches
            if not len(Output_A_pred) == self.num_samples:
                raise TypeError("Predicted output decision has to have the same number of samples as true output.") 

            # check if Output_T_train is a numpy vector
            if not (isinstance(Output_T_E_pred, np.ndarray) and len(Output_T_E_pred.shape) == 1):
                raise TypeError("Output_T_E_pred is expected to be a numpy vector.")
            # check if number smaples matches
            if not len(Output_T_E_pred) == self.num_samples:
                raise TypeError("Output_T_E_pred is expected to include same number of samples as true output.") 
            # check if output type matches (requires quatiles at [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            for i in range(self.num_samples):
                if len(Output_T_E_pred[i]) != 9:
                    raise TypeError("For preiction, time is expected to have nine entries for quantile values.")
                    
            self.Output_A_pred = Output_A_pred
            self.Output_T_E_pred = Output_T_E_pred
            
        
        else:
            [Output_A_pred] = Output_pred
            # check if output type is a numpy vector
            if not (isinstance(Output_A_pred, np.ndarray) and len(Output_A_pred.shape) == 1):
                raise TypeError("Output is expected to be a numpy vector.")       
            # check if number smaples matches
            if not len(Output_A_pred) == self.num_samples:
                raise TypeError("Predicted output has to have the same number of samples as true output.")
            
            self.Output_A_pred = Output_A_pred
            
        
        path = os.path.dirname(model_save_file)[:-6]
        file = os.path.basename(model_save_file)[:-4]
       
        test_file = (path +
                     'Metrics/' + 
                     file +
                     '-' + 
                     self.get_name() + 
                     '.npy')
        
        if os.path.isfile(test_file):
            results = list(np.load(test_file, allow_pickle = True)[:-1])
        else:
            results = self.evaluate_prediction_method() # output needs to be a list of components
            
            save_data = np.array(results + [0], object) #0 is there to avoid some numpy load and save errros
            np.save(test_file, save_data)
        
        if create_plot_if_possible:
            self.create_plot(results, test_file)
 
        return results
        
    
    def create_plot(self, results, test_file):
        # Function that visualizes result if possible
        if self.allows_plot():
            raise AttributeError('Has to be overridden in actual metric class.')
        else:
            pass
        
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                    Evaluation metric dependend functions                          ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    def setup_method(self):
        # Will do any preparation the method might require, like calculating
        # weights.
        # creates:
            # self.weights_saved -  The weights that were created for this metric,
            #                       will be in the form of a list
        raise AttributeError('Has to be overridden in actual metric class.')
        
        
    def evaluate_prediction_method(self):
        # Takes true outputs and corresponding predictions to calculate some
        # metric to evaluate a model
        raise AttributeError('Has to be overridden in actual metric class.')
        # return results # results is a list
        
    
    def get_output_type_class(self = None):
        # Should return 'binary', 'binary_and_time', 'path'
        raise AttributeError('Has to be overridden in actual metric class')

    def get_t0_type():
        # Should return 'start', 'crit', 'col_m'
        raise AttributeError('Has to be overridden in actual metric class')
        
    def get_name(self):
        # Returns a string with the name of the class
        # IMPORTANT: No '-' in the name
        raise AttributeError('Has to be overridden in actual metric class.')
        
    def requires_preprocessing(self):
        # Returns a boolean output, True if preprocesing of true output
        # data for the calculation of weights is required, which might be 
        # avoided in repeated cases
        raise AttributeError('Has to be overridden in actual metric class.')
        
    def allows_plot(self):
        # Returns a boolean output, True if a plot can be created, False if not.
        raise AttributeError('Has to be overridden in actual metric class.')
        
    
        