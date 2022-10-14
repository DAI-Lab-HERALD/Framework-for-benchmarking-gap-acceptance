import pandas as pd
import numpy as np
import os




class model_template():
    
    def __init__(self, Input_path_train, Input_T_train,
                 Output_path_train, Output_T_train, 
                 Output_A_train, Output_T_E_train, 
                 Domain_train, splitter_save_file, num_samples_path_pred): 
        
        if not isinstance(Input_path_train, pd.core.frame.DataFrame):
            raise TypeError("Input path is expected to be a pandas dataframe.")
        self.input_names_train = Input_path_train.columns
        # check if required input exists
        if not self.check_input_names(Input_path_train.columns, train = True):
            raise TypeError("Input does not fulfill model requirements.") 
        
        # check if number smaples matches
        self.num_samples_train = len(Input_path_train)
            
        # check if Input_T_train is a numpy vector
        if not (isinstance(Input_T_train, np.ndarray) and len(Input_T_train.shape) == 1):
            raise TypeError("Input_T is expected to be a numpy vector.")
        
        # check if the number of samples matches
        if not len(Input_T_train) == self.num_samples_train:
            raise TypeError("Input_T is expected to include same number of samples as Input.")
        
        # check if number timesteps matches
        for i in range(self.num_samples_train):                
            # check if time input consists out of tuples
            if not isinstance(Input_T_train[i], np.ndarray):
                raise TypeError("Input_T is expected to be consisting of arrays.") 
            
            test_length = len(Input_T_train[i])
            for j in range(len(Input_path_train.columns)):
                # check if time input consists out of tuples
                if not isinstance(Input_path_train.iloc[i,j], np.ndarray):
                    raise TypeError("Input path is expected to be consisting of arrays.") 
                # test if input tuples have right length  
                length = len(Input_path_train.iloc[i,j])
                if test_length != length:
                    raise TypeError("Input path does not have a matching number of timesteps")  
        
        
        if not isinstance(Output_path_train, pd.core.frame.DataFrame):
            raise TypeError("Output path is expected to be a pandas dataframe.")
        
        # check if output has two columns (x and y position)
        if not all(Output_path_train.columns == self.input_names_train[2:4]):
            raise TypeError("Output path does not fulfill requirements.")
        
        # check if the number of samples matches
        if not len(Output_path_train) == self.num_samples_train:
            raise TypeError("Output path is expected to include same number of samples as Input.")
            
        # check if Output_T_train is a numpy vector
        if not (isinstance(Output_T_train, np.ndarray) and len(Output_T_train.shape) == 1):
            raise TypeError("Output_T is expected to be a numpy vector.")
        
        # check if the number of samples matches
        if not len(Output_T_train) == self.num_samples_train:
            raise TypeError("Output_T is expected to include same number of samples as Input.")
        
        # check if number timesteps matches
        for i in range(self.num_samples_train):                
            # check if time input consists out of tuples
            if not isinstance(Output_T_train[i], np.ndarray):
                raise TypeError("Output_T is expected to be consisting of arrays.") 
            
            test_length = len(Output_T_train[i])
            for j in range(2):
                # check if time input consists out of tuples
                if not isinstance(Output_path_train.iloc[i,j], np.ndarray):
                    raise TypeError("Output path is expected to be consisting of arrays.") 
                # test if input tuples have right length  
                length = len(Output_path_train.iloc[i,j])
                if test_length != length:
                    raise TypeError("Output path does not have a matching number of timesteps")    
               
        if not (isinstance(Output_A_train, np.ndarray) and len(Output_A_train.shape) == 1):
            raise TypeError("Output action is expected to be a numpy vector.")       
        # check if the number of samples matches
        if not len(Output_A_train) == self.num_samples_train:
            raise TypeError("Output action is expected to include same number of samples as Input.")
        
        # check if Output_T_train is a numpy vector
        if not (isinstance(Output_T_E_train, np.ndarray) and len(Output_T_E_train.shape) == 1):
            raise TypeError("Output_T_E is expected to be a numpy vector.")
        # check if number smaples matches
        if not len(Output_T_E_train) == self.num_samples_train:
            raise TypeError("Output_T_E is expected to include same number of samples as Input.") 
        # check if Output_T_train has only scalars
        for i in range(self.num_samples_train):
            if not np.isscalar(Output_T_E_train[i]):
                raise TypeError("For binary ouput, time is expected to be a scalar.")
            Output_T_E_train[i] = float(Output_T_E_train[i])
         
        if not (isinstance(Output_T_E_train, np.ndarray) and len(Output_T_E_train.shape) == 1):
            raise TypeError("Output_T_E is expected to be a numpy vector.")
        # check if number smaples matches
        if not len(Output_T_E_train) == self.num_samples_train:
            raise TypeError("Output_T_E is expected to include same number of samples as Input.")  
            
        if type(num_samples_path_pred) != type(0):
            raise TypeError("num_samples_path_pred should be an integer")

        
        self.Input_path_train = Input_path_train
        self.Input_T_train = Input_T_train
        self.Output_path_train = Output_path_train
        self.Output_T_train = Output_T_train
        self.Output_A_train = Output_A_train
        self.Output_T_E_train = Output_T_E_train
        self.Domain_train = Domain_train
        self.num_samples_path_pred = max(1, num_samples_path_pred)
        
        if splitter_save_file == None:
            self.file_name = None
            
        elif os.path.dirname(splitter_save_file)[-9:] == 'Splitting':
            # This is then used for transformation in a data_set, so independent of splitting method
            self.is_data_transformer = False
            path = os.path.dirname(splitter_save_file)[:-9]
            file = os.path.basename(splitter_save_file)[:-4]
     
            self.file_name = (path + 'Models/' + file)
        elif os.path.dirname(splitter_save_file)[-4:] == 'Data':
            self.is_data_transformer = True
            path = os.path.dirname(splitter_save_file)[:-4]
            file = os.path.basename(splitter_save_file)[:-4]
     
            self.file_name = (path + 'Data/' + file)
        
        else:
            raise AttributeError("Unusable save file given.")
        self.setup_method()
        
        # Set trianed to flase, this prevents a prediction on an untrained model
        self.trained = False
        
    
    def train(self):
        if self.file_name == None:
            self.train_method() 
            self.model_file = None
        else:
            self.model_file = (self.file_name +
                         '-' + 
                         self.get_name() + 
                         '.npy')
            
            if os.path.isfile(self.model_file):
                self.weights_saved = list(np.load(self.model_file, allow_pickle = True)[:-1])
                self.load_method()
            
            else:
                # creates /overrides self.weights_saved
                self.train_method() 
                #0 is there to avoid some numpy load and save errros
                save_data = np.array(self.weights_saved + [0], object) 
                np.save(self.model_file, save_data)
           
        self.trained = True
        
        return self.model_file
        
        
        
    def predict(self, Input_path_test, Input_T_test, Output_T_test, save_results = True):
        
        # check if input fulfills requirements
        if self.trained == False:
            raise AttributeError('Model has to be trained before prediction.')
        
        
        # Test if input data is a pandas dataframe (needed for input names)
        if not isinstance(Input_path_test, pd.core.frame.DataFrame):
            raise TypeError("Input is expected to be a pandas dataframe.")
        
        # save number of training samples
        self.num_test_samples = len(Input_path_test)

        # check if required input exists
        if not self.check_input_names(Input_path_test.columns, train = False):
            raise TypeError("Input does not fulfill model requirements.")
        
        # check if Input_T_train is a numpy vector
        if not (isinstance(Input_T_test, np.ndarray) and len(Input_T_test.shape) == 1):
            raise TypeError("Input_T_test is expected to be a numpy vector.")
        
        # check if the number of samples matches
        if not len(Input_T_test) == self.num_test_samples:
            raise TypeError("Input_T_test is expected to include same number of samples as Input.")
        
        # check if number timesteps matches
        for i in range(self.num_test_samples):            
            # check if time input consists out of tuples
            if not isinstance(Input_T_test[i], np.ndarray):
                raise TypeError("Input_T_test is expected to be consisting of arrays.") 
            if not isinstance(Output_T_test[i], np.ndarray):
                raise TypeError("Output_T_test is expected to be consisting of arrays.") 
            
            test_length = len(Input_T_test[i])
            for j in range(len(Input_path_test.columns)):
                # check if time input consists out of tuples
                if not isinstance(Input_path_test.iloc[i,j], np.ndarray):
                    raise TypeError("Input is expected to be consisting of arrays.") 
                # test if input tuples have right length  
                length = len(Input_path_test.iloc[i,j])
                if test_length != length:
                    raise TypeError("Input does not have a matching number of timesteps")

        
        self.Input_path_test = Input_path_test
        self.Input_T_test = Input_T_test
        self.Output_T_test = Output_T_test
        
        # perform prediction
        if save_results:
            test_file = (self.file_name +
                         '-' + 
                         self.get_name() + 
                         '_prediction' + 
                         '.npy')
            
            if os.path.isfile(test_file):
                output = list(np.load(test_file, allow_pickle = True)[:-1])
            else:
                output = self.predict_method() # output needs to be a list of components
                
                save_data = np.array(output + [0], object) #0 is there to avoid some numpy load and save errros
                np.save(test_file, save_data)
        else:
            output = self.predict_method()
        return output
        
        
    
    def check_input_names(self, names, train = True):
        # test if all required entries are there
        names_test = [name[2:] for name in names[:4]]
        if not names_test == ['ego_x', 'ego_y', 'tar_x', 'tar_y']:
            return False
        if len(names) > 4:
            for i in range(4, len(names), 2):
                if names[i][:-2] != names[i + 1][:-2]:
                    return False
        return self.check_input_names_method(names, train)
        
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                            Model dependend functions                              ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    def setup_method(self):
        # setsup the model, but only use the less expensive calculations.
        # Anything more expensive, especially if only needed for training,
        # should be done in train_method or load_method instead.
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def train_method(self):
        # Uses input data to train the model, saving the weights needed to fully
        # describe the trained model in self.weight_saved
        raise AttributeError('Has to be overridden in actual model.')
        
    
    def load_method(self):
        # Builds the models using loaded weights in self.weights_saved
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def predict_method(self):
        # takes test input and uses that to predict the output
        raise AttributeError('Has to be overridden in actual model.')
        return output

        
    def check_input_names_method(self, names, train = True):
        # check if output has two columns (x and y position)
        raise AttributeError('Has to be overridden in actual data-set class')
            
    def get_output_type_class():
        # Should return 'binary', 'binary_and_time', 'path'
        # the same as above, only this time callable from class
        raise AttributeError('Has to be overridden in actual model.')
        
    def get_name(self):
        # Returns a string with the name of the class
        # IMPORTANT: No '-' in the name
        raise AttributeError('Has to be overridden in actual data-set class.')