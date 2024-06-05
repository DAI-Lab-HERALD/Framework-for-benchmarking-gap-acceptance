import numpy as np
import pandas as pd
import os
import importlib
import sys
import matplotlib.pyplot as plt
from helper import *

plt.rcParams['text.usetex'] = True
# NOTE: Here still local path is assumed, so it might be useful
# to adapt that later
path = '/'.join(os.path.dirname(__file__).split('\\'))


def plot():
    plot_paths(path)


data_set_path = os.path.join(path, 'Data_sets', '')
if not data_set_path in sys.path:
    sys.path.insert(0, data_set_path)

metrics_path = os.path.join(path, 'Evaluation_metrics', '')
if not metrics_path in sys.path:
    sys.path.insert(0, metrics_path)
       
split_path = os.path.join(path, 'Splitting_methods', '')
if not split_path in sys.path:
    sys.path.insert(0, split_path)

model_path = os.path.join(path, 'Models', '')
if not model_path in sys.path:
    sys.path.insert(0, model_path)
    
#%% 
    
##### Select modules
Data_sets = ['HighD_lane_change', 'HighD_lane_change_intent', 'RounD_round_about', 'CoR_left_turns']
Metrics = ['Accuracy', 'Accuracy_col', 'ROC_curve', 'ROC_curve_col', 'ADE', 'ADE_k5', 'FDE', 'FDE_k5', 'Miss_rate', 'Miss_rate_col']
Data_params = [{'dt': 0.2, 'num_timesteps_in': 2}, {'dt': 0.2, 'num_timesteps_in': 10}] 
Splitters = ['Random_split', 'Critical_split']
Models = ['trajectron_salzmann', 'agent_yuan', 'logit_theofilatos', 'rt_mafi', 'db_xie', 'meta_khelfa']


##### Select method for D.T
model_to_path_name = 'trajectron_salzmann'
model_to_path_module = importlib.import_module(model_to_path_name)
model_class_to_path = getattr(model_to_path_module, model_to_path_name)

##### Prepare run
Results = np.zeros((len(Data_sets),
                    len(Metrics),
                    len(Data_params),
                    len(Splitters),
                    len(Models)),
                   object)

num_samples_path_pred = 100
create_plot_if_possible = False

for i, data_set_name in enumerate(Data_sets):
    # Get data set
    data_set_module = importlib.import_module(data_set_name)
    data_set_class = getattr(data_set_module, data_set_name)
    data_set = data_set_class(model_class_to_path = model_class_to_path, 
                              num_samples_path_pred = num_samples_path_pred)
    
    
    for j, metric_name in enumerate(Metrics):
        # Get metric used
        metric_module = importlib.import_module(metric_name)
        metric_class = getattr(metric_module, metric_name)
        # get the required output type for metric
        metric_type = metric_class.get_output_type_class()
        # Get the required t0 type
        metric_t0_type = metric_class.get_t0_type()
        
        
        for k, data_param in enumerate(Data_params):
            data_set.reset()
            # Select training data
            data = data_set.get_data(t0_type = metric_t0_type,
                                     exclude_post_crit = True,
                                     **data_param)
            
            # Check if a usable dataset was produced
            if data != None:
                [Input_path, Input_T, 
                  Output_path, Output_T, 
                  Output_A, Output_T_E,
                  Domain, data_set_save_file] = data
                for l, splitter_name in enumerate(Splitters):
                    # Get data set
                    splitter_module = importlib.import_module(splitter_name)
                    splitter_class = getattr(splitter_module, splitter_name)
                    splitter = splitter_class()
                    [Train_index, Test_index, 
                      Test_sim_all, Test_Sim_any,
                      splitter_save_file] = splitter.split_data(Input_path, Input_T, 
                                                                Output_path, Output_T, 
                                                                Output_A, Output_T_E,
                                                                Domain, data_set_save_file)
                                                               
                    # initialize estimator output
                    # all possible output data used to allow for calculation of
                    # more complex weighting methods
                    metric = metric_class(Output_path = Output_path.iloc[Test_index], 
                                          Output_T = Output_T[Test_index], 
                                          Output_A = Output_A[Test_index], 
                                          Output_T_E = Output_T_E[Test_index], 
                                          Domain = Domain.iloc[Test_index],
                                          splitter_save_file = splitter_save_file)
                    
                     
                    for m, model_name in enumerate(Models):
                        model_module = importlib.import_module(model_name)
                        model_class = getattr(model_module, model_name)        
                        model_type = model_class.get_output_type_class()
                        
                        print('')
                        print('Applying data-set ' + 
                              format(i + 1, '.0f').rjust(len(str(len(Data_sets)))) +
                              '/{}'.format(len(Data_sets)) + 
                              ' and metric ' + 
                              format(j + 1, '.0f').rjust(len(str(len(Metrics)))) +
                              '/{}'.format(len(Metrics)) + 
                              ' and input params ' + 
                              format(k + 1, '.0f').rjust(len(str(len(Data_params)))) +
                              '/{}'.format(len(Data_params)) +
                              ' and split ' + 
                              format(l + 1, '.0f').rjust(len(str(len(Splitters)))) +
                              '/{}'.format(len(Splitters)) +
                              ' to model ' + 
                              format(m + 1, '.0f').rjust(len(str(len(Models)))) +
                              '/{}'.format(len(Models)))
        
                        model = model_class(Input_path_train = Input_path.iloc[Train_index],
                                            Input_T_train = Input_T[Train_index],
                                            Output_path_train = Output_path.iloc[Train_index], 
                                            Output_T_train = Output_T[Train_index], 
                                            Output_A_train = Output_A[Train_index], 
                                            Output_T_E_train = Output_T_E[Train_index], 
                                            Domain_train = Domain.iloc[Train_index], 
                                            splitter_save_file = splitter_save_file,
                                            num_samples_path_pred = num_samples_path_pred)
                        
                        model_save_file = model.train()
                        
                        output = model.predict(Input_path_test = Input_path.iloc[Test_index], 
                                                Input_T_test = Input_T[Test_index],
                                                Output_T_test = Output_T[Test_index])
                
                        
                        if metric_type == 'path':        
                            # Transform model output if necessary
                            if model_type == 'path':
                                [Output_path_pred] = output
                            elif model_type == 'binary':
                                [Output_A_pred] = output
                                Output_T_E_pred = data_set.binary_to_time(Output_A_pred,
                                                                          Domain.iloc[Test_index],
                                                                          model_save_file, 
                                                                          splitter_save_file)
                                Output_path_pred = data_set.binary_and_time_to_path(Output_A_pred, 
                                                                                    Output_T_E_pred,
                                                                                    Domain.iloc[Test_index],
                                                                                    model_save_file, 
                                                                                    splitter_save_file)
                            else:
                                [Output_A_pred, Output_T_E_pred] = output
                                Output_path_pred = data_set.binary_and_time_to_path(Output_A_pred, 
                                                                                    Output_T_E_pred, 
                                                                                    Domain.iloc[Test_index],
                                                                                    model_save_file, 
                                                                                    splitter_save_file)
                                
                            metric_result = metric.evaluate_prediction(Output_pred = [Output_path_pred],
                                                                        model_save_file = model_save_file, 
                                                                        create_plot_if_possible = create_plot_if_possible)
                            
                        elif metric_type == 'binary':
                            # Transform model output if necessary
                            if model_type == 'path':
                                [Output_path_pred] = output
                                [Output_A_pred, _] = data_set.path_to_binary_and_time(Output_path_pred, 
                                                                                      Output_T[Test_index],
                                                                                      Domain.iloc[Test_index], 
                                                                                      model_save_file)
                            elif model_type == 'binary':
                                [Output_A_pred] = output
                            else:
                                [Output_A_pred, _] = output
                            
                            metric_result = metric.evaluate_prediction(Output_pred = [Output_A_pred],
                                                                        model_save_file = model_save_file, 
                                                                        create_plot_if_possible = create_plot_if_possible)
                            
                        else:
                            # Transform model output if necessary
                            if model_type == 'path':
                                [Output_path_pred] = output
                                [Output_A_pred, 
                                  Output_T_E_pred] = data_set.path_to_binary_and_time(Output_path_pred, 
                                                                                      Output_T[Test_index], 
                                                                                      Domain.iloc[Test_index],
                                                                                      model_save_file)
                            elif model_type == 'binary':
                                [Output_A_pred] = output
                                Output_T_E_pred = data_set.binary_to_time(Output_A_pred, 
                                                                          Domain.iloc[Test_index],
                                                                          model_save_file, 
                                                                          splitter_save_file)
                            else:
                                [Output_A_pred, Output_T_E_pred] = output
                            
                            metric_result = metric.evaluate_prediction(Output_pred = [Output_A_pred, 
                                                                                      Output_T_E_pred],
                                                                        model_save_file = model_save_file, 
                                                                        create_plot_if_possible = create_plot_if_possible)
                            
                        Results[i,j,k,l,m] = metric_result
      
# plot()
R = np.zeros_like(Results, dtype = type(Results[0,0,0,0,0][0]))
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        for k in range(R.shape[2]):
            for l in range(R.shape[3]):
                for m in range(R.shape[4]):
                    R[i,j,k,l,m] = Results[i,j,k,l,m][-1]