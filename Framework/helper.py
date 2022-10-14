import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import shutil
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import time

plt.rcParams['text.usetex'] = True

def plot_paths(path):
    plt.close('all')
    time.sleep(0.1)
    
    
    #%% Determine model
    dir_path = path + '/model_options'
    path_model = path + '/results/models'
    files = [f for f in os.listdir(path_model) 
             if 'prediction' not in f]
    
    for file in files:
        source = path_model + '/' + file
        target = dir_path + '/' + file
        if not os.path.isfile(target):
            print('copied')
            shutil.copyfile(source, target)
    
    root = tk.Tk()
    time.sleep(0.1)
    file_path = filedialog.askopenfilename(initialdir = dir_path)
    root.withdraw()
    
    [data_set_name, data_params, split, model] = os.path.basename(file_path[:-4]).split('-')
    data_params_list = data_params[9:-1].split('_')

    
    #%% Load input data
    data_file = path + '/results/data/' + data_set_name + '-' + data_params + '.npy'
    [Input_path, 
     Input_T, 
     Output_path, 
     Output_T, 
     Output_A, 
     Output_T_E, 
     Domain, _] = np.load(data_file, allow_pickle = True)
   
    
    split_file = path + '/results/splitting/' + data_set_name + '-' + data_params + '-' + split + '.npy'
    [_, Test_index, _, _, _] = np.load(split_file, allow_pickle = True)
    
    Input_path = Input_path.iloc[Test_index]
    Output_path = Output_path.iloc[Test_index]
    Input_T = Input_T[Test_index]
    Output_T = Output_T[Test_index]
    Output_A = Output_A[Test_index]
    Output_T_E = Output_T_E[Test_index]
    Domain = Domain.iloc[Test_index]
    Index = np.arange(len(Input_path))
    
    print('------------------------------------------------------------------')
    print('There are {} accepted gaps and {} rejected gaps available.'.format(np.sum(Output_A), np.sum(np.invert(Output_A))))
    print('Select case (type A for accepted gaps or R for rejected gaps): ')
    case = input()
    while case not in ['A', 'R']:
        print('This answer was not accepted.')
        print('Select case (type A for accepted gaps or R for rejected gaps): ')
        case = input()
    
    print('')
    
    if case == 'A':
        print('Type a number between in {1, ..., ' + str(np.sum(Output_A)) + '} to select the example: ')
        try:
            number = int(input())
        except:
            number = 0
        while number < 1 or number > np.sum(Output_A):
            print('This answer was not accepted.')
            print('Type a number between in {1, ..., ' + str(np.sum(Output_A)) + '} to select the example: ')
            try:
                number = int(input())
            except:
                number = 0
        print('')
        
        ind = Index[Output_A][number - 1]
        strv = 't_A'
        
    else:
        print('Type a number between in {1, ..., ' + str(np.sum(np.invert(Output_A))) + '} to select the example: ')
        try:
            number = int(input())
        except:
            number = 0
        while number < 1 or number > np.sum(np.invert(Output_A)):
            print('This answer was not accepted.')
            print('Type a number between in {1, ..., ' + str(np.sum(np.invert(Output_A))) + '} to select the example: ')
            try:
                number = int(input())
            except:
                number = 0
        print('')
        
        ind = Index[np.invert(Output_A)][number - 1]
        strv = 't_C'
    
    input_path = Input_path.iloc[ind]
    output_path = Output_path.iloc[ind]
    input_T = Input_T[ind]
    output_T = Output_T[ind]
    output_A = Output_A[ind]
    output_T_E = Output_T_E[ind]
    domain = Domain.iloc[ind]
    
    #%% Load map
    
    data_set_class = getattr(importlib.import_module(data_set_name), data_set_name)
    
    # Load line segments of data_set 
    map_lines_solid, map_lines_dashed = data_set_class.provide_map_drawing(self = None, domain = domain)
    
    bcdict = {'red': ((0, 0.0, 0.0),
                      (1, 0.0, 0.0)),
           'green': ((0, 0.0, 0.0),
                     (1, 0.0, 0.0)),
           'blue': ((0, 0.0, 0.0),
                    (1, 0.0, 0.0))}

    cmap = LinearSegmentedColormap('custom_cmap_l', bcdict)
    
    map_solid = LineCollection(map_lines_solid, cmap=cmap, linewidths=2, linestyle = 'solid')
    map_solid_colors = list(np.ones(len(map_lines_solid)))
    map_solid.set_array(np.asarray(map_solid_colors))
    
    map_dashed = LineCollection(map_lines_dashed, cmap=cmap, linewidths=1, linestyle = 'dashed')
    map_dashed_colors = list(np.ones(len(map_lines_dashed)))
    map_dashed.set_array(np.asarray(map_dashed_colors))
    
    
    #%% Load predicted data 
    model_to_path_name = 'trajectron_salzmann'
    model_to_path_module = importlib.import_module(model_to_path_name)
    model_class_to_path = getattr(model_to_path_module, model_to_path_name)
    
    
    data_set = data_set_class(model_class_to_path = model_class_to_path, 
                              num_samples_path_pred = 100)
    
    data_set.get_data(t0_type = data_params_list[0][3:], 
                      dt = float(data_params_list[1][3:]), 
                      num_timesteps_in = int(data_params_list[2][3:]), 
                      exclude_post_crit = True)

    model_avail_name = [f[:-3] for f in os.listdir(path + '/Models/') 
                   if f[-3:] == '.py' and f not in ['model_template.py']]
        
    model_avail = [getattr(importlib.import_module(model), model).get_name(None) 
                        for model in model_avail_name]

    model_name = model_avail_name[model_avail.index(model)]
    model_class = getattr(importlib.import_module(model_name), model_name)
    model_type = model_class.get_output_type_class()
    
    model_file = (path_model + '/' + os.path.basename(file_path))
    model_pred_file = (path_model + '/' + os.path.basename(file_path[:-4]) + '_prediction.npy')
    
    output = list(np.load(model_pred_file, allow_pickle = True)[:-1])
    
    if model_type == 'path':
        [Output_path_pred] = output
        [Output_A_pred, Output_T_A_pred] = data_set.path_to_binary_and_time(Output_path_pred, 
                                                                            Output_T, domain, model_file)
        
    elif model_type == 'binary':
        [Output_A_pred] = output
        
        Output_T_A_pred = data_set.binary_to_time(Output_A_pred, domain, model_file, split_file)
        Output_path_pred = data_set.binary_and_time_to_path(Output_A_pred, Output_T_A_pred, domain,
                                                            model_file, split_file)
    else:
        [Output_A_pred, Output_T_A_pred] = output
        Output_path_pred = data_set.binary_and_time_to_path(Output_A_pred, Output_T_A_pred, domain,
                                                            model_file, split_file)
        
        
    
    output_a_pred = Output_A_pred[ind]
    output_t_a_pred = Output_T_A_pred[ind][4]
    output_path_pred = Output_path_pred.iloc[ind]
    
    
    #%% Get maximum values
    max_v = np.stack([np.max(np.stack(np.array(output_path_pred)), axis = (1,2)), 
                      np.max(np.stack(np.array(output_path)), axis = (1)),
                      np.max(np.stack(np.array(input_path)).reshape(-1,2,int(data_params_list[2][3:])), axis = (0,2))])
    
    min_v = np.stack([np.min(np.stack(np.array(output_path_pred)), axis = (1,2)), 
                      np.min(np.stack(np.array(output_path)), axis = (1)),
                      np.min(np.stack(np.array(input_path)).reshape(-1,2,int(data_params_list[2][3:])), axis = (0,2))])
    
    max_v = np.ceil((np.max(max_v, 0) + 10) / 10) * 10
    min_v = np.floor((np.min(min_v, 0) - 10) / 10) * 10
    
    
    #%% plot figure
    fig, ax = plt.subplots(figsize = (10,8))
    # plot map
    ax.add_collection(map_solid)
    ax.add_collection(map_dashed)
    
    # plot inputs
    ax.plot(input_path.iloc[0], input_path.iloc[1], color = '#ff0000', 
            marker = 'x', ms = 5, label = r'$V_E$', linewidth = 2)
    ax.plot(input_path.iloc[2], input_path.iloc[3], color = '#0000ff', 
            marker = 'x', ms = 5, label = r'$V_T$', linewidth = 2)
    for i in range(int((len(input_path) - 4)/2)):
        ax.plot(input_path.iloc[2 * i + 4], input_path.iloc[2 * i + 5], 
                marker = 'x', ms = 5, label = r'$V_' + str(i + 1) + r'$', linewidth = 2)
    
    # plot predicted future
    path_pred = np.stack(np.array(output_path_pred)).transpose(1,0,2)
    for path in path_pred:
        ax.plot(np.concatenate((input_path.iloc[2][[-1]], path[0])), np.concatenate((input_path.iloc[3][[-1]], path[1])), color = '#4488ff', marker = 'o', 
                ms = 3, linestyle = 'dashed', linewidth = 0.5)
    
    # plot ground truth
    ax.plot(np.concatenate((input_path.iloc[2][[-1]], output_path.iloc[0])), np.concatenate((input_path.iloc[3][[-1]], output_path.iloc[1])), color = '#0000ff', 
            marker = 'o', ms = 5, linewidth = 2)
    
    ax.set_aspect('equal', adjustable='box') 
    ax.set_xlim([min_v[0],max_v[0]])
    ax.set_ylim([min_v[1],max_v[1]])
    title = (r'\textbf{Predicted paths}' + 
                 r'\\Data set: ' + data_set_name + r' with $n_I = ' + data_params_list[2][3:] + 
                 r'$, $\delta t = ' + data_params_list[1][3:] + 
                 r'$, and $t_0$ using \textit{' + data_params_list[0][3:] + 
                 r'} \\Split: ' + split.split('_')[0] + r' split' + 
                 r'\\Model: ' + model.replace('_', ' ') + 
                 r'\\True output: $a = ' + str(int(output_A)) + r'$, $' + strv + r'=' + str(output_T_E)[:5] + '$'  + 
                 r'\\Predicted output: $a_{pred} =' + str(output_a_pred)[:4] + r'$, $t_{A,pred} =' + 
                 str(output_t_a_pred)[:5] + '$')
    ax.set_title(title)
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()