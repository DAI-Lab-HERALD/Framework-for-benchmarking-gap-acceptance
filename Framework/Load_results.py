import numpy as np
import os
import importlib
import sys

# NOTE: Here still local path is assumed, so it might be useful
# to adapt that later
path = '/'.join(os.path.dirname(__file__).split('\\'))


data_set_path = os.path.join(path, 'Data_sets', '')
if not data_set_path in sys.path:
    sys.path.insert(0, data_set_path)

metric_path = os.path.join(path, 'Evaluation_metrics', '')
if not metric_path in sys.path:
    sys.path.insert(0, metric_path)
       
split_path = os.path.join(path, 'Splitting_methods', '')
if not split_path in sys.path:
    sys.path.insert(0, split_path)

model_path = os.path.join(path, 'Models', '')
if not model_path in sys.path:
    sys.path.insert(0, model_path)
    
#%% Set loading and saving parameters
# For loading plots while loading results
create_plot_if_possible     = False

# For the creation of figures
save_to_figure              = True

# For the creation of tables
save_to_table               = True


#%% Select modules
Data_sets = ['HighD_lane_change', 'HighD_lane_change_intent', 'RounD_round_about', 'CoR_left_turns']
Metrics = ['Accuracy', 'Accuracy_col', 'ROC_curve', 'ROC_curve_col', 'ADE', 'FDE', 'Useless_brake']
Data_params = [{'dt': 0.2, 'num_timesteps_in': 2}, {'dt': 0.2, 'num_timesteps_in': 10}] 
Splitters = ['Random_split', 'Critical_split']
Models = ['trajectron_salzmann', 'agent_yuan', 'logit_theofilatos', 'rt_mafi', 'db_xie', 'meta_khelfa']

#%% preprocess modules
# Splitting
Split_type = []
for i, split in enumerate(Splitters):
    end_split = split.split('_')[-1]
    if end_split != 'split':
        n = - 1 - len(end_split)
        Split_type.append(split[:n])
    else:
        Split_type.append(split)

split_types, split_index = np.unique(Split_type, return_inverse = True)

assert 'Critical_split' in split_types
assert 'Random_split' in split_types

Random_index = np.where(split_index == list(split_types).index('Random_split'))[0]
Critic_index = np.where(split_index == list(split_types).index('Critical_split'))[0]

# -----------------------------------------------------------------------------

# Metrics
Metrics_minimize = [] 
Metrics_T0_start = []

T0_start = {'start': 0,
            'col':   1,
            'crit':  2}

for metric_name in Metrics:
    metric_module = importlib.import_module(metric_name)
    metric_class = getattr(metric_module, metric_name)
    Metrics_minimize.append(metric_class.get_opt_goal() == 'minimize')
    Metrics_T0_start.append(T0_start[metric_class.get_t0_type()])
    
Metrics_minimize = np.array(Metrics_minimize)
Metrics_T0_start = np.array(Metrics_T0_start)


#%% Helper functions   
def write_data_point_into_plot(x, dx, random_values, critical_values, color):
    data_string = ' \n'
    
    x0 = x - dx
    x1 = x + dx
    colort = color + ', thin'
    
    if random_values is not None:
        data_string += (r'        \draw[' + colort + r'] ' + 
                        '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x0, np.mean(random_values) + 0.5 * dx, 
                                                                              x0, np.mean(random_values) - 0.5 * dx))
        data_string += (r'        \draw[' + colort + r'] ' + 
                        '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x1, np.mean(random_values) + 0.5 * dx, 
                                                                              x1, np.mean(random_values) - 0.5 * dx))
        data_string += (r'        \draw[' + colort + r'] ' + 
                        '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x0, np.mean(random_values), 
                                                                              x1, np.mean(random_values)))
    
    # critical value cross
    if critical_values is not None:
        data_string += (r'        \draw[' + colort + r'] ' + 
                        '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x0, np.mean(critical_values) + dx, 
                                                                              x1, np.mean(critical_values) - dx))
        data_string += (r'        \draw[' + colort + r'] ' + 
                        '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x0, np.mean(critical_values) - dx, 
                                                                              x1, np.mean(critical_values) + dx))
    
    return data_string


#%% Load desired results
# Result vecotr (can include objects)
Results = np.ones((len(Data_sets),
                   len(Metrics),
                   len(Data_params),
                   len(Splitters),
                   len(Models)),
                  float) * np.nan


# Deciding wether to allways use strict metrics
num_samples_path_pred = 100

for j, metric_name in enumerate(Metrics):
    metric_module = importlib.import_module(metric_name)
    metric_class = getattr(metric_module, metric_name)
    t0_type      = metric_class.get_t0_type()
    result_index = metric_class.main_result_idx()
    for i, data_set_name in enumerate(Data_sets):
        for k, data_param in enumerate(Data_params):  
            for m, model_name in enumerate(Models):
                model_module = importlib.import_module(model_name)
                model_class = getattr(model_module, model_name)
                
                for l, splitter_name in enumerate(Splitters):
                    splitter_module = importlib.import_module(splitter_name)
                    splitter_class = getattr(splitter_module, splitter_name)  
                    
                    data_param['num_timesteps_in']
                    num_timesteps_in = data_param['num_timesteps_in']
                    
                    results_file_name = (path + '/results/metrics/' + 
                                         data_set_name + '-io_paths(t0=' + t0_type + '_dt=' + str(data_param['dt']) + 
                                         '_nI={}'.format(num_timesteps_in) + '_EC)-' +
                                         splitter_class.get_name(0) + '-' + 
                                         model_class.get_name(0) + '-' + 
                                         metric_class.get_name(0) + '.npy')
                    try:
                        metric_result = np.load(results_file_name, allow_pickle = True)[:-1]
                        Results[i,j,k,l,m] = metric_result[result_index]
                    except:
                        pass


#%% save to figure
if save_to_figure:
    Data_set_names = {'HighD_lane_change':               r'\emph{highD}',
                      'HighD_lane_change_intent':        r'\emph{highD} (rest)',
                      'CoR_left_turns':                  r'\emph{L-GAP}',
                      'RounD_round_about':               r'\emph{rounD}'}
    
    Model_shortcut = {'trajectron_salzmann':    r'$T++$',
                      'agent_yuan':             r'$AF$',
                      'logit_theofilatos':      r'$LR$',
                      'rt_mafi':                r'$RF$',
                      'db_xie':                 r'$DB$',
                      'meta_khelfa':            r'$MH$'}
    
    
    Metric_shortcut = {'Accuracy':      r'\emph{Acc.}',
                       'Accuracy_col':  r'\emph{Acc.}',
                       'ROC_curve':     r'\emph{AUC}',
                       'ROC_curve_col': r'\emph{AUC}',
                       'ADE':           r'\emph{ADE}',
                       'FDE':           r'\emph{FDE}',
                       'Useless_brake': r'\emph{TNR-PR}'}
    
    
    T0_names = {0: r'Initial gaps',
                1: r'Fixed-sized gaps',
                2: r'Critical gaps'}
    
    
    Colors = ['lowcolor', 'midcolor', 'highcolor']

    # Get maximum figure width in cm
    num_para_values = len(Models) * len(Data_sets) * len(Data_params)
    if 60 < num_para_values:
        allowed_width = 22 # use textheight
    elif 30 < num_para_values <= 60:
        allowed_width = 18.13275 # use textwidth
    else:
        allowed_width = 8.85553  # use linewidth
    
    # Define empty spaces
    outer_space = 1.5
        
    inter_plot_space = 0.3 
    
    plot_height = 2.0
    plot_width  = (allowed_width - outer_space - inter_plot_space * (len(Data_sets) - 1)) / len(Data_sets)
    
    unique_t0_starts, num_t0_starts = np.unique(Metrics_T0_start, return_counts = True)
    
    overall_height = (outer_space + len(Metrics) * plot_height + 
                      inter_plot_space * (len(unique_t0_starts) - 1) + 
                      0.5 * inter_plot_space * (len(Metrics) - len(unique_t0_starts)))
    
    
    Figure_string  = r'\documentclass[journal]{standalone}' + ' \n' + ' \n'
    
    Figure_string += r'\input{header}' + ' \n'
    
    Figure_string += r'\begin{document}' + ' \n'
    
    if 60 < num_para_values:
        Figure_string += r'\begin{tikzpicture}[rotate = -90,transform shape]' + ' \n'
    else:
        Figure_string += r'\begin{tikzpicture}' + ' \n'
        
    Figure_string += r'    \definecolor{lowcolor}{RGB}{94,60,153}' + ' \n'
    Figure_string += r'    \definecolor{midcolor}{RGB}{174,122,126}' + ' \n'
    Figure_string += r'    \definecolor{highcolor}{RGB}{253,184,99}' + ' \n'
    
    # Start drawing the outer grid
    # Draw the left line 
    if len(Data_sets) > 1:
        Figure_string += ' \n' + r'    % Draw the outer x-axis' + ' \n'
        if len(unique_t0_starts) > 1:
            Figure_string += (r'    \draw[black] (0.5, 0.25) -- (0.5, 0.5) -- ' + 
                              '({:0.3f}, 0.5) -- ({:0.3f}, 0.25); \n'.format(allowed_width, allowed_width))
        else:
            Figure_string += (r'    \draw[black] (1.5, 0.25) -- (1.5, 0.5) -- ' + 
                              '({:0.3f}, 0.5) -- ({:0.3f}, 0.25); \n'.format(allowed_width, allowed_width))
            Figure_string += r'    \draw[black] (1.5, 0.75) -- (1.5, 0.5);' + ' \n'
            
        
        Figure_string += (r'    \draw[black] ' + 
                          '({:0.3f}, 0.5) -- ({:0.3f}, 0.75); \n'.format(allowed_width, allowed_width))
        
        for i in range(len(Data_sets)):
            label_value = outer_space + (i + 0.5) * plot_width + i * inter_plot_space
            
            if 60 < num_para_values:
                Figure_string += (r'    \node[black, rotate = 180, above, align = center, font = \footnotesize] at ' + 
                                  '({:0.3f}, 0.5) '.format(label_value) + 
                                  r'{' + Data_set_names[Data_sets[i]] + r'};' + ' \n') 
            else:
                Figure_string += (r'    \node[black, below, align = center, font = \footnotesize] at ' + 
                                  '({:0.3f}, 0.5) '.format(label_value) + 
                                  r'{' + Data_set_names[Data_sets[i]] + r'};' + ' \n') 
            
            if i < len(Data_sets) - 1:
                x_value = outer_space + (i + 1) * (plot_width + inter_plot_space) - 0.5 * inter_plot_space
                Figure_string += (r'    \draw[black] ' + 
                                  '({:0.3f}, 0.5) -- ({:0.3f}, 0.25); \n'.format(x_value, x_value))
                Figure_string += (r'    \draw[black, dashed] ' + 
                                  '({:0.3f}, 0.5) -- ({:0.3f}, {:0.3f}); \n'.format(x_value, x_value, overall_height))
    
    # Draw the lower line 
    if len(unique_t0_starts) > 1:
        Figure_string += ' \n' + r'    % Draw the outer y-axis' + ' \n'
        
        if len(Data_sets) > 1:
            Figure_string += (r'    \draw[black] (0.25, 0.5) -- (0.5, 0.5) -- ' + 
                              '(0.5, {:0.3f}) -- (0.25, {:0.3f}); \n'.format(overall_height, overall_height))
        else:
            Figure_string += (r'    \draw[black] (0.25, 1.5) -- (0.5, 1.5) -- ' + 
                              '(0.5, {:0.3f}) -- (0.25, {:0.3f}); \n'.format(overall_height, overall_height))
            Figure_string += r'    \draw[black] (0.75, 1.5) -- (0.5, 1.5);' + ' \n'
            
        
        Figure_string += (r'    \draw[black] ' + 
                          '(0.5, {:0.3f}) -- (0.75, {:0.3f}); \n'.format(overall_height, overall_height))
        
        lower_bound = overall_height + 0.5 * inter_plot_space
        for i in range(len(unique_t0_starts)):
            num_plots = num_t0_starts[i]
            upper_bound = lower_bound
            lower_bound = upper_bound - (1 + 0.5 * (num_plots - 1)) * inter_plot_space - plot_height * num_plots 
            label_value = 0.5 * (lower_bound + upper_bound)
            Figure_string += (r'    \node[black, rotate = 90, font = \footnotesize] at ' + 
                              '(0.25, {:0.3f}) '.format(label_value) + 
                              r'{' + T0_names[i] + r'};' + ' \n')           
            
            if i < len(unique_t0_starts) - 1:
                Figure_string += (r'    \draw[black] ' + 
                                  '(0.5, {:0.3f}) -- (0.25, {:0.3f}); \n'.format(lower_bound, lower_bound))
                Figure_string += (r'    \draw[black, dashed] ' + 
                                  '(0.5, {:0.3f}) -- ({:0.3f}, {:0.3f}); \n'.format(lower_bound, allowed_width, lower_bound))
    
    # start drawing the individual plots
    for j, metric_name in enumerate(Metrics):
        Figure_string += (' \n' + r'    % Draw the metric ' + Metric_shortcut[metric_name] + 
                          ' for ' + T0_names[Metrics_T0_start[j]] + ' \n')
        min_value, max_value = np.floor(np.nanmin(Results[:,j]) + 1e-5), np.ceil(np.nanmax(Results[:,j]) - 1e-5)
        y_value  = (overall_height - plot_height * (num_t0_starts[:Metrics_T0_start[j]].sum() + 1) - 
                    0.5 * inter_plot_space * (num_t0_starts[:Metrics_T0_start[j]].sum() + Metrics_T0_start[j]))
        y_value -= (plot_height + 0.5 * inter_plot_space) * (Metrics_T0_start[:j] == Metrics_T0_start[j]).sum()
        
        number = 1 + num_t0_starts[:Metrics_T0_start[j]].sum() + (Metrics_T0_start[:j] == Metrics_T0_start[j]).sum()
        
        y_max = len(Models) * plot_height / plot_width
        n_y_tick = 4 # Redo
        
        for k, dataset_name in enumerate(Data_sets):
            x_value = outer_space + k * (plot_width + inter_plot_space)
            Figure_string += (' \n' + r'    % Draw the metric ' + Metric_shortcut[metric_name] + 
                              ' for ' + T0_names[Metrics_T0_start[j]] + 
                              ' on dataset ' + Data_set_names[dataset_name] +  ' \n')
            
            
            Results_jk = Results[k,j,:,:,:]
            
            Figure_string += r'    \begin{axis}[' + ' \n'
            Figure_string += r'        at = {(' + '{:0.3f}cm, {:0.3f}cm'.format(x_value, y_value) + r')},' + ' \n'
            Figure_string += r'        width = ' + '{:0.3f}'.format(plot_width) + r'cm,' + ' \n'
            Figure_string += r'        height = ' + '{:0.3f}'.format(plot_height) + r'cm,' + ' \n'
            Figure_string += r'        scale only axis = true,' + ' \n'
            Figure_string += r'        axis lines = left,' + ' \n'
            Figure_string += r'        xmin = 0,' + ' \n'
            Figure_string += r'        xmax = ' + str(len(Models)) + r',' + ' \n'
            Figure_string += r'        xtick = {' + str([*range(1, len(Models) + 1)])[1:-1] + r'},' + ' \n'
            Figure_string += r'        xticklabels = {' 
            if number == len(Metrics): 
                for m, model_name in enumerate(Models):
                    Figure_string += Model_shortcut[model_name]
                    if m < len(Models) - 1:
                        Figure_string += ', '
            Figure_string += r'},' + ' \n' 
            Figure_string += (r'        x tick label style = {rotate=90, yshift = ' +
                              '{:0.3f}'.format(0.5 * plot_width / len(Models)) + 
                              r'cm, xshift = 0.1cm, align = right, font=\tiny},' + ' \n')
            Figure_string += r'        xmajorgrids = true,' + ' \n'
            Figure_string += r'        ymin = 0,' + ' \n'
            Figure_string += r'        ymax = ' + '{:0.3f}'.format(y_max) + r',' + ' \n'
            Figure_string += (r'        ytick = {' + 
                              '{:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}'.format(*np.linspace(0, y_max, n_y_tick + 1)) +
                              r'},' + ' \n')
            Figure_string += r'        yticklabels = {},' + ' \n'
            Figure_string += r'        ymajorgrids = true,' + ' \n'
            Figure_string += r'    ]' + ' \n'
            
            # Add values
            dx = min(y_max / 10, 1 / (2 * len(Data_params))) * 0.5
            for m, model_name in enumerate(Models):
                for n, data_params in enumerate(Data_params):
                    x_pos = m + (n + 1) / (len(Data_params) + 1)
                    results = Results_jk[n,:,m]
                    if not np.isnan(results).any():
                        results_trans = (results - min_value) * y_max / max_value
                        Figure_string += write_data_point_into_plot(x_pos, dx, results_trans[Random_index], 
                                                                    results_trans[Critic_index], Colors[n])
                    
            
            Figure_string += r'    \end{axis}' + ' \n'
            
            
            # Add y-axis labels if needed
            if k == 0:
                Figure_string += ' \n' + r'    % Draw the inner y-axis' + ' \n'
                # Add lower bound
                Figure_string += (r'    \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                  '(1.25, {:0.3f}) '.format(y_value) + 
                                  r'{$' + '{}'.format(min_value) + r'$};' + ' \n')  
                # Add upper bound
                Figure_string += (r'    \node[black, rotate = 90, inner sep = 0, left, font = \footnotesize] at ' + 
                                  '(1.25, {:0.3f}) '.format(y_value + plot_height) + 
                                  r'{$' + '{}'.format(max_value) + r'$};' + ' \n') 
                # Add metric name
                Figure_string += (r'    \node[black, rotate = 90, font = \footnotesize] at ' + 
                                  '(0.9, {:0.3f}) '.format(y_value + 0.5 * plot_height) + 
                                  r'{' + Metric_shortcut[metric_name] + r'};' + ' \n') 
                
        
    # Add the legend for n_I
    Figure_string += ' \n' + r'    % Draw the legend' + ' \n'
    
    if 60 < num_para_values:
        legend_width = overall_height - 0.5
    else:
        legend_width = allowed_width - 0.5
        
    legend_height = 0.8
    legend_entry_width = 4
    legend_entries = len(Data_params) * 2
    legend_entries_per_row = int(np.floor(legend_width / legend_entry_width))
    legend_rows = int(np.ceil(legend_entries / legend_entries_per_row))
    
    legend_entries_per_row = int(np.ceil(legend_entries / legend_rows))
    
    x_offset = 0.5 + 0.5 * (legend_width - legend_entries_per_row * legend_entry_width)
    
    Figure_string += r'    \filldraw[draw=black,fill = black!25!white, fill opacity = 0.2]'
    if 60 < num_para_values:
        Figure_string += '({:0.3f}, 0.5) rectangle ({:0.3f}, {:0.3f});'.format(allowed_width + inter_plot_space, 
                                                                               allowed_width + inter_plot_space + 
                                                                               (0.25 + 0.75 * legend_rows) * legend_height,
                                                                               overall_height) + ' \n'
    else:
        Figure_string += '(0.5, 0.0) rectangle ({:0.3f}, {:0.3f});'.format(allowed_width, 
                                                                           - (0.25 + 0.75 * legend_rows) * legend_height) + ' \n'
    
    
    for i, data_params in enumerate(Data_params):
        dx = 0.15 * legend_height
        if legend_rows == 1:
            x0_r = x_offset + 2 * i * legend_entry_width
            x0_c = x_offset + (2 * i + 1) * legend_entry_width
            y0_r = - 0.5 * legend_height
            y0_c = - 0.5 * legend_height
        elif legend_rows == 2:
            x0_r = x_offset + i * legend_entry_width
            x0_c = x_offset + i * legend_entry_width
            y0_r = - 0.5 * legend_height
            y0_c = - 1.25 * legend_height
        else:
            raise AttributeError("Implement a solution, lazy piece of shit")
        
        
        if 60 < num_para_values:
            # Random split
            Figure_string += write_data_point_into_plot(allowed_width + inter_plot_space - y0_r, dx, 
                                                        [x0_r + 0.5 * legend_height], None, Colors[i])
            Figure_string += (r'        \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                              '({:0.3f}, {:0.3f}) '.format(allowed_width + inter_plot_space - y0_r, x0_r + legend_height) + 
                              r'{$n_I =' + str(data_params['num_timesteps_in'][0]) + r'$, Random split};' + ' \n')
            
            # Critical split
            Figure_string += write_data_point_into_plot(allowed_width + inter_plot_space - y0_c, dx,
                                                        None, [x0_c + 0.5 * legend_height], Colors[i])
            Figure_string += (r'        \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                              '({:0.3f}, {:0.3f}) '.format(allowed_width + inter_plot_space - y0_c, x0_c + legend_height) + 
                              r'{$n_I =' + str(data_params['num_timesteps_in'][0]) + r'$, Critical split};' + ' \n')
        else:
            # Random split
            Figure_string += write_data_point_into_plot(x0_r + 0.5 * legend_height, dx, [y0_r], None, Colors[i])
            Figure_string += (r'        \node[black, inner sep = 0, right, font = \footnotesize] at ' + 
                              '({:0.3f}, {:0.3f}) '.format(x0_r + legend_height, y0_r) + 
                              r'{$n_I =' + str(data_params['num_timesteps_in'][0]) + r'$, Random split};' + ' \n')
            
            # Critical split
            Figure_string += write_data_point_into_plot(x0_c + 0.5 * legend_height, dx, None, [y0_c], Colors[i])
            Figure_string += (r'        \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                              '({:0.3f}, {:0.3f}) '.format(x0_c + legend_height, y0_c) + 
                              r'{$n_I =' + str(data_params['num_timesteps_in'][0]) + r'$, Critical split};' + ' \n')
    
    Figure_string += r'\end{tikzpicture}' + ' \n'
    Figure_string += r'\end{document}'
    
    figure_file_name = (path + '/latex_files/Figure' + '.tex')
    # Print latex file
    Figure_lines = Figure_string.split('\n')
    f = open(figure_file_name, 'w+')
    for line in Figure_lines:
        f.write(line + '\n')
    f.close() 
                      
#%% save to table
if save_to_table:
    Data_set_names = {'HighD_lane_change':               r'\emph{highD}',
                      'HighD_lane_change_intent':        r'\emph{highD} (rest)',
                      'CoR_left_turns':                  r'\emph{L-GAP}',
                      'RounD_round_about':               r'\emph{rounD}'}
    
    Model_shortcut = {'trajectron_salzmann':    r'$T++$',
                      'agent_yuan':             r'$AF$',
                      'logit_theofilatos':      r'$LR$',
                      'rt_mafi':                r'$RF$',
                      'db_xie':                 r'$DB$',
                      'meta_khelfa':            r'$MH$'}
        
        
    for j, metric_name in enumerate(Metrics):
        metric_module = importlib.import_module(metric_name)
        metric_class = getattr(metric_module, metric_name)
        t0_type_orig = metric_class.get_t0_type()   
        
        nP = len(Data_params)
        
        # start table header
        
        if len(Data_params) * len(Models) <= 12:
            if 9 < len(Data_params) * len(Models):
                Output_string  = r'\begin{tabularx}{\textheight}'
            elif 4 < len(Data_params) * len(Models) <= 9:
                Output_string  = r'\begin{tabularx}{\textwidth}'
            else:
                Output_string  = r'\begin{tabularx}{\linewidth}'
                
            Output_string += r'{X' + len(Models) * (r' | ' + r'C' * nP) + r'} '
            Output_string += '\n'
            Output_string += r'\toprule '
            Output_string += '\n'
            Output_string += r'\multirow{{2}}{{*}}{{\textbf{{Dataset}}}} '.format()
            Output_string += r'& \multicolumn{{{}}}{{c}}{{\textbf{{Models}}}} \\'.format(len(Models) * nP) 
            Output_string += '\n'
            for model_name in Models:
                Output_string += r'& \multicolumn{{{}}}'.format(nP) + r'{c}{' + Model_shortcut[model_name] + r'}'
            Output_string += r' \\'
            Output_string += '\n'
            Output_string += r'\midrule '
            Output_string += '\n'
            for i, data_set_name in enumerate(Data_sets): 
                
                t_single = r'& {{\scriptsize ${:0.3f}$}} ' * len(Models) * nP + r'\\'
                t_multi  = r'& {{\scriptsize ${:0.3f}^{{\pm {:0.3f}}}$}} ' * len(Models) * nP + r'\\'
                
                Results_random = Results[i, j][:, Random_index]
                Results_critical = Results[i, j][:, Critic_index]    
                
                Results_random_mean = Results_random.mean(axis = 1)
                Results_random_std  = Results_random.std(axis = 1)
                
                Results_critical_mean = Results_critical.mean(axis = 1)
                Results_critical_std  = Results_critical.std(axis = 1)
                
                multi_random = len(Random_index) > 1
                multi_critic = len(Critic_index) > 1
                
                Random_mean_in = Results_random_mean.T.reshape(-1)
                Random_std_in  = Results_random_std.T.reshape(-1)
                Critic_mean_in = Results_critical_mean.T.reshape(-1)
                Critic_std_in  = Results_critical_std.T.reshape(-1)
                
                if multi_random or multi_critic:
                    Str_random   = t_multi.format(*np.array([Random_mean_in, Random_std_in]).T.reshape(-1))
                    Str_critical = t_multi.format(*np.array([Critic_mean_in, Critic_std_in]).T.reshape(-1))
                    
                    Str_random   = Str_random.replace("\\pm 0.000", r"\hphantom{\pm 0.000}")
                    Str_critical = Str_critical.replace("\\pm 0.000", r"\hphantom{\pm 0.000}")
                else:
                    Str_random = t_single.format(*Random_mean_in)
                    Str_critical = t_single.format(*Critic_mean_in)
                # Get to minimize
                
                
                if metric_class.get_opt_goal() == 'minimize':
                    best_value_random = np.min(Results_random_mean, axis = -1)
                    best_value_critical = np.min(Results_critical_mean, axis = -1)
                    adapt_string = True
                elif metric_class.get_opt_goal() == 'maximize':
                    best_value_random = np.max(Results_random_mean, axis = -1)
                    best_value_critical = np.max(Results_critical_mean, axis = -1)
                    adapt_string = True
                else:
                    adapt_string = False
                
                
                if adapt_string:
                    Str_random_parts = Str_random.split('$} ')
                    for idx, bv in enumerate(best_value_random):
                        useful = np.where(bv == Results_random_mean[idx])[0] * nP + idx
                        for useful_idx in useful:
                            Str_random_parts[useful_idx] = (r'& {\scriptsize $\underline{' + Str_random_parts[useful_idx][16:21] + 
                                                            r'}' + Str_random_parts[useful_idx][21:])
                    Str_random = '$} '.join(Str_random_parts)
                    
                    Str_critical_parts = Str_critical.split('$} ')
                    for idx, bv in enumerate(best_value_critical):
                        useful = np.where(bv == Results_critical_mean[idx])[0] * nP + idx
                        for useful_idx in useful:
                            Str_critical_parts[useful_idx] = (r'& {\scriptsize $\underline{' + Str_critical_parts[useful_idx][16:21] + 
                                                              r'}' + Str_critical_parts[useful_idx][21:]) 
                    Str_critical = '$} '.join(Str_critical_parts)
                
                if not (np.isnan(Results_critical).all() and np.isnan(Results_random).all()):
                    Output_string += (Data_set_names[data_set_name] + ' ' + Str_random + ' \n')
                    Output_string += (Str_critical + ' \n\midrule \n')
            
            # replace last midrule with bottom rule  
            Output_string  = Output_string[:-10] + r'\bottomrule'
            Output_string += '\n'
            Output_string += r'\end{tabularx}'
            
            # split string into lines
            Output_lines = Output_string.split('\n')
                
            table_file_name = (path + 
                               '/latex_files/Table_' +
                               metric_class.get_name(0) + '_at_' +
                               t0_type_orig + '.tex')
            t = open(table_file_name, 'w+')
            for line in Output_lines:
                t.write(line + '\n')
            t.close()
        else:
            num_tables = int(np.ceil(len(Models) / np.floor(12 / len(Data_params))))
            models_per_table = int(np.ceil(len(Models) / num_tables))
            # Allow for split table
            Output_strings = [r'\begin{tabularx}{\textheight}'] * num_tables
            for n in range(num_tables):
                models_n = np.arange(n * models_per_table, min(len(Models), (n + 1) * models_per_table))
                Output_strings[n] += r'{X' + models_per_table * (r' | ' + r'Z' * nP) + r'} '
                Output_strings[n] += '\n'
                if n > 0:
                    Output_strings[n] += r'\multicolumn{' + str(models_per_table * nP + 1) + r'}{c}{} \\' + ' \n'
                Output_strings[n] += r'\toprule '
                Output_strings[n] += '\n'
                if n == 0:
                    Output_strings[n] += r'\multirow{{2}}{{*}}{{\textbf{{Dataset}}}} '.format()
                    Output_strings[n] += r'& \multicolumn{{{}}}{{c}}{{\textbf{{Models}}}} \\'.format(models_per_table * nP) 
                    Output_strings[n] += '\n'
                for model_idx in models_n:
                    model_name = Models[model_idx]
                    Output_strings[n] += r'& \multicolumn{{{}}}'.format(nP) + r'{c}{' + Model_shortcut[model_name] + r'}'
                Output_strings[n] += r' \\'
                Output_strings[n] += '\n'
                Output_strings[n] += r'\midrule '
                Output_strings[n] += '\n'
                for i, data_set_name in enumerate(Data_sets): 
                    
                    t_single = (r'& {{\scriptsize ${:0.3f}$}} ' * len(models_n) * nP + 
                                '& ' * (models_per_table - len(models_n)) * nP + r'\\')
                    t_multi  = (r'& {{\scriptsize ${:0.3f}^{{\pm {:0.3f}}}$}} ' * len(models_n) * nP + 
                                '& ' * (models_per_table - len(models_n)) * nP + r'\\')
                    
                    Results_random = Results[i, j][:, Random_index]
                    Results_critical = Results[i, j][:, Critic_index]    
                    
                    Results_random_mean = Results_random.mean(axis = 1)
                    Results_random_std  = Results_random.std(axis = 1)
                    
                    Results_critical_mean = Results_critical.mean(axis = 1)
                    Results_critical_std  = Results_critical.std(axis = 1)
                    
                    multi_random = len(Random_index) > 1
                    multi_critic = len(Critic_index) > 1
                    
                    Random_mean_in = Results_random_mean[:,models_n].T.reshape(-1)
                    Random_std_in  = Results_random_std[:,models_n].T.reshape(-1)
                    Critic_mean_in = Results_critical_mean[:,models_n].T.reshape(-1)
                    Critic_std_in  = Results_critical_std[:,models_n].T.reshape(-1)
                    
                    if multi_random or multi_critic:
                        Str_random   = t_multi.format(*np.array([Random_mean_in, Random_std_in]).T.reshape(-1))
                        Str_critical = t_multi.format(*np.array([Critic_mean_in, Critic_std_in]).T.reshape(-1))
                        
                        Str_random   = Str_random.replace("\\pm 0.000", r"\hphantom{\pm 0.000}")
                        Str_critical = Str_critical.replace("\\pm 0.000", r"\hphantom{\pm 0.000}")
                    else:
                        Str_random = t_single.format(*Random_mean_in)
                        Str_critical = t_single.format(*Critic_mean_in)
                    # Get to minimize
                    
                    
                    if metric_class.get_opt_goal() == 'minimize':
                        best_value_random = np.min(Results_random_mean, axis = -1)
                        best_value_critical = np.min(Results_critical_mean, axis = -1)
                        adapt_string = True
                    elif metric_class.get_opt_goal() == 'maximize':
                        best_value_random = np.max(Results_random_mean, axis = -1)
                        best_value_critical = np.max(Results_critical_mean, axis = -1)
                        adapt_string = True
                    else:
                        adapt_string = False
                    
                    
                    if adapt_string:
                        Str_random_parts = Str_random.split('$} ')
                        for idx, bv in enumerate(best_value_random):
                            useful = np.where(bv == Results_random_mean[idx, models_n])[0] * nP + idx
                            for useful_idx in useful:
                                Str_random_parts[useful_idx] = (r'& {\scriptsize $\underline{' + Str_random_parts[useful_idx][16:21] + 
                                                                r'}' + Str_random_parts[useful_idx][21:])
                        Str_random = '$} '.join(Str_random_parts)
                        
                        Str_critical_parts = Str_critical.split('$} ')
                        for idx, bv in enumerate(best_value_critical):
                            useful = np.where(bv == Results_critical_mean[idx, models_n])[0] * nP + idx
                            for useful_idx in useful:
                                Str_critical_parts[useful_idx] = (r'& {\scriptsize $\underline{' + Str_critical_parts[useful_idx][16:21] + 
                                                                  r'}' + Str_critical_parts[useful_idx][21:]) 
                        Str_critical = '$} '.join(Str_critical_parts)
                    
                    if not (np.isnan(Results_critical).all() and np.isnan(Results_random).all()):
                        Output_strings[n] += (Data_set_names[data_set_name] + ' ' + Str_random + ' \n')
                        Output_strings[n] += (Str_critical + ' \n\midrule \n')
                
                # replace last midrule with bottom rule  
                Output_strings[n]  = Output_strings[n][:-10] + r'\bottomrule'
                Output_strings[n] += '\n'
                Output_strings[n] += r'\end{tabularx}'
                
                # split string into lines
                Output_lines = Output_strings[n].split('\n')
                
                addon = '_(' + str(n) + ').tex'
                    
                table_file_name = (path + 
                                   '/latex_files/Table_' +
                                   metric_class.get_name(0) + '_at_' +
                                   t0_type_orig + addon)
                t = open(table_file_name, 'w+')
                for line in Output_lines:
                    t.write(line + '\n')
                t.close()


