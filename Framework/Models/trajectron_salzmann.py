import sys
import os
from model_template import model_template
sys.path.insert(0, os.path.dirname(__file__) + '/Trajectron/trajectron/')
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import dill
import random
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from environment.node_type import NodeTypeEnum

class trajectron_salzmann(model_template):
    
    def setup_method(self, seed = 0):
        # check for gpus
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            raise TypeError("The GPU has gone and fucked itself")
        else:
            if torch.cuda.device_count() == 1:
                # If you have CUDA_VISIBLE_DEVICES set, which you should,
                # then this will prevent leftover flag arguments from
                # messing with the device allocation.
                device = 'cuda:0'
        
            device = torch.device(device)
        
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        batch_size = 50
        # Get params
        Y_help = self.Output_path_train.to_numpy() 
        self.num_timesteps_in = len(self.Input_path_train.to_numpy()[0,0])
        self.num_timesteps_out = np.zeros(len(Y_help), int)
        for i_sample in range(Y_help.shape[0]):
            usefulx = np.where(np.isnan(Y_help[i_sample,0]))[0]
            usefuly = np.where(np.isnan(Y_help[i_sample,1]))[0]
            if len(usefulx) == 0 and len(usefuly) == 0:
                len_use = len(Y_help[i_sample,0])
            else:
                try: 
                    lenx = usefulx[0] 
                except:
                    lenx = len(Y_help[i_sample,0])
                try: 
                    leny = usefuly[0] 
                except:
                    leny = len(Y_help[i_sample,0])
                
                len_use = min(lenx, leny)
            
            if len_use > 5:
                self.num_timesteps_out[i_sample] = len_use - np.mod(len_use,5)
            else: 
                self.num_timesteps_out[i_sample] = len_use
                    
        self.num_timesteps_out = np.minimum(self.num_timesteps_out, 100)
        
        
        
        hyperparams = {'batch_size': batch_size,
                       'grad_clip': 1.0,
                       'learning_rate_style': 'exp',
                       'learning_rate': 0.003,
                       'min_learning_rate': 1e-05,
                       'learning_decay_rate': 0.9999,
                       'prediction_horizon': self.num_timesteps_out.max(),
                       'minimum_history_length': 2,
                       'maximum_history_length': self.num_timesteps_in - 1,
                       'map_encoder': {'VEHICLE': {'heading_state_index': 6,
                                                   'patch_size': [50, 10, 50, 90],
                                                   'map_channels': 3,
                                                   'hidden_channels': [10, 20, 10, 1],
                                                   'output_size': 32,
                                                   'masks': [5, 5, 5, 3],
                                                   'strides': [2, 2, 1, 1],
                                                   'dropout': 0.5}},
                       'k': 1,
                       'k_eval': 25,
                       'kl_min': 0.07,
                       'kl_weight': 100.0,
                       'kl_weight_start': 0,
                       'kl_decay_rate': 0.99995,
                       'kl_crossover': 400,
                       'kl_sigmoid_divisor': 4,
                       'rnn_kwargs': {'dropout_keep_prob': 0.75},
                       'MLP_dropout_keep_prob': 0.9,
                       'enc_rnn_dim_edge': 32,
                       'enc_rnn_dim_edge_influence': 32,
                       'enc_rnn_dim_history': 32,
                       'enc_rnn_dim_future': 32,
                       'dec_rnn_dim': 128,
                       'q_z_xy_MLP_dims': None,
                       'p_z_x_MLP_dims': 32,
                       'GMM_components': 1,
                       'log_p_yt_xz_max': 6,
                       'N': 1,
                       'K': 25,
                       'tau_init': 2.0,
                       'tau_final': 0.05,
                       'tau_decay_rate': 0.997,
                       'use_z_logit_clipping': True,
                       'z_logit_clip_start': 0.05,
                       'z_logit_clip_final': 5.0,
                       'z_logit_clip_crossover': 300,
                       'z_logit_clip_divisor': 5,
                       'dynamic': {'PEDESTRIAN': {'name': 'SingleIntegrator',
                                                  'distribution': True,
                                                  'limits': {}},
                                   'VEHICLE': {'name': 'Unicycle',
                                               'distribution': True,
                                               'limits': {'max_a': 4,
                                                          'min_a': -5,
                                                          'max_heading_change': 0.7,
                                                          'min_heading_change': -0.7}}},
                       'state': {'PEDESTRIAN': {'position': ['x', 'y'],
                                                'velocity': ['x', 'y'],
                                                'acceleration': ['x', 'y']},
                                 'VEHICLE': {'position': ['x', 'y'],
                                             'velocity': ['x', 'y'],
                                             'acceleration': ['x', 'y'],
                                             'heading': ['°', 'd°']}},
                       'pred_state': {'VEHICLE': {'position': ['x', 'y']},
                                      'PEDESTRIAN': {'position': ['x', 'y']}},
                       'log_histograms': False,
                       'dynamic_edges': 'yes',
                       'edge_state_combine_method': 'sum',
                       'edge_influence_combine_method': 'attention',
                       'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
                       'edge_removal_filter': [1.0, 0.0],
                       'offline_scene_graph': 'yes',
                       'incl_robot_node': False,
                       'node_freq_mult_train': False,
                       'node_freq_mult_eval': False,
                       'scene_freq_mult_train': False,
                       'scene_freq_mult_eval': False,
                       'scene_freq_mult_viz': False,
                       'edge_encoding': True,
                       'use_map_encoding': False,
                       'augment': True,
                       'override_attention_radius': []}
    

        train_data_path = 'Models/Trajectron/experiments/processed/eth_train.pkl'
        with open(train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')
    

        # Offline Calculate Scene Graph
        model_registrar = ModelRegistrar(None, device)
        self.trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     None,
                                     device)
        train_env.NodeType = NodeTypeEnum(['PEDESTRIAN', 'VEHICLE'])
        self.dt = self.Input_T_train[0][-1] - self.Input_T_train[0][-2]
        train_env.scenes[0].dt = self.dt
        
        self.trajectron.set_environment(train_env)
        self.trajectron.set_annealing_params()
        
        
        

    def train_method(self, epochs = 200):   
        batch_size = self.trajectron.hyperparams['batch_size']     
        # prepare input data
        attention_radius = dict()
        attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
        attention_radius[('PEDESTRIAN', 'VEHICLE')] = 50.0
        attention_radius[('VEHICLE', 'PEDESTRIAN')] = 25.0
        attention_radius[('VEHICLE', 'VEHICLE')] = 150.0
        
        DIM = {'VEHICLE': 8, 'PEDESTRIAN': 6}

        X_help = self.Input_path_train.to_numpy()
        X_help = X_help.reshape(len(X_help), -1, 2)
        Y_help = self.Output_path_train.to_numpy() 
        
        X = np.zeros(list(X_help.shape) + [self.num_timesteps_in])
        Type = [name[0] for name in np.array(self.input_names_train).reshape(-1,2)[:,0]]
        
        for i_sample in range(X.shape[0]):
            for i_v in range(X.shape[1]):
                if i_sample == 0:
                    if Type[i_v] == 'P':
                        Type[i_v] = 'PEDESTRIAN'
                    else:
                        Type[i_v] = 'VEHICLE'
                for dim in range(2):
                    X[i_sample, i_v, dim, :] = X_help[i_sample, i_v, dim]
        
        # Assume that target vehicle will always be vehicle or pedestrian
        node_type = Type[1]            
        
        num, count = np.unique(self.num_timesteps_out, return_counts = True)
        
        num_batches = np.maximum(1, count // batch_size)
                       
        Y = np.ones(list(Y_help.shape) + [self.num_timesteps_out.max()]) * np.nan
        
        for i_sample in range(X.shape[0]):
            for dim in range(2):
                Y[i_sample, dim, :self.num_timesteps_out[i_sample]] = Y_help[i_sample, dim][:self.num_timesteps_out[i_sample]]
                if np.any(np.isnan(Y[i_sample, dim, :self.num_timesteps_out[i_sample]])):
                    raise TypeError('asdasdasd')
                
        
        # get velocitites
        V = (X[:,:,:,1:] - X[:,:,:,:-1]) / self.dt
        V = np.concatenate((V[:,:,:,[0]], V), axis = -1)
        
        # get accelaration
        A = (V[:,:,:,1:] - V[:,:,:,:-1]) / self.dt
        A = np.concatenate((A[:,:,:,[0]], A), axis = -1)
        
        H = np.arctan2(V[:,:,1,:], V[:,:,0,:])
        
        DH = np.unwrap(H, axis = -1) 
        DH = (DH[:,:,1:] - DH[:,:,:-1]) / self.dt
        DH = np.concatenate((DH[:,:,[0]], DH), axis = -1)
        
        #final state S
        S = np.concatenate((X, V, A, H[:,:,np.newaxis,:], DH[:,:,np.newaxis,:]), axis = 2).astype(np.float32).transpose(0,1,3,2)
        if np.any(np.isnan(S)):
            print('The inputs include a nan value')
            
        # Test min distance from tar to rest
        D = np.min(np.sqrt(np.sum((X[:,[1],:,:] - X) ** 2, axis = 2)), axis = - 1)
        D_max = np.array([[attention_radius[(Type[1], Type[i_v])] for i_v in range(X.shape[1])]])
        
        Neighbor_bool = D < D_max

        tar_help = np.arange(X.shape[1]) != 1
        
        Neighbor = {}
        Neighbor_edge_value = {}
        for node_goal in DIM.keys():
            Neighbor[(node_type,node_goal)] = []
            Neighbor_edge_value[(node_type,node_goal)] = []
            Dim = DIM[node_goal]
            
            for i_sample in range(S.shape[0]):
                Neighbor[(node_type,node_goal)].append([])
                
                i_v_goal = np.where((np.array(Type) == node_goal) & 
                                    Neighbor_bool[i_sample] & tar_help)[0]
                
                Neighbor_edge_value[(node_type,node_goal)].append(torch.from_numpy(np.ones(len(i_v_goal), np.float32)))
                for i_v in i_v_goal:
                    Neighbor[(node_type,node_goal)][i_sample].append(torch.from_numpy(S[i_sample, i_v, :, :Dim]))
                
        
        
        
        # Get gradient clipping values              
        clip_value_final = self.trajectron.hyperparams['grad_clip']
        # clip_value_inter = clip_value_final * 1000
        
        # prepare training
        optimizer = dict()
        lr_scheduler = dict()
        if node_type  in self.trajectron.hyperparams['pred_state']:
            optimizer[node_type] = optim.Adam([{'params': self.trajectron.model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                                {'params': self.trajectron.model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], 
                                              lr = self.trajectron.hyperparams['learning_rate'])
            # Set Learning Rate
            if self.trajectron.hyperparams['learning_rate_style'] == 'const':
                lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
            elif self.trajectron.hyperparams['learning_rate_style'] == 'exp':
                lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                            gamma=self.trajectron.hyperparams['learning_decay_rate'])
        else:
            raise TypeError('Missing model componenets')
            
        batch_index = np.arange(num_batches.sum())
        batch_index_num = np.argmax(batch_index[:, np.newaxis] < np.cumsum(num_batches)[np.newaxis,:], 1)
        curr_iter = 0
        first_h = torch.from_numpy(np.zeros(batch_size, np.int32))
        # go over epochs
        for epoch in range(1, epochs + 1):
            print('Train trajectron: Epoch ' + 
                  str(epoch).rjust(len(str(epochs))) + 
                  '/{}'.format(epochs))
            self.trajectron.model_registrar.to(self.trajectron.device)
            np.random.shuffle(batch_index_num)
            for i_batch, batch in enumerate(batch_index_num):
                num_steps = num[batch]
                if num_steps > 1:
                    Weights = list(self.trajectron.model_registrar.parameters())
                    batch_count = count[batch]
                    Index_batch = np.where(self.num_timesteps_out == num[batch])[0]
                    if batch_count >= batch_size:
                        use = np.random.choice(batch_count, batch_size, replace = False)
                    else:
                        if np.random.rand() < batch_count / batch_size:
                            use = np.random.choice(batch_count, batch_size, replace = True)
                        else:
                            continue
                    
                    print('Train trajectron: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{} - Batch '.format(epochs) + 
                          str(i_batch).rjust(len(str(len(batch_index_num)))) +
                          '/{}'.format(len(batch_index_num)))
                    
                    Index_use = Index_batch[use]
                    
                    self.trajectron.set_curr_iter(curr_iter)
                    self.trajectron.step_annealers(node_type)
                    optimizer[node_type].zero_grad()
                    
                    S_batch = torch.from_numpy(S[Index_use,1,:,:DIM[node_type]])
                    Y_batch = torch.from_numpy(Y[Index_use,:,:num_steps].astype(np.float32).transpose(0,2,1))
                    
                    # Get batch data
                    Neighbor_batch = {}
                    Neighbor_edge_value_batch = {}
                    for node_goal in DIM.keys():
                        Neighbor_batch[(node_type,node_goal)] = []
                        Neighbor_edge_value_batch[(node_type,node_goal)] = []
                        
                        for i_sample in Index_use:
                            Neighbor_batch[(node_type,node_goal)].append(Neighbor[(node_type,node_goal)][i_sample])
                            Neighbor_edge_value_batch[(node_type,node_goal)].append(Neighbor_edge_value[(node_type,node_goal)][i_sample]) 
                    
                    # Run forward pass
                    model = self.trajectron.node_models_dict[node_type]
                    train_loss = model.train_loss(inputs = S_batch.to(self.trajectron.device),
                                                  inputs_st = S_batch.to(self.trajectron.device),
                                                  first_history_indices = first_h.to(self.trajectron.device),
                                                  labels = Y_batch.to(self.trajectron.device),
                                                  labels_st = Y_batch.to(self.trajectron.device),
                                                  neighbors = Neighbor_batch,
                                                  neighbors_edge_value = Neighbor_edge_value_batch,       
                                                  robot = None,
                                                  map = None,
                                                  prediction_horizon = num_steps)
    
                    assert train_loss.isfinite().all(), "The overall loss of the model is nan"
    
                    train_loss.backward()
     
                    gradients_good = all([(weights.grad.isfinite().all() if weights.grad is not None else True) 
                                          for weights in Weights])
                    
                    if gradients_good:
                        if self.trajectron.hyperparams['grad_clip'] is not None:
                            nn.utils.clip_grad_value_(self.trajectron.model_registrar.parameters(), clip_value_final)
        
                        optimizer[node_type].step()
                        lr_scheduler[node_type].step()
                        curr_iter += 1
                    else:
                        print('To many output timesteps => exploding gradients => weights not updated')
                
                else:
                    print("Not enough output timesteps => no loss can be calculated")
                
                   
                    
                
        
        
        
        
        # save weigths 
        # after checking here, please return num_epochs to 100 and batch size to 
        Weights = list(self.trajectron.model_registrar.parameters())
        self.weights_saved = []
        for weigths in Weights:
            self.weights_saved.append(weigths.detach().cpu().numpy())
        
        
    def load_method(self, l2_regulization = 0):
        Weights = list(self.trajectron.model_registrar.parameters())
        with torch.no_grad():
            for i, weights in enumerate(self.weights_saved):
                Weights[i][:] = torch.from_numpy(weights)[:]
        
    def predict_method(self):
        # get desired output length
        YT_help = self.Output_T_test
        self.num_timesteps_out_test = np.zeros(len(YT_help), int)
        for i_sample in range(len(YT_help)):
            self.num_timesteps_out_test[i_sample] = len(YT_help[i_sample])
            
        # prepare input data
        attention_radius = dict()
        attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
        attention_radius[('PEDESTRIAN', 'VEHICLE')] = 50.0
        attention_radius[('VEHICLE', 'PEDESTRIAN')] = 25.0
        attention_radius[('VEHICLE', 'VEHICLE')] = 150.0
        
        DIM = {'VEHICLE': 8, 'PEDESTRIAN': 6}

        X_help = self.Input_path_test.to_numpy()
        X_help = X_help.reshape(len(X_help), -1, 2)
        
        X = np.zeros(list(X_help.shape) + [self.num_timesteps_in])
        Type = [name[0] for name in np.array(self.input_names_train).reshape(-1,2)[:,0]]

        for i_sample in range(X.shape[0]):
            for i_v in range(X.shape[1]):
                if i_sample == 0:
                    if Type[i_v] == 'P':
                        Type[i_v] = 'PEDESTRIAN'
                    else:
                        Type[i_v] = 'VEHICLE'
                for dim in range(2):
                    X[i_sample, i_v, dim, :] = X_help[i_sample, i_v, dim]
        
        # Assume that target vehicle will always be either vehicle or pedestrian
        node_type = Type[1]            

        # get velocitites
        V = (X[:,:,:,1:] - X[:,:,:,:-1]) / self.dt
        V = np.concatenate((V[:,:,:,[0]], V), axis = -1)
        
        # get accelaration
        A = (V[:,:,:,1:] - V[:,:,:,:-1]) / self.dt
        A = np.concatenate((A[:,:,:,[0]], A), axis = -1)
        
        H = np.arctan2(V[:,:,1,:], V[:,:,0,:])
        
        DH = np.unwrap(H, axis = -1) 
        DH = (DH[:,:,1:] - DH[:,:,:-1]) / self.dt
        DH = np.concatenate((DH[:,:,[0]], DH), axis = -1)
        
        #final state S
        S = np.concatenate((X, V, A, H[:,:,np.newaxis,:], DH[:,:,np.newaxis,:]), axis = 2).astype(np.float32).transpose(0,1,3,2)
        
        # Test min distance from tar to rest
        D = np.min(np.sqrt(np.sum((X[:,[1],:,:] - X) ** 2, axis = 2)), axis = - 1)
        D_max = np.array([[attention_radius[(Type[1], Type[i_v])] for i_v in range(X.shape[1])]])
        
        Neighbor_bool = D < D_max

        tar_help = np.arange(X.shape[1]) != 1
        
        Neighbor = {}
        Neighbor_edge_value = {}
        for node_goal in DIM.keys():
            Neighbor[(node_type,node_goal)] = []
            Neighbor_edge_value[(node_type,node_goal)] = []
            Dim = DIM[node_goal]
            
            for i_sample in range(S.shape[0]):
                Neighbor[(node_type,node_goal)].append([])
                
                i_v_goal = np.where((np.array(Type) == node_goal) & 
                                    Neighbor_bool[i_sample] & tar_help)[0]
                
                Neighbor_edge_value[(node_type,node_goal)].append(torch.from_numpy(np.ones(len(i_v_goal), np.float32)))
                for i_v in i_v_goal:
                    Neighbor[(node_type,node_goal)][i_sample].append(torch.from_numpy(S[i_sample, i_v, :, :Dim]))
                
        
        
        Output_Path = pd.DataFrame(np.empty((S.shape[0], 2), object), columns = self.Output_path_train.columns)
        
        nums = np.unique(self.num_timesteps_out_test)
        
        batch_size = self.trajectron.hyperparams['batch_size']
        samples_done = 0
        calculations_done = 0
        samples_all = len(S)
        calculations_all = np.sum(self.num_timesteps_out_test)
        for num in nums:
            Index_batch = np.where(self.num_timesteps_out_test == num)[0]
            batch_size_real = int(np.ceil(batch_size / (num / 5)))
            if batch_size_real > len(Index_batch):
                Index_uses = [Index_batch]
            else:
                Index_uses = [Index_batch[i * batch_size_real : (i + 1) * batch_size_real] 
                              for i in range(int(np.ceil(len(Index_batch)/ batch_size_real)))] 
            
            for Index_use in Index_uses:
                S_batch = torch.from_numpy(S[Index_use,1,:,:DIM[node_type]])
                # Get batch data
                Neighbor_batch = {}
                Neighbor_edge_value_batch = {}
                for node_goal in DIM.keys():
                    Neighbor_batch[(node_type,node_goal)] = []
                    Neighbor_edge_value_batch[(node_type,node_goal)] = []
                    
                    for i_sample in Index_use:
                        Neighbor_batch[(node_type,node_goal)].append(Neighbor[(node_type,node_goal)][i_sample])
                        Neighbor_edge_value_batch[(node_type,node_goal)].append(Neighbor_edge_value[(node_type,node_goal)][i_sample]) 
                
                first_h = torch.from_numpy(np.zeros(len(Index_use), np.int32))
                # Run prediction pass
                model = self.trajectron.node_models_dict[node_type]
                
                Pred = []
                needed_max = 10000
                helper = int(np.floor(needed_max/ num / len(Index_use)))
                splits = int(np.ceil(self.num_samples_path_pred / helper))
                for k in range(splits):
                    num_samples = min(helper, self.num_samples_path_pred - k * helper)
                    self.trajectron.model_registrar.to(self.trajectron.device)
                    with torch.no_grad(): # Do not build graph for backprop
                        predictions = model.predict(inputs = S_batch.to(self.trajectron.device),
                                                    inputs_st = S_batch.to(self.trajectron.device),
                                                    first_history_indices = first_h.to(self.trajectron.device),
                                                    neighbors = Neighbor_batch,
                                                    neighbors_edge_value = Neighbor_edge_value_batch,
                                                    robot = None,
                                                    map = None,
                                                    prediction_horizon = num,
                                                    num_samples = num_samples)
                    
                    Pred.append(predictions.detach().cpu().numpy())
                    torch.cuda.empty_cache()
                
                Pred = np.concatenate(Pred, axis = 0)   
                for i, i_sample in enumerate(Index_use):
                    for j, index in enumerate(Output_Path.columns):
                        res = np.zeros((Pred.shape[0], Pred.shape[2]))
                        for n in range(self.num_samples_path_pred):
                            res[n,:] = Pred[n, i, :, j]
                        
                        Output_Path.iloc[i_sample][index] = res.astype('float32')
                
                samples_done += len(Index_use)
                calculations_done += len(Index_use) * num
                
                samples_perc = 100 * samples_done / samples_all
                calculations_perc = 100 * calculations_done / calculations_all
                
                print('Predict trajectron: ' + 
                      format(samples_perc, '.2f').rjust(len('100.00')) + 
                      '% of samples, ' + 
                      format(calculations_perc, '.2f').rjust(len('100.00')) +
                      '% of calculations')
                
                

        return [Output_Path]

    def check_input_names_method(self, names, train = True):
        if all(names == self.input_names_train):
            return True
        else:
            return False
    
    def get_output_type_class():
        return 'path'
    
    def get_name(self):
        return 'trajetron'
        

        
        
        
        
        
    
        
        
        