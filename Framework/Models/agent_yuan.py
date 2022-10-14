import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from model_template import model_template

sys.path.insert(0, os.path.dirname(__file__) + '/agentformer/') 

from agentformer.data.dataloader import data_generator
from agentformer.model.model_lib import model_dict
from agentformer.utils.torch import *
from agentformer.utils.config import Config
from agentformer.utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring

class agent_yuan(model_template):
    
    def setup_method(self, seed = 0):
        # check for gpus
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        prepare_seed(0)
        torch.set_default_dtype(torch.float32)
        if torch.cuda.is_available():
            self.device = torch.device('cuda', index=0) 
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cpu')
        
        torch.cuda.empty_cache()
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
             
            self.num_timesteps_out[i_sample] = len_use
                    
        if len(self.num_timesteps_out) < 2500:
            self.num_timesteps_out = np.minimum(self.num_timesteps_out, 100)
        else:
            self.num_timesteps_out = np.minimum(self.num_timesteps_out, 50)
        
        max_agents = 200
        num_agents = len(self.Input_path_train.columns) / 2
        
        max_timesteps = 20
        num_timesteps = self.num_timesteps_out.max()
        
        self.batch_size = int(np.floor((max_agents / num_agents) * (max_timesteps / num_timesteps)))
        
        
        

    def train_method(self):
        num_samples_path_pred_train = int(self.num_samples_path_pred/10)
        # Prepare data preliminary
        X_help = self.Input_path_train.to_numpy()
        X_help = X_help.reshape(len(X_help), -1, 2)
        Y_help = self.Output_path_train.to_numpy() 
        
        X = np.zeros(list(X_help.shape) + [self.num_timesteps_in])
        
        for i_sample in range(X.shape[0]):
            for i_v in range(X.shape[1]):
                for dim in range(2):
                    X[i_sample, i_v, dim, :] = X_help[i_sample, i_v, dim]
        
        # Assume that target vehicle will always be vehicle or pedestrian
        Y = np.ones(list(Y_help.shape) + [self.num_timesteps_out.max()]) * np.nan
        
        for i_sample in range(X.shape[0]):
            for dim in range(2):
                Y[i_sample, dim, :self.num_timesteps_out[i_sample]] = Y_help[i_sample, dim][:self.num_timesteps_out[i_sample]]
        
        # prepare data
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        Data = []
        out = max(self.num_timesteps_out)
        for i in range(len(X)):
            pre_motion_3D = [x.t() for x in X[i]]
            pre_motion_mask = [torch.ones(self.num_timesteps_in) for i in range(len(pre_motion_3D))] 
            
            fut_motion_3D = [torch.zeros((out,2)) for i in range(len(pre_motion_3D))] 
            fut_motion_3D[1][:self.num_timesteps_out[i],:] = Y[i,:,:self.num_timesteps_out[i]].t()
            
            fut_motion_mask = [torch.zeros(out) for i in range(len(pre_motion_3D))] 
            fut_motion_mask[1][:self.num_timesteps_out[i]] = 1.0
            
            data = {
                'pre_motion_3D': pre_motion_3D,
                'fut_motion_3D': fut_motion_3D,
                'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask,
                'pre_data': None,
                'fut_data': None,
                'heading': None,
                'valid_id': [1.0, 2.0, 3.0, 4.0, 5.0],
                'traj_scale': 1,
                'pred_mask': None,
                'scene_map': None,
                'seq': 'Not_needed',
                'frame': i
            }
            
            Data.append(data)
        
        Data = np.array(Data)
        Index = np.argsort(-self.num_timesteps_out)
        
        max_batch_size_vae = self.batch_size
        Index_batches_vae = []
        batch_vae = []
        batch_size_vae = 0
        for i, ind in enumerate(Index):
            batch_vae.append(ind)
            batch_size_vae += self.num_timesteps_out[ind] / self.num_timesteps_out.max()
            if batch_size_vae >= max_batch_size_vae or i == len(Index) - 1:
                Index_batches_vae.append(batch_vae)
                batch_vae = []
                batch_size_vae = 0
                
        Batches_vae = np.arange(len(Index_batches_vae))
        
        max_batch_size_dlow = self.batch_size / 10
        Index_batches_dlow = []
        batch_dlow = []
        batch_size_dlow = 0
        for i, ind in enumerate(Index):
            batch_dlow.append(ind)
            batch_size_dlow += self.num_timesteps_out[ind] / self.num_timesteps_out.max()
            if batch_size_dlow >= max_batch_size_dlow or i == len(Index) - 1:
                Index_batches_dlow.append(batch_dlow)
                batch_dlow = []
                batch_size_dlow = 0
                
        Batches_dlow = np.arange(len(Index_batches_dlow))
        
        
        ######################################################################
        ##                Train VAE                                         ##
        ######################################################################

    
        # load hyperparams and set up model
        cfg = Config('hyperparams_pre', False, create_dirs = False)        
        cfg.yml_dict["past_frames"] = self.num_timesteps_in
        cfg.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg.yml_dict["future_frames"] = max(self.num_timesteps_out)
        cfg.yml_dict["min_future_frames"] = min(self.num_timesteps_out)
        
        cfg.yml_dict["sample_k"] = num_samples_path_pred_train
        cfg.yml_dict["loss_cfg"]["sample"]["k"] = num_samples_path_pred_train
        
        # load model
        model_id = cfg.get('model_id', 'agentformer')
        self.model_vae = model_dict[model_id](cfg)
        
        optimizer = optim.Adam(self.model_vae.parameters(), lr=cfg.lr)
        scheduler_type = cfg.get('lr_scheduler', 'linear')
        if scheduler_type == 'linear':
            scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
        elif scheduler_type == 'step':
            scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
        else:
            raise ValueError('unknown scheduler type!')
            
        self.model_vae.set_device(self.device)
        self.model_vae.train()
        
        # train vae model
        epochs = cfg.yml_dict["num_epochs"]
        for epoch in range(1, epochs + 1):
            print('Train VAE: Epoch ' + 
                  str(epoch).rjust(len(str(epochs))) + 
                  '/{}'.format(epochs))
            np.random.shuffle(Batches_vae)
            epoch_loss = 0.0
            if epoch < epochs - 5:
                for i, ind in enumerate(Batches_vae):
                    print('Train VAE: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch '.format(epochs) + 
                          str(i + 1).rjust(len(str(len(Batches_vae)))) + 
                          '/{}'.format(len(Batches_vae)))
                    data = Data[Index_batches_vae[ind]]
                    self.model_vae.future_decoder.future_frames = self.num_timesteps_out[Index_batches_vae[ind]].max()
                    self.model_vae.set_data(data)
                    self.model_vae()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_vae.compute_loss()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.detach().cpu().numpy()
            else:
                optimizer.zero_grad()
                for i, ind in enumerate(Batches_vae): 
                    print('Train VAE: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch '.format(epochs) + 
                          str(i + 1).rjust(len(str(len(Batches_vae)))) + 
                          '/{}'.format(len(Batches_vae)))
                    data = Data[Index_batches_vae[ind]]
                    self.model_vae.future_decoder.future_frames = self.num_timesteps_out[Index_batches_vae[ind]].max()
                    self.model_vae.set_data(data)
                    self.model_vae()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_vae.compute_loss()
                    total_loss.backward()
                    epoch_loss += total_loss.detach().cpu().numpy()       
                optimizer.step()
            print('Train VAE: Epoch ' + str(epoch).rjust(len(str(epochs))) + 
                  '/{} with loss {:0.3f}'.format(epochs, epoch_loss/len(Index)))
            print('')
        
        blocked_numbers = [int(f[10:-2]) for f in os.listdir('Models/agentformer') 
                           if f[-2:] == '.p' and f[:9] == 'model_vae']
        available_numbers = np.array([i for i in np.arange(10000)
                                      if i not in blocked_numbers])
        model_number = np.random.choice(available_numbers)
        cp_path = 'Models/agentformer\\model_vae_{}.p'.format(model_number)
        model_cp = {'model_dict': self.model_vae.state_dict(), 'opt_dict': optimizer.state_dict(), 
                    'scheduler_dict': scheduler.state_dict(), 'epoch': epochs}
        torch.save(model_cp, cp_path)   
         
        # save weights
        Weights_vae = list(self.model_vae.parameters())
        self.weights_vae = []
        for weigths in Weights_vae:
            self.weights_vae.append(weigths.detach().cpu().numpy())

        ######################################################################
        ##                Train DLow                                        ##
        ######################################################################
        
        cfg = Config('hyperparams', False, create_dirs = False)   
        cfg.yml_dict['model_number'] = model_number
        cfg.yml_dict["past_frames"] = self.num_timesteps_in
        cfg.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg.yml_dict["future_frames"] = max(self.num_timesteps_out)
        cfg.yml_dict["min_future_frames"] = min(self.num_timesteps_out)
        
        cfg.yml_dict["sample_k"] = self.num_samples_path_pred
        
        # create model
        model_id = cfg.get('model_id', 'agentformer')
        self.model_dlow = model_dict[model_id](cfg)
        
        optimizer = optim.Adam(self.model_dlow.parameters(), lr=cfg.lr)
        scheduler_type = cfg.get('lr_scheduler', 'linear')
        if scheduler_type == 'linear':
            scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
        elif scheduler_type == 'step':
            scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
        else:
            raise ValueError('unknown scheduler type!')
            
        
        self.model_dlow.set_device(self.device)
        self.model_dlow.train()
            
        # train vae model
        epochs = cfg.yml_dict["num_epochs"]
        for epoch in range(1, epochs + 1):
            print('Train DLow: Epoch ' + 
                  str(epoch).rjust(len(str(epochs))) + 
                  '/{}'.format(epochs))
            np.random.shuffle(Batches_dlow)
            epoch_loss = 0.0
            if epoch < epochs - 5:
                for i, ind in enumerate(Batches_dlow):
                    print('Train DLow: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch '.format(epochs) + 
                          str(i + 1).rjust(len(str(len(Batches_dlow)))) + 
                          '/{}'.format(len(Batches_dlow)))
                    data = Data[Index_batches_dlow[ind]]
                    self.model_dlow.pred_model[0].future_decoder.future_frames = self.num_timesteps_out[Index_batches_dlow[ind]].max()
                    self.model_dlow.set_data(data)
                    self.model_dlow()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_dlow.compute_loss()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.detach().cpu().numpy()
            else:
                optimizer.zero_grad()
                for i, ind in enumerate(Batches_dlow):
                    print('Train DLow: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch '.format(epochs) + 
                          str(i + 1).rjust(len(str(len(Batches_dlow)))) + 
                          '/{}'.format(len(Batches_dlow)))
                    data = Data[Index_batches_dlow[ind]]
                    self.model_dlow.pred_model[0].future_decoder.future_frames = self.num_timesteps_out[Index_batches_dlow[ind]].max()
                    self.model_dlow.set_data(data)
                    self.model_dlow()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_dlow.compute_loss()
                    total_loss.backward()
                    epoch_loss += total_loss.detach().cpu().numpy()
                optimizer.step()
            print('Train DLow: Epoch ' + str(epoch).rjust(len(str(epochs))) + 
                  '/{} with loss {:0.3f}'.format(epochs, epoch_loss/len(Index)))
            print('')  
         
        # save weights
        Weights_dlow = list(self.model_dlow.parameters())
        self.weights_dlow = []
        for weigths in Weights_dlow:
            self.weights_dlow.append(weigths.detach().cpu().numpy()) 
        
            
        os.remove(cp_path)
        self.weights_saved = [self.weights_vae, self.weights_dlow]
        
        
    def load_method(self, l2_regulization = 0):
        [self.weights_vae, self.weights_dlow] = self.weights_saved
        
        num_samples_path_pred_train = int(self.num_samples_path_pred/10)
        ######################################################################
        ##                Load VAE                                          ##
        ######################################################################
        
        cfg = Config('hyperparams_pre', False, create_dirs = False)        
        cfg.yml_dict["past_frames"] = self.num_timesteps_in
        cfg.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg.yml_dict["future_frames"] = max(self.num_timesteps_out)
        cfg.yml_dict["min_future_frames"] = min(self.num_timesteps_out)
        
        cfg.yml_dict["sample_k"] = num_samples_path_pred_train
        cfg.yml_dict["loss_cfg"]["sample"]["k"] = num_samples_path_pred_train
        
        # load model
        model_id = cfg.get('model_id', 'agentformer')
        self.model_vae = model_dict[model_id](cfg)
        optimizer = optim.Adam(self.model_vae.parameters(), lr=cfg.lr)
        scheduler_type = cfg.get('lr_scheduler', 'linear')
        if scheduler_type == 'linear':
            scheduler = get_scheduler(optimizer, policy='lambda', 
                                      nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
        elif scheduler_type == 'step':
            scheduler = get_scheduler(optimizer, policy='step', 
                                      decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
        else:
            raise ValueError('unknown scheduler type!')
            
            
        Weights_vae = list(self.model_vae.parameters())
        with torch.no_grad():
            for i, weights in enumerate(self.weights_vae):
                Weights_vae[i][:] = torch.from_numpy(weights)[:]
        
        blocked_numbers = [int(f[10:-2]) for f in os.listdir('Models/agentformer') 
                           if f[-2:] == '.p' and f[:9] == 'model_vae']
        available_numbers = np.array([i for i in np.arange(10000)
                                      if i not in blocked_numbers])
        model_number = np.random.choice(available_numbers)
        cp_path = 'Models/agentformer\\model_vae_{}.p'.format(model_number)
        model_cp = {'model_dict': self.model_vae.state_dict(), 'opt_dict': optimizer.state_dict(), 
                    'scheduler_dict': scheduler.state_dict(), 'epoch': cfg.yml_dict["num_epochs"]}
        torch.save(model_cp, cp_path)   
        
        ######################################################################
        ##                Load DLow                                         ##
        ######################################################################
        
        cfg = Config('hyperparams', False, create_dirs = False) 
        cfg.yml_dict['model_number'] = model_number       
        cfg.yml_dict["past_frames"] = self.num_timesteps_in
        cfg.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg.yml_dict["future_frames"] = max(self.num_timesteps_out)
        cfg.yml_dict["min_future_frames"] = min(self.num_timesteps_out)
        
        cfg.yml_dict["sample_k"] = self.num_samples_path_pred
        
        # create model
        model_id = cfg.get('model_id', 'agentformer')
        self.model_dlow = model_dict[model_id](cfg)
        
        Weights_dlow = list(self.model_dlow.parameters())
        with torch.no_grad():
            for i, weights in enumerate(self.weights_dlow):
                Weights_dlow[i][:] = torch.from_numpy(weights)[:]
        
        os.remove(cp_path)

        
    def predict_method(self):
        # get desired output length
        X_help = self.Input_path_test.to_numpy()
        X_help = X_help.reshape(len(X_help), -1, 2)
        
        X = np.zeros(list(X_help.shape) + [self.num_timesteps_in])
        
        for i_sample in range(X.shape[0]):
            for i_v in range(X.shape[1]):
                for dim in range(2):
                    X[i_sample, i_v, dim, :] = X_help[i_sample, i_v, dim]
        
        self.num_timesteps_out_test = np.zeros(len(self.Output_T_test), int)
        for i in range(len(self.Output_T_test)):
            self.num_timesteps_out_test[i] = len(self.Output_T_test[i])
            
        # prepare data
        X = torch.from_numpy(X).float()
        Data = []
        out = max(self.num_timesteps_out_test)
        for i in range(len(X)):
            pre_motion_3D = [x.t() for x in X[i]]
            pre_motion_mask = [torch.ones(self.num_timesteps_in) for i in range(len(pre_motion_3D))] 
            
            fut_motion_3D = [torch.zeros((out,2)) for i in range(len(pre_motion_3D))] 
            
            fut_motion_mask = [torch.zeros(out) for i in range(len(pre_motion_3D))] 
            fut_motion_mask[1][:self.num_timesteps_out_test[i]] = 1.0
            
            data = {
                'pre_motion_3D': pre_motion_3D,
                'fut_motion_3D': fut_motion_3D,
                'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask,
                'pre_data': None,
                'fut_data': None,
                'heading': None,
                'valid_id': [1.0, 2.0, 3.0, 4.0, 5.0],
                'traj_scale': 1,
                'pred_mask': None,
                'scene_map': None,
                'seq': 'Not_needed',
                'frame': i
            }
            
            Data.append(data)
        
        Data = np.array(Data)
        
        Index = np.argsort(-self.num_timesteps_out_test)
        
        max_batch_size_dlow = self.batch_size / 10
        Index_batches_dlow = []
        batch_dlow = []
        batch_size_dlow = 0
        for i, ind in enumerate(Index):
            batch_size_dlow_new = self.num_timesteps_out_test[ind] / self.num_timesteps_out.max()
            if batch_size_dlow + batch_size_dlow_new >= max_batch_size_dlow and i < len(Index) - 1:
                if i > 0:
                    Index_batches_dlow.append(batch_dlow)
                    batch_dlow = []
                batch_size_dlow = batch_size_dlow_new
                batch_dlow.append(ind)
                batch_dlow 
            elif i == len(Index) - 1:
                if batch_size_dlow + batch_size_dlow_new >= max_batch_size_dlow:
                    Index_batches_dlow.append(batch_dlow)
                    batch_dlow = []
                    batch_dlow.append(ind)
                    Index_batches_dlow.append(batch_dlow)
                else:
                    batch_dlow.append(ind)
                    Index_batches_dlow.append(batch_dlow)
                
            else:
                batch_dlow.append(ind)
                batch_size_dlow += batch_size_dlow_new
                
        Batches_dlow = np.arange(len(Index_batches_dlow))
        
        Output_Path = pd.DataFrame(np.empty((X.shape[0], 2), object), columns = self.Output_path_train.columns)
            
        self.model_dlow.set_device(self.device)
        self.model_dlow.eval()
        
        for i in Batches_dlow:
            print('Eval DLow: Batch ' + 
                  str(i + 1).rjust(len(str(len(Batches_dlow)))) + 
                  '/{}'.format(len(Batches_dlow)))
            
            # check if problem was already solved in saved data
            index_batch = Index_batches_dlow[i]
            
            
            data = Data[index_batch]
            # OOM protection
            if len(data) == 1 and self.num_timesteps_out_test[index_batch][0] / self.num_timesteps_out.max() > max_batch_size_dlow:
                num_frames = int(max_batch_size_dlow * self.num_timesteps_out.max())
                with torch.no_grad():
                    self.model_dlow.pred_model[0].future_decoder.future_frames = num_frames
                    self.model_dlow.set_data(data)
                    sample_motion_3D, _ = self.model_dlow.inference(mode = 'infer', 
                                                                    sample_num = self.num_samples_path_pred)
                pred = sample_motion_3D[:,1].detach().cpu().numpy()
                Vel_last = pred[:,:,[-1]] - pred[:,:,[-2]]
                steps = np.arange(1, self.num_timesteps_out_test[index_batch][0] - num_frames + 1)
                pred_extended = Vel_last * steps[np.newaxis,np.newaxis,:,np.newaxis] + pred[:,:,[-1]]
                pred = np.concatenate((pred, pred_extended), axis = 2)
                for j, ind in enumerate(index_batch):
                    Output_Path.iloc[ind,0] = pred[j,:, :self.num_timesteps_out_test[ind], 0]
                    Output_Path.iloc[ind,1] = pred[j,:, :self.num_timesteps_out_test[ind], 1]
            else:
                with torch.no_grad():
                    self.model_dlow.pred_model[0].future_decoder.future_frames = self.num_timesteps_out_test[index_batch].max()
                    self.model_dlow.set_data(data)
                    sample_motion_3D, _ = self.model_dlow.inference(mode = 'infer', 
                                                                    sample_num = self.num_samples_path_pred)
                pred = sample_motion_3D[:,1].detach().cpu().numpy()
                for j, ind in enumerate(index_batch):
                    Output_Path.iloc[ind,0] = pred[j,:, :self.num_timesteps_out_test[ind], 0]
                    Output_Path.iloc[ind,1] = pred[j,:, :self.num_timesteps_out_test[ind], 1]
                    
        return [Output_Path]

    def check_input_names_method(self, names, train = True):
        if all(names == self.input_names_train):
            if names[2][-5:] == 'tar_x' and names[3][-5:] == 'tar_y':
                return True
        
        return False
    
    def get_output_type_class():
        return 'path'
    
    def get_name(self):
        return 'agentformer'
        

        
        
        
        
        
    
        
        
        