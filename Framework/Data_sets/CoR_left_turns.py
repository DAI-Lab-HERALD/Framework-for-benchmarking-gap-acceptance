import numpy as np
import pandas as pd
from data_set_template import data_set_template
import os
class CoR_left_turns(data_set_template):
   
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + '/Data_raw/CoR - Unprotected left turns/CoR_processed.pkl')
        # analize raw dara 
        self.num_samples = len(self.Data)
        self.Path = []
        self.T = []
        self.Domain_old = []
        names = ['V_ego_x', 'V_ego_y', 'V_tar_x', 'V_tar_y']
        # extract raw samples
        for i in range(self.num_samples):
            path = pd.Series(np.empty(len(names), tuple), index = names)
            
            t_index = self.Data.bot_track.iloc[i].index
            t = tuple(self.Data.bot_track.iloc[i].t[t_index])
            path.V_ego_x = tuple(self.Data.bot_track.iloc[i].x[t_index])
            path.V_ego_y = tuple(self.Data.bot_track.iloc[i].y[t_index])
            path.V_tar_x = tuple(self.Data.ego_track.iloc[i].x[t_index])
            path.V_tar_y = tuple(self.Data.ego_track.iloc[i].y[t_index])

            domain = pd.Series(np.ones(1, int) * self.Data.subj_id.iloc[i], index = ['Subj_ID'])
            
            self.Path.append(path)
            self.T.append(t)
            self.Domain_old.append(domain)
        
        self.Path = pd.DataFrame(self.Path)
        self.T = np.array(self.T+[()], tuple)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
     
    
    def position_states_in_path(self, path, t, domain):
        ego_x = np.array(path.V_ego_x)
        tar_y = np.array(path.V_tar_y)
        
        try:
            i_end = np.where(ego_x > - 20)[0][0]
        except:
            i_end = len(ego_x) - 1
        dxdt = (ego_x[i_end] - ego_x[0]) / (t[i_end] - t [0])
        
        
        # NOTE: I assume lane width of 3m here, but further research necessary
        offset = 3
        dx = - ego_x - offset
        dtc = dx / dxdt
        tchat = dtc + t
        # determine t_crit
        a_brake = 4 # I am not sure about this
        # especially when differentiating between what is technically possible and 
        # what is still safe on the road
        
        x_brake = 0.5 * max(dxdt, 0) ** 2 / a_brake
        
        undecided = (dx > 0) & (tar_y > 0)
        turned    = (dx > 0) & (tar_y < 0)
        passed    = (dx < 0) & (tar_y > 0)
        critical = (x_brake > dx) & undecided
        
        # set interpolation parameters, which are negative during undecided and positive afterwards
        undecided_2_turned = -tar_y
        undecided_2_passed = -dx
        not_critical_2_critical = x_brake - dx

        return [undecided, turned, passed, critical, 
                undecided_2_turned, undecided_2_passed, not_critical_2_critical, tchat]
    
    
    
    def path_to_binary_and_time_sample(self, output_path, output_t, domain):
        # Assume that t_A is when vehicle crosses into street,
        # One could also assume reaction as defined by Zgonnikov et al. (tar_ax > 0)
        # but that might be stupid
        tar_y = output_path.V_tar_y
        t = output_t
        
        tar_yh = tar_y[np.invert(np.isnan(tar_y))]
        th = t[np.invert(np.isnan(tar_y))]
        try:
            # assume width of car of 1 m
            # interpolate here 
            ind_2 = np.where(tar_yh < 0)[0][0]
            if ind_2 == 0:
                output_t_e = th[ind_2] - 0.5 * self.dt
                output_a = True
            else:
                ind_1 = ind_2 - 1
                tar_y_1 = tar_yh[ind_1]
                tar_y_2 = tar_yh[ind_2]
                fac = np.abs(tar_y_1) / (np.abs(tar_y_1) + np.abs(tar_y_2))
                output_t_e = th[ind_1] * (1 - fac) + th[ind_2] * fac
                output_a = True
        except:
            output_t_e = None
            output_a  = False
        
        return output_a, output_t_e
    
    def fill_empty_input_path(self):
        # No processing necessary
        pass
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_solid.append(np.array([[-300, -4],[-4, -4],[-4, -300]]))
        lines_solid.append(np.array([[300, -4],[4, -4],[4, -300]]))
        lines_solid.append(np.array([[-300, 4],[-4, 4],[-4, 300]]))
        lines_solid.append(np.array([[300, 4],[4, 4],[4, 300]]))
        
        lines_solid.append(np.array([[-4, -4],[-4, 0],[-6, 0]]))
        lines_solid.append(np.array([[4, 4],[4, 0],[6, 0]]))
        lines_solid.append(np.array([[4, -4],[0, -4],[0, -6]]))
        lines_solid.append(np.array([[-4, 4],[0, 4],[0, 6]]))
        
        lines_dashed = []
        lines_dashed.append(np.array([[0, 6],[0, 300]]))
        lines_dashed.append(np.array([[0, -6],[0, -300]]))
        lines_dashed.append(np.array([[6, 0],[300, 0]]))
        lines_dashed.append(np.array([[-6, 0],[-300, 0]]))
        
        
        return lines_solid, lines_dashed

