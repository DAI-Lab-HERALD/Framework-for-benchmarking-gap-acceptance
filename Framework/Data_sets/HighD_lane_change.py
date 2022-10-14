import numpy as np
import pandas as pd
from data_set_template import data_set_template
import os
from scipy import interpolate

class HighD_lane_change(data_set_template):
   
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + '/Data_raw/HighD - Lane changes and merging/HighD_processed.pkl')
        # analize raw dara 
        num_samples_max = len(self.Data)
        self.Path = []
        self.T = []
        self.Domain_old = []
        # v_1 is the vehicle in front of the ego vehicle
        # v_2 is the vehicle following the target vehicle
        # v_3 is the vehicle the target vehicle is following 
        names = ['V_ego_x', 'V_ego_y', 
                 'V_tar_x', 'V_tar_y', 
                 'V_v_1_x', 'V_v_1_y',
                 'V_v_2_x', 'V_v_2_y',
                 'V_v_3_x', 'V_v_3_y']
        # extract raw samples
        self.num_samples = 0
        for i in range(num_samples_max):
            # to keep track:
            if np.mod(i,1000) == 0:
                print('trajectory ' + str(i).rjust(len(str(num_samples_max))) + '/{} analized'.format(num_samples_max))
            
            data_i = self.Data.iloc[i]
            track_i = data_i.track[['frame','x', 'xVelocity', 
                                    'y', 'laneId',
                                    'followingId',
                                    'precedingId',
                                    'leftFollowingId',
                                    'leftAlongsideId',
                                    'leftPrecedingId']]
            
            # check if there is even a vehicle that might offer a gap
            
            Accepted_indices = []
            Denied_indices = []
            if track_i.leftFollowingId.max() == 0:
                continue
            
            includes_t_A_or_t_C = False
            # check for t_C
            if any(np.logical_and((track_i.leftFollowingId.iloc[:-1].to_numpy() == 
                                   track_i.leftAlongsideId.iloc[1:].to_numpy()),
                                  (track_i.leftAlongsideId.iloc[1:].to_numpy() != 0))):
                includes_t_A_or_t_C = True
                Denied_indices = np.where(np.logical_and((track_i.leftFollowingId.iloc[:-1].to_numpy() == 
                                                     track_i.leftAlongsideId.iloc[1:].to_numpy()),
                                                    (track_i.leftAlongsideId.iloc[1:].to_numpy() != 0)))[0] + 1
            
            # check for any lane change
            if data_i.numLaneChanges > 0 and any((track_i.laneId.iloc[1:].to_numpy() - 
                                                  track_i.laneId.iloc[:-1].to_numpy()) == 1):
                # check if the time difference here was enough to make it relevant.
                Lane_change_indices = np.where(np.logical_and((track_i.laneId.iloc[1:].to_numpy() - 
                                                               track_i.laneId.iloc[:-1].to_numpy()) == 1, 
                                                              track_i.laneId.iloc[:-1].to_numpy() > 0))[0] + 1
                
                for lane_change_index in Lane_change_indices:
                    ego_id = track_i.followingId.iloc[lane_change_index]
                    if ego_id == 0:
                        continue
                    else:
                        frame_id = track_i.frame.iloc[lane_change_index]
                        ego_track = self.Data[self.Data.id == ego_id].track.values[0]
                        ego_track = ego_track[ego_track.frame == frame_id]
                        dx = float(track_i.iloc[lane_change_index].x) - float(ego_track.x)
                        dv = float(track_i.iloc[lane_change_index].xVelocity) - float(ego_track.xVelocity)
                        
                        # this is a very generous amount
                        if (0 < dv) or dx < 100:
                            includes_t_A_or_t_C = True
                            Accepted_indices.append(lane_change_index)
                            continue
                                
                    
                    
            if not includes_t_A_or_t_C:
                continue
            
            # check for denied if they all refer to the same ego vehicle or different ones
            Denied_ego_id = [track_i.leftAlongsideId.iloc[index] for index in Denied_indices]
            _, true_Denied = np.unique(Denied_ego_id, return_index = True)
            
            Denied_indices = list(np.sort(np.array(Denied_indices)[true_Denied]))
                
            
            for d_index in Denied_indices:
                path = pd.Series(np.empty(len(names), tuple), index = names)
                domain = pd.Series(np.zeros(4, int), index = ['Loc_ID', 'Type_ID', 'Intent_ID', 'TS_ID'])
                domain.Loc_ID = data_i.locationId
                if data_i['class'] == 'Truck':
                    domain.Type_ID = 1
                # check if target vehicle later performed a lane change anyway
                check_accepted = [a_i > d_index for a_i in Accepted_indices]
                if any(check_accepted):
                    a_possible = Accepted_indices[np.where(check_accepted)[0][0]] 
                    if all(track_i.laneId.iloc[d_index : a_possible - 1] == track_i.laneId.iloc[d_index]):
                        domain.Intent_ID = 1
                
                
                frame_id = track_i.frame.iloc[d_index]
                ego_id = track_i.leftAlongsideId.iloc[d_index]
                v_1_id = track_i.leftPrecedingId.iloc[d_index]
                v_2_id = track_i.followingId.iloc[d_index]
                v_3_id = track_i.precedingId.iloc[d_index]
                
                
                if data_i.laneMarkings[0] < 0:
                    y_offset = data_i.laneMarkings[track_i.laneId.iloc[d_index] + 1]
                else:
                    y_offset = data_i.laneMarkings[track_i.laneId.iloc[d_index]]
                
                ego_track = self.Data[self.Data.id == ego_id].track.values[0]
                
                frame_min = max(track_i.frame.min(), ego_track.frame.min())
                frame_max = min(track_i.frame.max(), ego_track.frame.max())
                
                if frame_id - 5 < frame_min:
                    continue
                
                tar_track = track_i.set_index('frame').loc[frame_min:frame_max]
                ego_track = ego_track.set_index('frame').loc[frame_min:frame_max]
                
                right_pos = ((tar_track.laneId.loc[frame_min:frame_id] == tar_track.laneId.loc[frame_id]) &
                             (ego_track.laneId.loc[frame_min:frame_id] == ego_track.laneId.loc[frame_id]))
                
                if not right_pos.all():
                    frame_min = right_pos.index[np.invert(right_pos)][-1] + 1
                    tar_track = tar_track.loc[frame_min:frame_max]
                    ego_track = ego_track.loc[frame_min:frame_max]
                
                if frame_max - frame_min < 50:
                    continue
                # check if v_1_id existed as leftAlongsideId at some point,
                # meaning that T_S is included in this sample
                if (v_1_id != 0) and (v_1_id in list(tar_track.leftAlongsideId)):
                    domain.TS_ID = 1
                
                # check conditions for incöusion
                if domain.Intent_ID == 0:
                    if domain.TS_ID == 1:
                        if domain.Type_ID == 1:
                            if np.random.rand() > 0.1:
                                continue
                        else:
                            if np.random.rand() > 0.3:
                                continue
                    else:
                        if domain.Type_ID == 1:
                            if np.random.rand() > 0.02:
                                continue
                        else:
                            if np.random.rand() > 0.08:
                                continue
                
                
                    
                t = tuple(ego_track.index / 25)
                path.V_ego_x = tuple(ego_track.x)
                path.V_ego_y = tuple(ego_track.y - y_offset)
                path.V_tar_x = tuple(tar_track.x)
                path.V_tar_y = tuple(tar_track.y - y_offset)
                
                
                if v_1_id > 0:
                    v_1_track = self.Data[self.Data.id == v_1_id].track.values[0].set_index('frame').loc[frame_min:frame_max]
                    
                    frame_min_v1 = v_1_track.index.min()
                    frame_max_v1 = v_1_track.index.max()
                    
                    v1x = np.ones(frame_max + 1 - frame_min) * np.nan
                    v1x[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.x
                    
                    v1y = np.ones(frame_max + 1 - frame_min) * np.nan
                    v1y[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.y
                    
                    path.V_v_1_x = tuple(v1x)
                    path.V_v_1_y = tuple(v1y - y_offset)
                else:
                    path.V_v_1_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_1_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                    
                if v_2_id > 0:
                    v_2_track = self.Data[self.Data.id == v_2_id].track.values[0].set_index('frame').loc[frame_min:frame_max]
                    
                    frame_min_v2 = v_2_track.index.min()
                    frame_max_v2 = v_2_track.index.max()
                    
                    v2x = np.ones(frame_max + 1 - frame_min) * np.nan
                    v2x[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.x
                    
                    v2y = np.ones(frame_max + 1 - frame_min) * np.nan
                    v2y[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.y
                    
                    path.V_v_2_x = tuple(v2x)
                    path.V_v_2_y = tuple(v2y - y_offset)
                else:
                    path.V_v_2_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_2_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                if v_3_id > 0:
                    v_3_track = self.Data[self.Data.id == v_3_id].track.values[0].set_index('frame').loc[frame_min:frame_max]
                    
                    frame_min_v3 = v_3_track.index.min()
                    frame_max_v3 = v_3_track.index.max()
                    
                    v3x = np.ones(frame_max + 1 - frame_min) * np.nan
                    v3x[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.x
                    
                    v3y = np.ones(frame_max + 1 - frame_min) * np.nan
                    v3y[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.y
                    
                    path.V_v_3_x = tuple(v3x)
                    path.V_v_3_y = tuple(v3y - y_offset)
                else:
                    path.V_v_3_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_3_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)

                self.Path.append(path)
                self.T.append(t)
                self.Domain_old.append(domain)
                self.num_samples = self.num_samples + 1
                
                
            for a_index in Accepted_indices:
                path = pd.Series(np.empty(len(names), tuple), index = names)
                domain = pd.Series(np.zeros(4, int), index = ['Loc_ID', 'Type_ID', 'Intent_ID', 'TS_ID'])
                domain.Loc_ID = data_i.locationId
                if data_i['class'] == 'Truck':
                    domain.Type_ID = 1
                # The vehicle merged, so there was obviously intent to cross the street
                domain.Intent_ID = 1
                
                frame_id = track_i.frame.iloc[a_index]
                ego_id = track_i.followingId.iloc[a_index]
                v_1_id = track_i.precedingId.iloc[a_index]
                v_2_id = track_i.followingId.iloc[a_index - 1]
                v_3_id = track_i.precedingId.iloc[a_index - 1]
                
                if data_i.laneMarkings[0] < 0:
                    y_offset = data_i.laneMarkings[track_i.laneId.iloc[a_index - 1] + 1]
                else:
                    y_offset = data_i.laneMarkings[track_i.laneId.iloc[a_index - 1]]
                
                
                ego_track = self.Data[self.Data.id == ego_id].track.values[0]
                
                frame_min = max(track_i.frame.min(), ego_track.frame.min())
                frame_max = min(track_i.frame.max(), ego_track.frame.max())
                
                if frame_id - 5 < frame_min:
                    continue
                
                tar_track = track_i.set_index('frame').loc[frame_min:frame_max]
                ego_track = ego_track.set_index('frame').loc[frame_min:frame_max]
                
                # check if v_1_id existed as leftAlongsideId at some point,
                # meaning that t_S is included in this sample
                if (v_1_id != 0) and (v_1_id in list(tar_track.leftAlongsideId)):
                    domain.TS_ID = 1

                # check conditions:
                if domain.TS_ID == 0:
                    dx = (tar_track.loc[:frame_id].x - ego_track.loc[:frame_id].x).to_numpy()
                    dv = np.maximum(5, - (tar_track.loc[:frame_id].x - ego_track.loc[:frame_id].x).to_numpy())
                    dt = dx / dv
                    if np.min(dt) > 10:
                        continue
                    
                right_pos = ((tar_track.laneId.loc[frame_min:frame_id - 1] == tar_track.laneId.loc[frame_id - 1]) &
                             (ego_track.laneId.loc[frame_min:frame_id - 1] == ego_track.laneId.loc[frame_id - 1]))

                
                
                if not right_pos.all():
                    frame_min = right_pos.index[np.invert(right_pos)][-1] + 1
                    tar_track = tar_track.loc[frame_min:frame_max]
                    ego_track = ego_track.loc[frame_min:frame_max]
                    
                if frame_max - frame_min < 50:
                    continue
                
                t = tuple(ego_track.index / 25)
                path.V_ego_x = tuple(ego_track.x)
                path.V_ego_y = tuple(ego_track.y - y_offset)
                path.V_tar_x = tuple(tar_track.x)
                path.V_tar_y = tuple(tar_track.y - y_offset)
                
                
                if v_1_id > 0:
                    v_1_track = self.Data[self.Data.id == v_1_id].track.values[0].set_index('frame').loc[frame_min:frame_max]
                    
                    frame_min_v1 = v_1_track.index.min()
                    frame_max_v1 = v_1_track.index.max()
                    
                    v1x = np.ones(frame_max + 1 - frame_min) * np.nan
                    v1x[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.x
                    
                    v1y = np.ones(frame_max + 1 - frame_min) * np.nan
                    v1y[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.y
                    
                    path.V_v_1_x = tuple(v1x)
                    path.V_v_1_y = tuple(v1y - y_offset)
                else:
                    path.V_v_1_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_1_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                    
                if v_2_id > 0:
                    v_2_track = self.Data[self.Data.id == v_2_id].track.values[0].set_index('frame').loc[frame_min:frame_max]
                    
                    frame_min_v2 = v_2_track.index.min()
                    frame_max_v2 = v_2_track.index.max()
                    
                    v2x = np.ones(frame_max + 1 - frame_min) * np.nan
                    v2x[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.x
                    
                    v2y = np.ones(frame_max + 1 - frame_min) * np.nan
                    v2y[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.y
                    
                    path.V_v_2_x = tuple(v2x)
                    path.V_v_2_y = tuple(v2y - y_offset)
                else:
                    path.V_v_2_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_2_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    
                if v_3_id > 0:
                    v_3_track = self.Data[self.Data.id == v_3_id].track.values[0].set_index('frame').loc[frame_min:frame_max]
                    
                    frame_min_v3 = v_3_track.index.min()
                    frame_max_v3 = v_3_track.index.max()
                    
                    v3x = np.ones(frame_max + 1 - frame_min) * np.nan
                    v3x[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.x
                    
                    v3y = np.ones(frame_max + 1 - frame_min) * np.nan
                    v3y[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.y
                    
                    path.V_v_3_x = tuple(v3x)
                    path.V_v_3_y = tuple(v3y - y_offset)
                else:
                    path.V_v_3_x = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)
                    path.V_v_3_y = tuple(np.ones(frame_max + 1 - frame_min) * np.nan)

                self.Path.append(path)
                self.T.append(t)
                self.Domain_old.append(domain)
                self.num_samples = self.num_samples + 1
        
        self.Path = pd.DataFrame(self.Path)
        self.T = np.array(self.T+[()], tuple)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
         
    def position_states_in_path(self, path, t, domain):
        ego_x = np.array(path.V_ego_x)
        ego_y = np.array(path.V_ego_y)
        tar_x = np.array(path.V_tar_x)
        tar_y = np.array(path.V_tar_y)
        
        # assumed vehicle lenght
        offset = 5
        
        # possibly substract the length of target vehicle here
        # dx is positive if the gap is not yet closed by ego vehicle
        dx = tar_x - ego_x - offset
        
        # dxdt is positive if ego is faster
        dxdt = np.zeros(len(dx))
        dxdt[10:] = - (dx[10:] - dx[:-10])/(t[10:]-t[:-10])
        dxdt[:10] = dxdt[10]
        
        # allow for division by dxdt
        dxdt[np.abs(dxdt) < 1e-6] = 1e-6
        
        # dtc is supposed to be positve if the gap if overtaking is still to happen
        dtc = dx / dxdt 
        # avoid error where dtc becomes negative if target is faster than ego
        inf = max(2 * np.abs(dtc).max(), 10000)
        dtc[(dx > 0) & (dxdt < 0)] = inf
        # avoid error where dtc becomes positive even after tar is overtaken if target is faster than ego 
        dtc[(dx < 0) & (dxdt < 0)] = - inf
        
        tchat = dtc + t
        
        a_brake = 4
        x_brake = 0.5 * np.maximum(dxdt, 0) ** 2 / a_brake
        
        
        v_1_x = np.array(path.V_v_1_x)
        v_1_y = np.array(path.V_v_1_y)
        dx_1 = tar_x - v_1_x + offset #plus offset, so gap opens for target 
        
        if np.isnan(dx_1).all():
            dx_1[:] = -500
            v_1_y[:] = 1.5
        else:    
            v_1_y = np.interp(t,t[np.invert(np.isnan(dx_1))],v_1_y[np.invert(np.isnan(dx_1))])
            dx_1m = interpolate.interp1d(t[np.invert(np.isnan(dx_1))], 
                                     dx_1[np.invert(np.isnan(dx_1))], 
                                     fill_value='extrapolate')(t)
            dx_1 = np.minimum(dx_1m, np.nanmin(dx_1))
        
        lane_width = 4 # higher than actual value, gives some tolerance
        out_of_position = ((ego_y < 0) | (ego_y > lane_width) | 
                           (((v_1_y < 0) | (v_1_y > lane_width)) & (dx_1 > - 10)) |  # v_1 only has to be in correct lane if close enough to tar
                           (dx_1 > 0) |
                           (tar_y < - lane_width) | (tar_y > lane_width))
        in_position = np.invert(out_of_position)
        
        undecided = in_position & (dx > 0) & (tar_y < 0)
        changed = in_position & (dx > 0) & (tar_y > 0)
        overtaken = in_position & (dx < 0) & (tar_y < 0) & (dxdt >= 0)
        critical = (x_brake > dx) & undecided
        
        # set interpolation parameters, which are negative during undecided and positive afterwards
        undecided_2_changed = tar_y
        undecided_2_overtaken = -dx
        not_critical_2_critical = x_brake - dx
        
        return [undecided, changed, overtaken, critical, 
                undecided_2_changed, undecided_2_overtaken, not_critical_2_critical, tchat]  
     
    def path_to_binary_and_time_sample(self, output_path, output_t, domain):
        # Assume that t_A is when vehicle crosses into street,
        # One could also assume reaction as defined by Zgonnikov et al. (tar_ax > 0)
        # but that might be stupid
        tar_y = output_path.V_tar_y
        t = output_t
        
        
        tar_yh = tar_y[np.invert(np.isnan(tar_y))]
        th = t[np.invert(np.isnan(tar_y))]
        try:
            #interpolate here
            ind_2 = np.where(tar_yh > 0)[0][0]
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
        keep = np.ones(len(self.Input_path), bool)
        for i_sample in range(len(self.Input_path)):
            path = self.Input_path.iloc[i_sample]
            if path.V_ego_y[-1] < 0.5:
                keep[i_sample] = False
            else:
                T = self.Input_T[i_sample]
                # check vehicle v_1 (in front of ego)
                v_1_x = path.V_v_1_x
                v_1_rewrite = np.isnan(v_1_x)
                if v_1_rewrite.any():
                    v_1_y = path.V_v_1_y
                    if len(v_1_rewrite) - v_1_rewrite.sum() < 2:
                        v_1_y = np.ones_like(v_1_y) * 2 
                        v_1_x = path.V_ego_x + 500
                    else:
                        v_1_y = np.interp(T,T[np.invert(v_1_rewrite)],v_1_y[np.invert(v_1_rewrite)])
                        v_1_x = interpolate.interp1d(T[np.invert(v_1_rewrite)], 
                                                     v_1_x[np.invert(v_1_rewrite)], 
                                                     fill_value='extrapolate')(T)
                        
                        
                    path.V_v_1_x = v_1_x
                    path.V_v_1_y = v_1_y
                
                # check vehicle v_2 (behind tar)
                v_2_x = path.V_v_2_x
                v_2_rewrite = np.isnan(v_2_x)
                if v_2_rewrite.any():
                    v_2_y = path.V_v_2_y
                    if len(v_2_rewrite) - v_2_rewrite.sum() < 2:
                        v_2_y = np.ones_like(v_2_y) * - 1.5 
                        v_2_x = path.V_tar_x - 500
                    else:
                        v_2_y = np.interp(T,T[np.invert(v_2_rewrite)],v_2_y[np.invert(v_2_rewrite)])
                        v_2_x = interpolate.interp1d(T[np.invert(v_2_rewrite)], 
                                                     v_2_x[np.invert(v_2_rewrite)], 
                                                     fill_value='extrapolate')(T)
                    path.V_v_2_x = v_2_x
                    path.V_v_2_y = v_2_y
                # check vehicle v_3 (in front of tar)
                v_3_x = path.V_v_3_x
                v_3_rewrite = np.isnan(v_3_x)
                if v_3_rewrite.any():
                    v_3_y = path.V_v_3_y
                    if len(v_3_rewrite) - v_3_rewrite.sum() < 2:
                        v_3_y = np.ones_like(v_3_y) * - 1.5
                        v_3_x = path.V_tar_x + 500
                    else:
                        v_3_y = np.interp(T,T[np.invert(v_3_rewrite)],v_3_y[np.invert(v_3_rewrite)])
                        v_3_x = interpolate.interp1d(T[np.invert(v_3_rewrite)], 
                                                     v_3_x[np.invert(v_3_rewrite)], 
                                                     fill_value='extrapolate')(T)
                    path.V_v_3_x = v_3_x
                    path.V_v_3_y = v_3_y
                    
                    
                    
                # get x-position offset
                x_offset = path.V_tar_x[-1]
                
                path.V_ego_x = path.V_ego_x - x_offset
                path.V_tar_x = path.V_tar_x - x_offset
                path.V_v_1_x = path.V_v_1_x - x_offset
                path.V_v_2_x = path.V_v_2_x - x_offset
                path.V_v_3_x = path.V_v_3_x - x_offset
                
                self.Input_path.iloc[i_sample] = path
                
                path_out = self.Output_path.iloc[i_sample]
                path_out.V_tar_x = path_out.V_tar_x - x_offset
                
                self.Output_path.iloc[i_sample] = path_out
            
        self.Input_path = self.Input_path.iloc[keep]
        self.Output_path = self.Output_path.iloc[keep]
        self.Input_T = self.Input_T[keep]
        self.Output_T = self.Output_T[keep] 
        self.Output_A = self.Output_A[keep] 
        self.Output_T_E = self.Output_T_E[keep] 
        self.Domain = self.Domain.iloc[keep] 
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_solid.append(np.array([[-300, -8],[300, -8]]))
        lines_solid.append(np.array([[-300, 8],[300, 8]]))
        
        lines_dashed = []
        lines_dashed.append(np.array([[-300, 4],[300, 4]]))
        lines_dashed.append(np.array([[-300, 0],[300, 0]]))
        lines_dashed.append(np.array([[-300, -4],[300, -4]]))
        
        
        return lines_solid, lines_dashed
