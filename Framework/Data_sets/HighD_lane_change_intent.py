import numpy as np
import pandas as pd
from HighD_lane_change import HighD_lane_change
from scipy import interpolate

class HighD_lane_change_intent(HighD_lane_change):
         
    def position_states_in_path(self, path, t, domain):
        ego_x = np.array(path.V_ego_x)
        ego_y = np.array(path.V_ego_y)
        tar_x = np.array(path.V_tar_x)
        tar_y = np.array(path.V_tar_y)
        v_1_x = np.array(path.V_v_1_x)
        v_1_y = np.array(path.V_v_1_y)
        v_3_x = np.array(path.V_v_3_x)
        v_3_y = np.array(path.V_v_3_y)
        
        # v_1 is the vehicle in front of the ego vehicle
        # v_3 is the vehicle the target vehicle is following 
        
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
            
        
        dx_3 = v_3_x - tar_x - offset #plus offset, so gap opens for target 
        
        if np.isnan(dx_3).all():
            dx_3[:] = 500
            v_3_y[:] = - 1.5
        else:    
            v_3_y = np.interp(t,t[np.invert(np.isnan(dx_3))],v_1_y[np.invert(np.isnan(dx_3))])
            dx_3m = interpolate.interp1d(t[np.invert(np.isnan(dx_3))], 
                                     dx_3[np.invert(np.isnan(dx_3))], 
                                     fill_value='extrapolate')(t)
            dx_3 = np.minimum(dx_3m, np.nanmin(dx_3))
        
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
        
        
        # Test if decision is there
        try:
            ind_decided = np.where(undecided[:-1] & (changed[1:] | overtaken[1:]))[0][0] + 1
            possible_ind_s = np.where(undecided[:ind_decided])[0]
            try:
                ind_s = possible_ind_s[np.where(possible_ind_s[:-1] + 1 != possible_ind_s[1:])[0][-1] + 1]
            except:
                ind_s = possible_ind_s[0]
        
        
            
            # if decision is there, check for rejected gaps
            if overtaken[ind_decided]:
                intent = False
                
                # check for later lane change to indicate intent
                changed_later = (dx < 0) & (tar_y > 0) & (np.arange(len(dx)) > ind_s) 
                if any(changed_later):
                    intent = True
                
                else:
                    # check if there was a need for braking by target vehicle
                    dtcs = dtc[ind_s]
                    
                    dx_3dt = np.zeros(len(dx_3))
                    dx_3dt[10:] = - (dx_3[10:] - dx_3[:-10])/(t[10:]-t[:-10])
                    dx_3dt[:10] = dx_3dt[10]
                    
                    # allow for division by dxdt
                    dx_3dt[np.abs(dx_3dt) < 1e-6] = 1e-6
                    
                    # dtc is supposed to be positve if the gap if overtaking is still to happen
                    dtc_3 = dx_3 / dx_3dt 
                    # avoid error where dtc becomes negative if target is faster than ego
                    dtc_3[(dx_3 > 0) & (dx_3dt < 0)] = inf
                    # avoid error where dtc becomes positive even after tar is overtaken if target is faster than ego 
                    dtc_3[(dx_3 < 0) & (dx_3dt < 0)] = - inf
                    
                    dtc_3s = dtc_3[ind_s]
                    assert dtc_3s > 0, "Something went horribly wrong"
                    
                    if 2.0 * dtc_3s < dtcs:
                        intent = True
        
                if not intent:
                    # to sabotage, set undecided to all false.
                    undecided[:] = False
            elif changed[ind_decided]:
                pass
            else:
                raise TypeError("neither accepted or rejected despite prior filtering")
        
            return [undecided, changed, overtaken, critical, 
                    undecided_2_changed, undecided_2_overtaken, not_critical_2_critical, tchat]
        
        except:
            # there never was as starting point here in the first place
            
            return [undecided, changed, overtaken, critical, 
                    undecided_2_changed, undecided_2_overtaken, not_critical_2_critical, tchat]
        
          
    
    
