import pandas as pd
import numpy as np
from os import path

pd.options.mode.chained_assignment=None
n=1

### Prepare to get situation data, adjust values and position reference

print('Preprocessing Data')
while path.exists('data/{}_recordingMeta.csv'.format(str(n).zfill(2))):
    Meta_data=pd.read_csv('data/{}_recordingMeta.csv'.format(str(n).zfill(2)))
    local_data=Meta_data[['locationId','speedLimit','upperLaneMarkings','lowerLaneMarkings']].copy(deep=True)
    upper_Lane=np.array(local_data['upperLaneMarkings'][0].split(';')).astype('float')    
    
    if len(upper_Lane)>4:
        new_upper_Lane=upper_Lane-upper_Lane[1]
    else:
        new_upper_Lane=upper_Lane-upper_Lane[0]
    
    lower_Lane=np.array(local_data['lowerLaneMarkings'][0].split(';')).astype('float')
    if len(lower_Lane)>4:
        new_lower_Lane=lower_Lane[-2]-lower_Lane
    else:
        new_lower_Lane=lower_Lane[-1]-lower_Lane
        
    
    local_data['adjustedupperLaneMarkings']=local_data['upperLaneMarkings'].copy(deep=True)
    local_data['upperLaneMarkings'][0]=upper_Lane
    local_data['adjustedupperLaneMarkings'][0]=new_upper_Lane  
    
    local_data['adjustedlowerLaneMarkings']=local_data['lowerLaneMarkings'].copy(deep=True)
    local_data['lowerLaneMarkings'][0]=lower_Lane
    local_data['adjustedlowerLaneMarkings'][0]=np.flip(new_lower_Lane)
    
    if n==1:
        Scenario=local_data
    else:
        Scenario=Scenario.append(local_data,ignore_index=True)
    n=n+1



    

id_addition=0
Final_out=[]
for n in range(1,1+len(Scenario)):
    print('Processing Scenario {}/{}'.format(str(n).zfill(2),len(Scenario)))
    Meta_data=pd.read_csv('data/{}_recordingMeta.csv'.format(str(n).zfill(2)))
    Track_data=pd.read_csv('data/{}_tracks.csv'.format(str(n).zfill(2)))
    Track_data=Track_data[['frame', 'id', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', \
                            'precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId',\
                            'rightPrecedingId', 'rightAlongsideId','rightFollowingId', 'laneId']]
    Track_Meta_data=pd.read_csv('data/{}_tracksMeta.csv'.format(str(n).zfill(2)))
    Final=Track_Meta_data[['id','width','height','class','numLaneChanges']].copy(deep=True)
    
    Final['numMerges']='0'
    
    Final['locationId']=Meta_data['locationId'][0]
    
    Final['recordingId']=Meta_data['id'][0]
    
    max_x=np.max(Track_data['x'])
    num_lane_markers=len(Scenario['upperLaneMarkings'][n-1])
    upper_Lane_Markers=Scenario['upperLaneMarkings'][n-1]
    adjusted_upper_Lane_Markers=Scenario['adjustedupperLaneMarkings'][n-1]
    lower_Lane_Markers=Scenario['lowerLaneMarkings'][n-1]
    adjusted_lower_Lane_Markers=Scenario['adjustedlowerLaneMarkings'][n-1]
    
    
    Final['laneMarkings']='0'
    Final['track']='0'
    for loc_id in range(len(Final)):
        
        local_track=Track_data[Track_data['id'] == loc_id+1].copy(deep=True)
        
        # Remove different notation for different travel directions
        # x: in direction of travel
        # y: in the left from the direction of travel, 0 is equal to the right lane marking (excluding merging lanes)
        if Track_Meta_data['drivingDirection'][loc_id]==1: #driving to the left in the upper part
            local_track['x']=max_x-local_track['x']
            local_track['xVelocity']=-local_track['xVelocity']
            local_track['xAcceleration']=-local_track['xAcceleration']
            
            # Adjust y to be in the middle in the front center of the vehicle
            local_track['y']=local_track['y']+0.5*Final['height'][loc_id]\
                -(upper_Lane_Markers[0]-adjusted_upper_Lane_Markers[0])
                
            if num_lane_markers>4:
                local_track['laneId']=local_track['laneId']-2
            else:
                local_track['laneId']=local_track['laneId']-1
                
            Final['laneMarkings'][loc_id]=adjusted_upper_Lane_Markers
        else:
            # Adjust y to be in the middle in the front center of the vehicle
            local_track['y']=(lower_Lane_Markers[0]+adjusted_lower_Lane_Markers[-1])\
                -(local_track['y']+0.5*Final['height'][loc_id])
                
            local_track['yVelocity']=-local_track['yVelocity']
            local_track['yAcceleration']=-local_track['yAcceleration']
            if num_lane_markers>4:
                local_track['laneId']=2*num_lane_markers-local_track['laneId']
            else:
                local_track['laneId']=2*num_lane_markers+1-local_track['laneId']
            
            Final['laneMarkings'][loc_id]=adjusted_lower_Lane_Markers
        
        local_track.drop(columns=['id'], inplace=True)
        local_track[['precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId',\
                        'rightPrecedingId', 'rightAlongsideId','rightFollowingId']] \
                =local_track[['precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId',\
                        'rightPrecedingId', 'rightAlongsideId','rightFollowingId']]+id_addition
        #set non existent vehicles to zero again.
        local_track[local_track[['precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId',\
                        'rightPrecedingId', 'rightAlongsideId','rightFollowingId']]==id_addition]=0
        if local_track['laneId'].iloc[0]<1:
            Final['numMerges'][loc_id]=np.int64(1)
            Final['numLaneChanges'][loc_id]=Final['numLaneChanges'][loc_id]-1
        else:
            Final['numMerges'][loc_id]=np.int64(0)
        Final['track'][loc_id]=local_track
        Final['id'][loc_id]=Final['id'][loc_id]+id_addition
    
    
    
    id_addition=id_addition+Meta_data['numVehicles'][0] 
    Final_out.append(Final.copy(deep=True))
   
print('Save Data')
Final_out= pd.concat(Final_out)
Final_out.to_pickle("highD_processed.pkl")