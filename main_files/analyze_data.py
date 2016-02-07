from __future__ import division
import pandas as pd
import numpy as np
import scipy.stats
import weighted_KDE as wKDE
import get_data as GD
from geopy.distance import vincenty
import datetime
import time

## Let's build the KDE for the crime incidents in the nearby vicinity of a destination

def analyze_crime_kde(crime_coords,weighted,crime_unix_times):

    # Let's mark the coordinates for the edge of the map first
    lon_lat_box = (sorted(crime_coords, key=lambda tup: tup[1])[0][1], sorted(crime_coords, key=lambda tup: tup[1])[-1][1], sorted(crime_coords, key=lambda tup: tup[0])[0][0], sorted(crime_coords, key=lambda tup: tup[0])[-1][0])
    clipsize = [[sorted(crime_coords, key=lambda tup: tup[1])[0][1], sorted(crime_coords, key=lambda tup: tup[1])[-1][1]],[sorted(crime_coords, key=lambda tup: tup[0])[0][0], sorted(crime_coords, key=lambda tup: tup[0])[-1][0]]]

    # Evaluating the KDE
    m_x = [i[1] for i in crime_coords]
    m_y = [i[0] for i in crime_coords]
    train_data = np.vstack([m_x, m_y])
    if weighted == True:
        weights_time = calculate_recency_weights(crime_unix_times)
        #kernel = wKDE.gaussian_kde(train_data, weights=weights_time, bw_method='scott') # from custom online
        kernel = wKDE.gaussian_kde(train_data, weights=weights_time, bw_method=0.2) # from custom online
    else:
        kernel = scipy.stats.gaussian_kde(train_data, bw_method='scott') # from scipy.stats

    # Make the Grid
    minx = min(clipsize[0])
    maxx = max(clipsize[0])
    miny = min(clipsize[1])
    maxy = max(clipsize[1])
    nx = 150 # number of xgrid points
    ny = 150 # number of ygrid points
    x_grid = np.arange(minx,maxx, abs(minx-maxx)/nx) # x-axis
    y_grid = np.arange(miny,maxy, abs(miny-maxy)/ny) # y-axis
    x,y = np.meshgrid(x_grid,y_grid) # create each node of the grid for evaluation
    grid_coords = np.vstack([x.ravel(), y.ravel()])

    # Apply the Model to the Grid
    kde = kernel(grid_coords) # evaluate kernel at each point
    z = np.reshape(kde.T, x.shape)
    
    return (kde,kernel)

def calculate_recency_weights(crime_unix_times):
    ## Let's add some weights based on recency of the event
    now_time = datetime.datetime.now()
    now_time_unix = time.mktime(now_time.timetuple())

    # elapsed time for 10 years ago from the time of the input (establish a baseline)
    delta_time_10 = datetime.timedelta(days=10*365) # 10 years ago delta
    time_10 = now_time - delta_time_10
    time_10_unix = time.mktime(time_10.timetuple()) # unix representation of the date 10 years ago
    elapsed_time_10 = (now_time_unix - time_10_unix)/1e8 # scaled version of unix elapsed time 10 years

    # parameters for the exponential-based weights
    A = 10 # a weight of 10 is for the most recent event happening at the same time as the expected arrival time
    alpha = -np.log(0.1)/elapsed_time_10

    # what are the elapsed times and weights (based on recency)? 
    elapsed_time = []
    weights_time = []
    for row_time in crime_unix_times:
        elapsed_time.append((now_time_unix-row_time)/1e8)
        weights_time.append((A*np.exp(-alpha*elapsed_time[-1])))
    
    return weights_time

def apply_kde_to_racks(kde,kernel,rack_coords):
    p_x = [i[1] for i in rack_coords]
    p_y = [i[0] for i in rack_coords]
    park_data = np.vstack([p_x, p_y])
    
    # normalize kde values
    kde_norm = (99*((kde-min(kde))/(max(kde)-min(kde))))+1 # rescales the KDE values to 1 to 100
    kde_norm_log = 1+99*(np.log10(kde_norm)/2) # rescales the log of the KDE values to 1 to 100
    
    kde_parking = kernel(park_data)
    
    #total_PDF = sum(sum(z))
    #mean_PDF = np.mean(np.mean(z))
    
    #risk_scores = 100*((kde_parking-mean_PDF)/mean_PDF)
    #risk_scores = (99*((risk_scores-min(risk_scores))/(max(risk_scores)-min(risk_scores))))+1
    risk_scores_norm = (99*((kde_parking-min(kde))/(max(kde)-min(kde))))+1 # rescales parking values to KDE scale of 1 to 100
    risk_scores_norm_log = 1+99*(np.log10(risk_scores_norm)/2) # rescales parking values to same scale as log of KDE
    risk_levels = []
    for score in risk_scores_norm_log:
#        if (score >= np.percentile(kde_norm,90)):
#            risk_levels.append(5)
#        elif (score >= np.percentile(kde_norm,70)) and (score < np.percentile(kde_norm,90)):
#            risk_levels.append(4)
#        elif (score >= np.percentile(kde_norm,50)) and (score < np.percentile(kde_norm,70)):
#            risk_levels.append(3)
#        elif (score >= np.percentile(kde_norm,30)) and (score < np.percentile(kde_norm,50)):
#            risk_levels.append(2)
#        elif (score >= np.percentile(kde_norm,0)) and (score < np.percentile(kde_norm,30)):
#            risk_levels.append(1)
        if (score >= 80):
            risk_levels.append(5)
        elif (score >= 60) and (score < 80):
            risk_levels.append(4)
        elif (score >= 40) and (score < 60):
            risk_levels.append(3)
        elif (score >= 20) and (score < 40):
            risk_levels.append(2)
        elif (score >= 0) and (score < 20):
            risk_levels.append(1)
    
    return (risk_scores_norm_log,risk_levels,kde_parking)

def recommend_parking(risk_scores,risk_levels,rack_coords,dist_from_dest,preference):
    # closest parking space
    closest_idx = dist_from_dest.index(min(dist_from_dest),0)
    closest_score = risk_scores[closest_idx]
    closest_coords = rack_coords[closest_idx]
    
    candidate_scores = risk_scores[np.where(risk_scores <= closest_score)]
    candidate_dists = np.array(dist_from_dest)[np.where(risk_scores <= closest_score)]
    candidate_coords = np.array(rack_coords)[np.where(risk_scores <= closest_score)]
        
    # Scale the features
    candidate_dists = np.asarray(candidate_dists)
    candidate_scores = np.asarray(candidate_scores)
    print(len(candidate_dists))
    if len(candidate_dists) > 1:
        candidate_dists_norm = 1-((candidate_dists-min(candidate_dists))/(max(candidate_dists)-min(candidate_dists)))
        candidate_scores_norm = 1-((candidate_scores-min(candidate_scores))/(max(candidate_scores)-min(candidate_scores)))
    else:
        candidate_dists_norm = np.asarray([1])
        candidate_scores_norm = np.asarray([1])
    
    # Do the weighted average
    A = int(preference) # (1 = only care about how close the rack is, 0 = only care about how safe the rack is)
    B = 100-A
    decision_score = A*candidate_dists_norm + B*candidate_scores_norm
    chosen_idx = np.where(decision_score==max(decision_score))[0][0]
    
    # What to extract from this chosen space and output
    chosen_score = candidate_scores[chosen_idx]
    chosen_change_in_distance = candidate_dists[chosen_idx] - min(dist_from_dest)
    chosen_coords = candidate_coords[chosen_idx]
    
    return (chosen_coords,chosen_score,chosen_change_in_distance,closest_coords,closest_score)
                       
 