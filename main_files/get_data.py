from __future__ import division
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from geopy.distance import vincenty

database_name = 'SF_bike_parking.db'
engine = create_engine('sqlite:///:' + database_name + ':')

def get_parking_coords(starting_coordinates,radius):
    # Get the parking spaces data
    table_name = 'SF_bike_parking'
    sql_query = 'SELECT Y,X,racks,spaces,yr_inst FROM ' + table_name + ' WHERE yr_installed < 2016'
    df_parking = pd.read_sql_query(sql_query,engine)

    # Calculate distances to list of bike racks and filter by racks within a radius
    rack_coords = []
    num_racks = []
    num_spaces = []
    parking_address = []
    dist_from_dest = []
    radius_racks = radius # in miles
    space_count_in_crime_area = 0
    for index, row in df_parking.ix[0:].iterrows():
        if (vincenty(starting_coordinates, [row['Y'],row['X']]).miles <= radius_racks):
            rack_coords.append([row['Y'],row['X']])
            num_racks.append(row['racks'])
            num_spaces.append(row['spaces'])
            parking_address.append(row['yr_inst'])
            dist_from_dest.append(round(vincenty(starting_coordinates, [row['Y'],row['X']]).miles,2))
            space_count_in_crime_area += row['spaces']
        elif (vincenty(starting_coordinates, [row['Y'],row['X']]).miles > radius_racks) and (vincenty(starting_coordinates, [row['Y'],row['X']]).miles <= 0.75):
            space_count_in_crime_area += row['spaces']
            continue
        else:
            continue

    return (rack_coords, num_racks, num_spaces, dist_from_dest, parking_address, space_count_in_crime_area)

def get_crime_coords(starting_coordinates,arrival_time):
    # Get the bike theft data
    table_name = 'SF_aggregate_stolen_bikes'
    
    if (arrival_time >= 0) and (arrival_time < 12):
        sql_query = 'SELECT Y,X,Date,Descript,Time,merged_time,Year FROM ' + table_name + ' WHERE (Year < 2016) AND (cum_minutes_day > ' + str(arrival_time) + '*60 AND cum_minutes_day <= ' + str(arrival_time) + '*60+12*60) AND (PdDistrict <> \'None\')'
    elif (arrival_time >= 12) and (arrival_time < 24):
        sql_query = 'SELECT Y,X,Date,Descript,Time,merged_time,Year FROM ' + table_name + ' WHERE (Year < 2016) AND (cum_minutes_day > ' + str(arrival_time) + '*60 OR cum_minutes_day <= ' + str(arrival_time) + '*60-12*60) AND (PdDistrict <> \'None\')'
    
    df_thefts = pd.read_sql_query(sql_query,engine)
    
    # Calculate distances to list of crimes and filter by crimes within radius
    crime_coords = []
    crime_coords_2012to2015 = []
    crime_coords_2008to2011 = []
    crime_coords_2003to2007 = []
    crime_dates = []
    crime_descripts_2012to2015 = []
    crime_descripts_2008to2011 = []
    crime_descripts_2003to2007 = []
    crime_times = []
    dest_to_crimes = []
    crime_unix_times = []
    crime_counts_2015 = 0
    SF_wide_crime_counts_2015 = 0
    radius_crimes = 0.75 # in miles
    for index, row in df_thefts.ix[0:].iterrows():
        if (vincenty(starting_coordinates, [row['Y'],row['X']]).miles <= radius_crimes):
            crime_coords.append([row['Y'],row['X']])
            crime_dates.append(row['Date'])
            crime_times.append(row['Time'])
            crime_unix_times.append(row['merged_time'])
            dest_to_crimes.append(round(vincenty(starting_coordinates, [row['Y'],row['X']]).miles,2))
            if row['Year'] >= 2012:
                crime_coords_2012to2015.append([row['Y'],row['X']])
                crime_descripts_2012to2015.append(row['Descript'])
                if row['Year'] == 2015:
                    crime_counts_2015 += 1 
                    SF_wide_crime_counts_2015 += 1
            elif (row['Year'] >= 2008) and (row['Year'] < 2012):
                crime_coords_2008to2011.append([row['Y'],row['X']])
                crime_descripts_2008to2011.append(row['Descript'])
            elif row['Year'] <= 2007:
                crime_coords_2003to2007.append([row['Y'],row['X']])
                crime_descripts_2003to2007.append(row['Descript'])
        else:
            if row['Year'] == 2015:
                SF_wide_crime_counts_2015 += 1
            continue

    return (crime_coords, crime_dates, crime_times, dest_to_crimes, crime_unix_times, crime_coords_2012to2015, crime_coords_2008to2011, crime_coords_2003to2007,crime_descripts_2012to2015,crime_descripts_2008to2011,crime_descripts_2003to2007,crime_counts_2015,SF_wide_crime_counts_2015)