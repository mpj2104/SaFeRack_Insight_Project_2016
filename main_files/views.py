from __future__ import division
from flask import render_template, request
from main_files import app
import get_data as GD
import analyze_data as AD
import pandas as pd
from a_Model import ModelIt
from map_displayer import display
from geopy.geocoders import GoogleV3
import mplleaflet
import folium
import numpy as np
from IPython.display import HTML
from geopy.distance import vincenty
import os
import json
#from googleplaces import GooglePlaces, types, lang

@app.route('/')

@app.route('/input')
def destination_input():
    return render_template("input.html")

@app.route('/input_alt')
def destination_input_alt():
    return render_template("input_alt.html")

@app.route('/presentation')
def destination_presentation():
    return render_template("presentation.html")

@app.route('/output')
def destination_output():
    # pull 'destination' from input field and store it
    destination = request.args.get('destination')
    if not destination:
        return render_template("input.html")
    radius = request.args.get('radius')
    preference = request.args.get('preference_level')
    arrival_time_hour = int(request.args.get('arrival_time_hour'))
    arrival_time_minute = float(request.args.get('arrival_time_minute'))
    am_pm = int(request.args.get('am_pm'))
    
    # make the map object
    google_api = 'AIzaSyD1URYVEGEVD5hX0NjDU1M97wAZmwOQg6o'
    #google_places = GooglePlaces(google_api)
    #query_result = google_places.text_search(destination)
    #place = query_result.places[0]
    #place.get_details()
    geolocator = GoogleV3(api_key=google_api)
    #address, (latitude, longitude) = geolocator.geocode(place.formatted_address)
    address, (latitude, longitude) = geolocator.geocode(destination)
    starting_coordinates= [latitude, longitude]
    zoom_start = 15
    
    # convert the time
    if am_pm == 0:
        if arrival_time_hour == 12:
            arrival_time_hour = 0
        arrival_time = arrival_time_hour + arrival_time_minute
    elif am_pm == 1:
        if arrival_time_hour == 12:
            arrival_time = arrival_time_hour + arrival_time_minute
        else:
            arrival_time = arrival_time_hour + 12 + arrival_time_minute
    
    # get crime coordinates
    [crime_coords, crime_dates, crime_times, dest_to_crimes, crime_unix_times, crime_coords_2012to2015, crime_coords_2008to2011, crime_coords_2003to2007,crime_descripts_2012to2015,crime_descripts_2008to2011,crime_descripts_2003to2007,crime_counts_2015,SF_wide_crime_counts_2015] = GD.get_crime_coords(starting_coordinates, arrival_time)
    
    if not crime_coords:
        return render_template("input_alt.html")
    else:
        crime_output = []
        for idx,coord in enumerate(crime_coords):
            datum = {
                    "lat": coord[0],
                    "lon": coord[1],
                    "crime_dates": str(crime_dates[idx]),
                    "crime_times": str(crime_times[idx]),
                    "dist_from_dest": str(dest_to_crimes[idx]),
                    }
            crime_output.append(datum)

        crime_2012to2015 = []
        for idx,coord in enumerate(crime_coords_2012to2015):
            datum = {
                    "lat": coord[0],
                    "lon": coord[1],
                    "crime_descripts": str(crime_descripts_2012to2015[idx])
                    }
            crime_2012to2015.append(datum)

        crime_2008to2011 = []
        for idx,coord in enumerate(crime_coords_2008to2011):
            datum = {
                    "lat": coord[0],
                    "lon": coord[1],
                    "crime_descripts": str(crime_descripts_2008to2011[idx])
                    }
            crime_2008to2011.append(datum)

        crime_2003to2007 = []
        for idx,coord in enumerate(crime_coords_2003to2007):
            datum = {
                    "lat": coord[0],
                    "lon": coord[1],
                    "crime_descripts": str(crime_descripts_2003to2007[idx])
                    }
            crime_2003to2007.append(datum)

        # analyze KDE (weighted --> (crime_coords,True,crime_unix_times), unweighted --> (crime_coords,False,0))
        [kde,kernel] = AD.analyze_crime_kde(crime_coords,True,crime_unix_times) 
        #[z,kernel] = AD.analyze_crime_kde(crime_coords,False,0)

        # get parking coordinates
        [rack_coords,num_racks,num_spaces,dist_from_dest,parking_address,space_count_in_crime_area] = GD.get_parking_coords(starting_coordinates,float(radius))
        [risk_scores,risk_levels,_] = AD.apply_kde_to_racks(kde,kernel,rack_coords)
        rack_output = []
        for idx,coord in enumerate(rack_coords):
            datum = {
                    "lat": coord[0],
                    "lon": coord[1],
                    "num_racks": str(int(num_racks[idx])),
                    "num_spaces": str(int(num_spaces[idx])),
                    "parking_address": str(parking_address[idx]),
                    "dist_from_dest": str(dist_from_dest[idx]),
                    "risk_score": str(int(risk_scores[idx])),
                    "risk_level": str(int(risk_levels[idx]))
                    }
            rack_output.append(datum)

        # get recommended coordinates
        [chosen_coords,chosen_score,chosen_change_in_distance,closest_coords,closest_score] = AD.recommend_parking(risk_scores,risk_levels,rack_coords,dist_from_dest,preference)
        chosen_is_closest = 0
        if (chosen_coords[0]==closest_coords[0]) and (chosen_coords[1]==closest_coords[1]):
            chosen_is_closest = 1
        recommended_output = []
        datum = {
                "lat": chosen_coords[0],
                "lon": chosen_coords[1],
                "chosen_score": str(int(chosen_score)),
                "closest_score": str(int(closest_score)),
                "added_dist": str(chosen_change_in_distance),
                "closest_coords_lat": closest_coords[0],
                "closest_coords_lon": closest_coords[1],
                "chosen_is_closest": chosen_is_closest
                }
        recommended_output.append(datum)

        # ratio for local area
        crime_per_space_local = round(crime_counts_2015/space_count_in_crime_area,2)
        
        # ratio for SF
        crime_per_space_SF = round(SF_wide_crime_counts_2015/6755,2)

        return render_template("output.html", starting_coordinates = starting_coordinates, zoom_start = zoom_start, title = 'SaFeRack', address = address, rack_coords = rack_output, crime_coords = crime_output, rack_counts = len(rack_output), crime_counts = len(crime_output), recommended_coords = recommended_output, crime_coords_2012to2015 = crime_2012to2015, counts_2012to2015 = len(crime_2012to2015), crime_coords_2008to2011 = crime_2008to2011, counts_2008to2011 = len(crime_2008to2011), crime_coords_2003to2007 = crime_2003to2007, counts_2003to2007 = len(crime_2003to2007), crime_per_space_local = str(crime_per_space_local),crime_per_space_SF = str(crime_per_space_SF))
