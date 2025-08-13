# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:21:00 2024
@author: xinyuan.wei
"""
#########################################################################
### PyFIA Carbon Climate Data ###
#########################################################################
import os
import rasterio
import pandas as pd
from pyproj import Transformer

###----------------------------------------------------------------------------
# Extract climate data from a TIF file at a specific lat/lon
def extract_var (tif_file, lat, lon):
    # Initialize a transformer to ensure the projection matches the TIF files
    transformer = Transformer.from_crs('epsg:4326', 'epsg:4326')
    try:
        with rasterio.open(tif_file) as dataset:
            # Transform coordinates if needed
            lat, lon = transformer.transform(lat, lon)
            # Get the row and column of the pixel corresponding to the lat/lon
            row, col = dataset.index(lon, lat)
            # Read the value of the pixel
            value = dataset.read(1)[row, col]
        return value
    
    except Exception as e:
        print(f'Error processing {tif_file} at lat: {lat}, lon: {lon}')
        print(e)
        return None

# Acquire climate information for each plot
def climate_data(data, climate_data, savefile):
    # Read the TIF climate data
    climate_dir = climate_data

    # List to store climate variable names
    climate_vars = []
    
    # Read the plot data
    data = pd.read_csv(data)

    # Process each TIF file in the climate directory
    for tif_file in os.listdir(climate_dir):
        if tif_file.endswith('.tif'):
            climate_var = tif_file.replace('.tif', '')
            climate_vars.append(climate_var)
            climate_file = os.path.join(climate_dir, tif_file)
            print(tif_file)
        
            # Extract climate data for each plot
            data[climate_var] = data.apply(lambda row: extract_var(climate_file, 
                                                                   row['LAT'], 
                                                                   row['LON']), axis=1)

    # Save the combined data to a new CSV file
    data.to_csv(savefile, index=False)