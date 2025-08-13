# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:13:12 2024

@author: xinyuan.wei
"""
#########################################################################
### PyFIA Map Functions ###
#########################################################################
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.colors as mcolors
from shapely.geometry import Point
from matplotlib.ticker import FuncFormatter
from pyproj import Transformer

###----------------------------------------------------------------------------
# Map all the inventory plots for a given state
def plot_map(FIA_Data, basemap, FIPS, savefile):
    
    plot_data = FIA_Data['plot_data']
    
    # Filter the basemap to include only the target state
    state_map = basemap[basemap['STATEFP'] == FIPS]

    # Create a geometry column for the plot data using longitude and latitude
    geometry = [Point(xy) for xy in zip(plot_data['LON'], plot_data['LAT'])]
    plot_gdf = gpd.GeoDataFrame(plot_data, crs='EPSG:4269', geometry=geometry)

    # Reproject both plot data and state map to EPSG:3857 for contextily
    state_map = state_map.to_crs(epsg=3857)
    plot_gdf = plot_gdf.to_crs(epsg=3857)

    # Plot the basemap and plot locations
    fig, ax = plt.subplots(figsize=(12, 8))
    state_map.plot(ax=ax, alpha=0.5, edgecolor='black', facecolor='none')
    plot_gdf.plot(ax=ax, markersize=2, color='black', marker='o', label='FIA Plots')

    # Add a basemap layer (e.g., OpenStreetMap)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Set title and labels
    ax.set_title('Locations of FIA inventory plots')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()

    # Convert x and y axis labels back to latitude and longitude
    def to_latlon(x, pos):
        point = gpd.GeoSeries([Point(x, 0)], crs="EPSG:3857").to_crs("EPSG:4326")
        return f"{point.x[0]:.2f}"

    def to_lat(x, pos):
        point = gpd.GeoSeries([Point(0, x)], crs="EPSG:3857").to_crs("EPSG:4326")
        return f"{point.y[0]:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(to_latlon))
    ax.yaxis.set_major_formatter(FuncFormatter(to_lat))

    # Show the plot
    plt.show()
    
    # Save the figure
    fig.savefig(savefile, dpi=1200, bbox_inches='tight')

###----------------------------------------------------------------------------    
# Map statewide plot-level forest attributes
def plot_forest_map(data, var, basemap, FIPS, savefile):
    
    # Read the data
    plot_data = pd.read_csv(data)
    
    # Create a basemap
    state_map = basemap[basemap['STATEFP'] == FIPS]

    # Create a GeoDataFrame from the data with geometry
    geometry = gpd.points_from_xy(plot_data['LON'], plot_data['LAT'])
    plot_geo = gpd.GeoDataFrame(plot_data, crs='EPSG:4326', geometry=geometry)

    # Save to shapefile
    shapefile_path = savefile.replace('.png', '.shp')
    plot_geo.to_file(shapefile_path)

    # Reproject to Web Mercator
    state_map_web = state_map.to_crs('EPSG:3857')
    plot_geo_web = plot_geo.to_crs('EPSG:3857')

    # Create color map from yellow to blue
    cmap = mcolors.LinearSegmentedColormap.from_list('blue_to_yellow', ['blue', 'yellow'])

    # Plot the basemap and plot data
    fig, ax = plt.subplots(figsize=(12, 8))
    state_map_web.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
    plot_geo_web.plot(column=var, ax=ax, markersize=10, 
                      legend=True, cmap=cmap, alpha=0.8, 
                      scheme='NaturalBreaks', k=7)

    # Add a basemap layer with a specified zoom level
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10)
    
    # Initialize Transformer
    transformer = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)

    # Define custom formatter functions
    def format_lon(x, pos):
        lon, _ = transformer.transform(x, 0)
        return f'{lon:.2f}°'

    def format_lat(y, pos):
        _, lat = transformer.transform(0, y)
        return f'{lat:.2f}°'

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Set axis labels and dynamic title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{var}')

    # Save the plot with tight bounding box
    plt.savefig(savefile, dpi=1200, bbox_inches='tight')

    # Show the plot
    plt.show()

###----------------------------------------------------------------------------
# Map county-level biomass/carbon density
def county_biomass_map(data, var, basemap, FIPS, savefile):
    # Read the data
    county_data = pd.read_csv(data)
    
    # Filter the counties in the shapefile to only include the target state
    counties = basemap[basemap['STATEFP'] == FIPS]

    # Join the county shapefile with the biomass data
    biomass_geo = counties.merge(county_data, left_on='NAME', right_on='COUNTYNM')

    # Plot the biomass density map
    fig, ax = plt.subplots(figsize=(12, 8))
    biomass_geo.plot(
        column=var,
        cmap='cividis',
        linewidth=0.1,
        edgecolor='black',
        legend=True,
        ax=ax,
        legend_kwds={'label': var}
    )

    ax.set_title(f'County-level {var}')
    ax.set_axis_off()
    
    # Save the plot
    plt.savefig(savefile, dpi=1200, bbox_inches='tight')
    plt.show()

