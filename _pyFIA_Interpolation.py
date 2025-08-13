# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:17:48 2024

@author: xinyuan.wei
"""
#########################################################################
### PyFIA Interplation State Map ###
#########################################################################
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata

###---------------------------------------------------------------------------- 
# Spatial map inrerpolation function
def map_interpolate(data, var, basemap, FIPS, resolution, interp_method, land_cover, savefile):
    # Read the plot data
    plot_data = pd.read_csv(data)

    # Filter the study area using FIPS code and validate geometries
    study_area = basemap[basemap['STATEFP'] == FIPS].copy()
    study_area = study_area[study_area.is_valid]

    if study_area.empty:
        raise ValueError(f'No geometries found for FIPS code {FIPS} in the basemap.')

    # Convert plot data to a GeoDataFrame
    plot_data['geometry'] = [Point(xy) for xy in zip(plot_data['LON'], plot_data['LAT'])]
    plot_data = gpd.GeoDataFrame(plot_data, geometry='geometry')

    # Set the CRS for plot_biomass if it doesn't have one
    if plot_data.crs is None:
        plot_data.set_crs(epsg=4326, inplace=True)

    # Reproject plot data to match the CRS of the land cover map
    with rasterio.open(land_cover) as lc:
        lc_crs = lc.crs
        lc_transform = lc.transform
        lc_shape = lc.shape

    if plot_data.crs != lc_crs:
        plot_data = plot_data.to_crs(lc_crs)

    # Interpolate plot data using griddata
    points = np.array([(geom.x, geom.y) for geom in plot_data.geometry])
    values = plot_data[var].values

    # Generate target grid coordinates matching the land cover map
    dst_x, dst_y = np.meshgrid(
        np.arange(lc_transform[2], lc_transform[2] + lc_shape[1] * lc_transform[0], lc_transform[0]),
        np.arange(lc_transform[5], lc_transform[5] + lc_shape[0] * lc_transform[4], lc_transform[4])
    )

    # Resample plot data to the land cover grid
    plot_resampled = griddata(
        points, values, (dst_x, dst_y), method=interp_method
    )

    # Apply the forest mask
    with rasterio.open(land_cover) as lc:
        land_cover = lc.read(1)
        forest_mask = np.isin(land_cover, [41, 42, 43])

    forest_map = np.full(land_cover.shape, np.nan, dtype=np.float32)
    forest_map[forest_mask] = plot_resampled[forest_mask]

    # Save the map
    with rasterio.open(
        savefile, 'w', driver='GTiff', height=forest_map.shape[0],
        width=forest_map.shape[1], count=1, dtype='float32', crs=lc_crs,
        transform=lc_transform) as dst:
        dst.write(forest_map, 1)

    print(f'Forest {var} map saved to {savefile}')

    # Filter the study area using FIPS code and validate geometries
    study_area = basemap[basemap['STATEFP'] == FIPS].copy()
    study_area = study_area[study_area.is_valid]

    # Open the raster and reproject it to EPSG:4326
    with rasterio.open(savefile) as src:
        target_crs = 'EPSG:4326'
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        reprojected_data = np.zeros((height, width), dtype=np.float32)
        rasterio.warp.reproject(
            source=src.read(1),
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=rasterio.warp.Resampling.nearest
        )
        
        # Mask no-data values
        reprojected_data = np.ma.masked_where(reprojected_data == src.nodata, reprojected_data)
        
    # Reproject the shapefile to EPSG:4326
    if study_area.crs != target_crs:
        study_area = study_area.to_crs(target_crs)

    # Get the extent of the reprojected raster
    lon_min, lat_min, lon_max, lat_max = (
        transform[2],
        transform[5] + transform[4] * height,
        transform[2] + transform[0] * width,
        transform[5]
    )

    # Plot the raster with transparency for no-data values
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a colormap with transparency for masked (no-data) values
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')

    im = ax.imshow(reprojected_data, extent=(lon_min, lon_max, lat_min, lat_max),
                   origin='upper', cmap=cmap, interpolation=interp_method)
    plt.colorbar(im, label=var)

    # Overlay the boundary
    study_area.boundary.plot(ax=ax, color='black', linewidth=0.8)

    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Interpolated {var} Map')
    plt.show()