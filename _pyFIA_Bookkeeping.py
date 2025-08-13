# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:03:55 2024

@author: xinyuan.wei
"""
#########################################################################
### PyFIA Bookkeeping Forest Biomass Tracking ###
#########################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import tempfile
from rasterio.warp import reproject
from rasterio.warp import Resampling
from scipy.optimize import curve_fit

###----------------------------------------------------------------------------
# Generate the stand growth model for bookkeeping model
def stand_growth(data, var, inv_yr, n_years, max_age, forest_type):
    # Read the data
    data = pd.read_csv(data)
    
    # Drop rows with missing values in 'STDAGE' or 'PlotAG'
    data.dropna(subset=['STDAGE', var, 'FORTYPCD'], inplace=True)
    
    # Filter the data where forest type is the target type
    data = data[data['FORTYPCD'].isin(forest_type)]
    
    # Filter the data where stand age is less than max_age
    data = data[data['STDAGE'] > 0]
    data = data[data['STDAGE'] < max_age]
    
    # Group by STDAGE and calculate mean and standard deviation
    grouped = data.groupby('STDAGE')[var].agg(['mean', 'std']).reset_index()
    
    # Prepare data for plotting
    stand_age = grouped['STDAGE']
    mean_biomass = grouped['mean']
    std_biomass = grouped['std']
    
    # Define the power function for regression
    def power_func(x, a, b):
        return a * x ** b
    
    # Prepare data for regression
    xdata = stand_age
    ydata = mean_biomass
    
    # Fit the curve using the power function, Initial parameter guesses
    initial_guess = [1, 1]
    
    # Fit the power function to the data
    popt, pcov = curve_fit(power_func, xdata, ydata, p0=initial_guess)
    
    # Extract parameters
    a, b = popt
    perr = np.sqrt(np.diag(pcov))
    print('Regression Parameters:')
    print(f'a: {a:.4f} ± {perr[0]:.4f}')
    print(f'b: {b:.4f} ± {perr[1]:.4f}')
    
    # Compute R-squared
    residuals = ydata - power_func(xdata, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'R-squared: {r_squared:.4f}')
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.errorbar(stand_age, mean_biomass, yerr=std_biomass, fmt='o', 
                 ecolor='lightgray', elinewidth=1, capsize=3, markersize=5, label='Data with STD')
    
    # Plot the regression line
    x_fit = np.linspace(min(stand_age), max(stand_age), 100)
    y_fit = power_func(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = {a:.2f} * x^{b:.2f}')
    
    plt.xlabel('Stand Age (years)', fontsize=20)
    plt.ylabel(var, fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 140)
    plt.grid(True)
    plt.show()

###----------------------------------------------------------------------------
# Bookkeeping model
def reproject_raster(input_raster, reference_raster):
    # Reprojects the raster image to match the reference raster
    with rasterio.open(reference_raster) as ref_src:
        dst_crs = ref_src.crs
        dst_transform = ref_src.transform
        dst_width = ref_src.width
        dst_height = ref_src.height
        dst_profile = ref_src.profile

    with rasterio.open(input_raster) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_profile = src.profile

    # Update the profile for the reprojected raster
    dst_profile.update({
        'crs': dst_crs,
        'transform': dst_transform,
        'width': dst_width,
        'height': dst_height,
        'nodata': src_profile.get('nodata', None),
        'dtype': src_profile['dtype']
    })

    # Create a temporary file to save the reprojected raster
    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    temp_filename = temp_file.name
    temp_file.close()  # Close the file so rasterio can write to it

    with rasterio.open(temp_filename, 'w', **dst_profile) as dst:
        reproject(
            source=src_data,
            destination=rasterio.band(dst, 1),
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
    return temp_filename

def bookkeeping(base_map, next_map, frtp_map, dist_map, yr, par_file):
    # Read the parameter file into a DataFrame
    params_df = pd.read_csv(par_file)
    params_df.set_index('Code', inplace=True)

    # Reproject the forest disturbance and type maps to match the base biomass map
    reprojected_dist_map = reproject_raster(dist_map, base_map)
    reprojected_frtp_map = reproject_raster(frtp_map, base_map)

    # Open the base biomass map, reprojected forest disturbance and forest type maps
    with rasterio.open(base_map) as base_src, \
         rasterio.open(reprojected_dist_map) as dist_src, \
         rasterio.open(reprojected_frtp_map) as frtp_src:
        profile = base_src.profile
        nodata_value = base_src.nodata
        profile.update(dtype=rasterio.float32, compress='lzw')

        # Initialize the next year biomass map
        with rasterio.open(next_map, 'w', **profile) as dst:
            # Iterate over blocks
            for ji, window in base_src.block_windows(1):
                # Read blocks from base biomass map
                base_biomass = base_src.read(1, window=window).astype(np.float32)
                nodata_value = base_src.nodata

                # Read corresponding blocks from disturbance and forest type maps
                disturbance_reprojected = dist_src.read(1, window=window)
                frtp_data = frtp_src.read(1, window=window)

                # Create nodata masks
                if nodata_value is not None:
                    nodata_mask = (base_biomass == nodata_value)
                else:
                    nodata_mask = np.zeros_like(base_biomass, dtype=bool)

                frtp_nodata_value = frtp_src.nodata
                if frtp_nodata_value is not None:
                    frtp_nodata_mask = (frtp_data == frtp_nodata_value)
                else:
                    frtp_nodata_mask = np.zeros_like(frtp_data, dtype=bool)

                # Initialize biomass growth array
                biomass_growth = base_biomass.copy()

                # Apply growth and disturbance effects for each forest type
                for code in params_df.index:
                    # Get parameters for this forest type
                    para_1 = params_df.loc[code, 'para_1']
                    para_2 = params_df.loc[code, 'para_2']
                    
                    distb_1 = params_df.loc[code, 'distb_1']

                    # Create mask for pixels of this forest type
                    forest_type_mask = (frtp_data == code)
                    valid_mask = ~nodata_mask & ~frtp_nodata_mask & forest_type_mask

                    # Apply growth to valid pixels (no disturbance)
                    stdage = pow(biomass_growth[valid_mask]/para_1, 1/para_2)
                    biomass_growth[valid_mask] = para_1 * pow((stdage+1), para_2)

                    # Apply disturbance effect where disturbance year matches biomass year
                    disturbance_mask = (disturbance_reprojected == yr)
                    affected_pixels = disturbance_mask & valid_mask
                    biomass_growth[affected_pixels] = base_biomass[affected_pixels] * distb_1

                # Write the block to the output file
                dst.write(biomass_growth, 1, window=window)

    # Remove the temporary reprojected maps
    os.remove(reprojected_dist_map)
    os.remove(reprojected_frtp_map)

