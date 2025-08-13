# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:03:55 2024
@author: xinyuan.wei
Updated 8/4/2025
"""
#########################################################################
### PyFIA Regional Biomass and Forest Attributes Summary Functions ###
#########################################################################
import pandas as pd
import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

###---------------------------------------------------------------------------- 
# State forest biomass
def state_biomass(FIA_Data, inv_yr, n_years, tree_size, tree_spp, tree_spg):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    coty_data = FIA_Data['coty_data']
    cond_data = FIA_Data['cond_data']
    popp_data = FIA_Data['popp_data']
    pops_data = FIA_Data['pops_data']

    # Create a list of inventory years (an inventory cycle)
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # DIA cm to inch
    tree_size = tree_size * 0.393701

    # Initialize an empty DataFrame to collect plot biomass data
    state_plot_biomass = pd.DataFrame()

    for year in inv_yrs:
        # Filter the population plot data for the current inventory year(s)
        popp_filtered = popp_data[popp_data['INVYR'] == year].copy()
        lfd = year % 100 * 100 + 3
        popp_filtered = popp_filtered[popp_filtered['EVALID'] % 10000 == lfd]

        # Merge with population stratum data to get expansion factors
        popp_merged = popp_filtered.merge(
            pops_data[['CN', 'EXPNS', 'ADJ_FACTOR_SUBP', 'ADJ_FACTOR_MICR']],
            left_on='STRATUM_CN', right_on='CN', how='left'
        )

        # Filter condition data for the current inventory year(s)
        cond_filtered = cond_data[cond_data['INVYR'] == year][['PLOT', 
                                                               'CONDPROP_UNADJ',
                                                               'INVYR',
                                                               'FORTYPCD',
                                                               'STDAGE']].copy()

        # Merge tree data with county data to get county names
        tree_merged = tree_data.merge(
            coty_data[['COUNTYCD', 'COUNTYNM']], on='COUNTYCD', how='left'
        )

        # Filter tree data for the inventory year(s), DBH size, and species
        # Start with the base conditions
        conditions = ((tree_merged['INVYR'] == year) & (tree_merged['DIA'] > tree_size))

        # Apply filter on SPCD if tree_spp is provided
        if tree_spp:
            conditions &= tree_merged['SPCD'].isin(tree_spp)

        # Apply filter on SPGRPCD if tree_spg is provided
        if tree_spg:
            conditions &= tree_merged['SPGRPCD'].isin(tree_spg)

        # Filter and create a copy of the dataframe
        tree_filtered = tree_merged[conditions].copy()

        # Remove records with missing carbon data
        tree_filtered.dropna(subset=['CARBON_AG', 'CARBON_BG', 'TPA_UNADJ'], inplace=True)

        # Calculate tree-level aboveground, belowground, and total biomass
        tree_filtered['treeAG'] = tree_filtered['CARBON_AG'] * tree_filtered['TPA_UNADJ']
        tree_filtered['treeBG'] = tree_filtered['CARBON_BG'] * tree_filtered['TPA_UNADJ']
        tree_filtered['treeBM'] = tree_filtered['treeAG'] + tree_filtered['treeBG']

        # Sum biomass at the plot level
        plot_biomass = tree_filtered.groupby('PLOT')[['treeAG', 'treeBG', 'treeBM']].sum().reset_index()
        plot_biomass.rename(columns={'treeAG': 'PlotAG(kg/m2)', 'treeBG': 'PlotBG(kg/m2)', 
                                     'treeBM': 'PlotTB(kg/m2)'}, inplace=True)

        # Merge plot biomass with population data to get expansion factors
        biomass_plot = plot_biomass.merge(
            popp_merged[['PLOT', 'EXPNS', 'ADJ_FACTOR_SUBP', 'ADJ_FACTOR_MICR']],
            on='PLOT', how='left'
        )

        # Merge with condition data to get condition proportion
        biomass_plot = biomass_plot.merge(cond_filtered, on='PLOT', how='left')

        # Merge with plot data to get latitude and longitude
        plot_location = plot_data[['PLOT', 'LAT', 'LON']].drop_duplicates(subset=['PLOT'])
        biomass_plot = biomass_plot.merge(plot_location, on='PLOT', how='left')

        # Fill missing condition proportions with 1 (assumes full plot is forested)
        biomass_plot['CONDPROP_UNADJ'] = biomass_plot['CONDPROP_UNADJ'].fillna(1)

        # Convert plot biomass to per unit area and adjust with condition proportion
        pakgm = 0.000112085  # Conversion factor from pounds per acre to kg per m^2
        biomass_plot['PlotAG(kg/m2)'] *= pakgm * biomass_plot['CONDPROP_UNADJ']
        biomass_plot['PlotBG(kg/m2)'] *= pakgm * biomass_plot['CONDPROP_UNADJ']
        biomass_plot['PlotTB(kg/m2)'] *= pakgm * biomass_plot['CONDPROP_UNADJ']

        # Calculate stratum-level biomass using expansion factors
        m2a = 4046.8564224  # Conversion factor from acres to m^2
        kgTg = 1e9          # Conversion factor from kg to Tg
        factors = biomass_plot['EXPNS'] * biomass_plot['ADJ_FACTOR_SUBP'] * biomass_plot['ADJ_FACTOR_MICR']

        biomass_plot['StraAG'] = (biomass_plot['PlotAG(kg/m2)'] * factors * m2a) / kgTg
        biomass_plot['StraBG'] = (biomass_plot['PlotBG(kg/m2)'] * factors * m2a) / kgTg
        biomass_plot['StraBM'] = (biomass_plot['PlotTB(kg/m2)'] * factors * m2a) / kgTg

        # Merge with tree data to get county names
        tree_plots = tree_filtered[['PLOT', 'COUNTYNM']].drop_duplicates(subset=['PLOT'])
        biomass_plot = biomass_plot.merge(tree_plots, on='PLOT', how='left')

        # Append the results to the state_plot_biomass DataFrame
        state_plot_biomass = pd.concat([state_plot_biomass, biomass_plot], ignore_index=True)

    # Calculate total biomass across all years
    state_biomass = {
        'Aboveground Biomass (TgC)': round(state_plot_biomass['StraAG'].sum(), 2),
        'Belowground Biomass (TgC)': round(state_plot_biomass['StraBG'].sum(), 2),
        'Total Biomass (TgC)': round(state_plot_biomass['StraBM'].sum(), 2)
    }
    
    return state_biomass

###---------------------------------------------------------------------------- 
# County level biomass density
def county_biomass(FIA_Data, inv_yr, n_years, tree_size, tree_spp, tree_spg):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    coty_data = FIA_Data['coty_data']
    cond_data = FIA_Data['cond_data']
    popp_data = FIA_Data['popp_data']
    pops_data = FIA_Data['pops_data']

    # Create a list of inventory years (an inventory cycle)
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # DIA cm to inch
    tree_size = tree_size * 0.393701

    # Initialize an empty DataFrame to collect plot biomass data
    state_plot_biomass = pd.DataFrame()

    for year in inv_yrs:
        # Filter the population plot data for the current inventory year(s)
        popp_filtered = popp_data[popp_data['INVYR'] == year].copy()
        lfd = year % 100 * 100 + 3
        popp_filtered = popp_filtered[popp_filtered['EVALID'] % 10000 == lfd]

        # Merge with population stratum data to get expansion factors
        popp_merged = popp_filtered.merge(
            pops_data[['CN', 'EXPNS', 'ADJ_FACTOR_SUBP', 'ADJ_FACTOR_MICR']],
            left_on='STRATUM_CN', right_on='CN', how='left'
        )

        # Filter condition data for the current inventory year(s)
        cond_filtered = cond_data[cond_data['INVYR'] == year][['PLOT', 
                                                               'CONDPROP_UNADJ',
                                                               'INVYR',
                                                               'FORTYPCD',
                                                               'STDAGE']].copy()

        # Merge tree data with county data to get county names
        tree_merged = tree_data.merge(
            coty_data[['COUNTYCD', 'COUNTYNM']], on='COUNTYCD', how='left'
        )

        # Filter tree data for the inventory year(s), DBH size, and species
        # Start with the base conditions
        conditions = ((tree_merged['INVYR'] == year) & (tree_merged['DIA'] > tree_size))

        # Apply filter on SPCD if tree_spp is provided
        if tree_spp:
            conditions &= tree_merged['SPCD'].isin(tree_spp)

        # Apply filter on SPGRPCD if tree_spg is provided
        if tree_spg:
            conditions &= tree_merged['SPGRPCD'].isin(tree_spg)

        # Filter and create a copy of the dataframe
        tree_filtered = tree_merged[conditions].copy()

        # Remove records with missing carbon data
        tree_filtered.dropna(subset=['CARBON_AG', 'CARBON_BG', 'TPA_UNADJ'], inplace=True)

        # Calculate tree-level aboveground, belowground, and total biomass
        tree_filtered['treeAG'] = tree_filtered['CARBON_AG'] * tree_filtered['TPA_UNADJ']
        tree_filtered['treeBG'] = tree_filtered['CARBON_BG'] * tree_filtered['TPA_UNADJ']
        tree_filtered['treeBM'] = tree_filtered['treeAG'] + tree_filtered['treeBG']

        # Sum biomass at the plot level
        plot_biomass = tree_filtered.groupby('PLOT')[['treeAG', 'treeBG', 'treeBM']].sum().reset_index()
        plot_biomass.rename(columns={'treeAG': 'PlotAG(kg/m2)', 'treeBG': 'PlotBG(kg/m2)', 
                                     'treeBM': 'PlotTB(kg/m2)'}, inplace=True)

        # Merge plot biomass with population data to get expansion factors
        biomass_plot = plot_biomass.merge(
            popp_merged[['PLOT', 'EXPNS', 'ADJ_FACTOR_SUBP', 'ADJ_FACTOR_MICR']],
            on='PLOT', how='left'
        )

        # Merge with condition data to get condition proportion
        biomass_plot = biomass_plot.merge(cond_filtered, on='PLOT', how='left')

        # Merge with plot data to get latitude and longitude
        plot_location = plot_data[['PLOT', 'LAT', 'LON']].drop_duplicates(subset=['PLOT'])
        biomass_plot = biomass_plot.merge(plot_location, on='PLOT', how='left')

        # Fill missing condition proportions with 1 (assumes full plot is forested)
        biomass_plot['CONDPROP_UNADJ'] = biomass_plot['CONDPROP_UNADJ'].fillna(1)

        # Convert plot biomass to per unit area and adjust with condition proportion
        pakgm = 0.000112085  # Conversion factor from pounds per acre to kg per m^2
        biomass_plot['PlotAG(kg/m2)'] *= pakgm * biomass_plot['CONDPROP_UNADJ']
        biomass_plot['PlotBG(kg/m2)'] *= pakgm * biomass_plot['CONDPROP_UNADJ']
        biomass_plot['PlotTB(kg/m2)'] *= pakgm * biomass_plot['CONDPROP_UNADJ']

        # Calculate stratum-level biomass using expansion factors
        m2a = 4046.8564224  # Conversion factor from acres to m^2
        kgTg = 1e9          # Conversion factor from kg to Tg
        factors = biomass_plot['EXPNS'] * biomass_plot['ADJ_FACTOR_SUBP'] * biomass_plot['ADJ_FACTOR_MICR']

        biomass_plot['StraAG'] = (biomass_plot['PlotAG(kg/m2)'] * factors * m2a) / kgTg
        biomass_plot['StraBG'] = (biomass_plot['PlotBG(kg/m2)'] * factors * m2a) / kgTg
        biomass_plot['StraBM'] = (biomass_plot['PlotTB(kg/m2)'] * factors * m2a) / kgTg

        # Merge with tree data to get county names
        tree_plots = tree_filtered[['PLOT', 'COUNTYNM']].drop_duplicates(subset=['PLOT'])
        biomass_plot = biomass_plot.merge(tree_plots, on='PLOT', how='left')

        # Append the results to the state_plot_biomass DataFrame
        state_plot_biomass = pd.concat([state_plot_biomass, biomass_plot], ignore_index=True)

    # Calculate county biomass density
    county_biomass = (
    state_plot_biomass.groupby('COUNTYNM')[['PlotAG(kg/m2)', 'PlotBG(kg/m2)', 'PlotTB(kg/m2)']]
    .mean().reset_index()
    .rename(columns={'PlotAG(kg/m2)': 'AGB(kg/m2)', 'PlotBG(kg/m2)': 'BGB(kg/m2)', 'PlotTB(kg/m2)': 'TB(kg/m2)'}))

    return county_biomass

###---------------------------------------------------------------------------- 
# State-level summary for a forest variable
def state_summary(data, var):
    # Load data (CSV path or DataFrame)
    df = pd.read_csv(data, low_memory=False) if isinstance(data, str) else data.copy()

    # Check target variable
    if var not in df.columns:
        raise ValueError(f'Variable "{var}" not found in the input data.')

    # Prepare numeric series
    s = pd.to_numeric(df[var], errors='coerce').dropna()
    if s.empty:
        raise ValueError(f'No valid numeric values for "{var}" in the input data.')

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(s.values, bins='auto', alpha=0.75, edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.title(f'Histogram of {var} (State)')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Compute summary statistics
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    mean = s.mean()
    std = s.std(ddof=1)
    cv = (std / mean) if mean not in (0, np.nan) else np.nan
    iqr = q.loc[0.75] - q.loc[0.25]
    skew = s.skew()
    kurt = s.kurtosis()

    # Print summary
    print(f'State Summary For {var}:')
    print(f'Count: {int(s.count())}')
    print(f'Mean: {mean:.4f}')
    print(f'Std: {std:.4f}')
    print(f'Coefficient Of Variation: {cv:.4f}')
    print(f'Min: {s.min():.4f}')
    print(f'5%: {q.loc[0.05]:.4f}')
    print(f'25%: {q.loc[0.25]:.4f}')
    print(f'Median: {q.loc[0.5]:.4f}')
    print(f'75%: {q.loc[0.75]:.4f}')
    print(f'95%: {q.loc[0.95]:.4f}')
    print(f'Max: {s.max():.4f}')
    print(f'IQR: {iqr:.4f}')
    print(f'Skewness: {skew:.4f}')
    print(f'Kurtosis: {kurt:.4f}')

    # Return cleaned table (rows where var is numeric) and stats
    return df.loc[s.index].copy(), {
        'count': int(s.count()),
        'mean': float(mean),
        'std': float(std),
        'cv': float(cv) if np.isfinite(cv) else np.nan,
        'min': float(s.min()),
        'p05': float(q.loc[0.05]),
        'p25': float(q.loc[0.25]),
        'median': float(q.loc[0.5]),
        'p75': float(q.loc[0.75]),
        'p95': float(q.loc[0.95]),
        'max': float(s.max()),
        'iqr': float(iqr),
        'skewness': float(skew),
        'kurtosis': float(kurt)
    }

###---------------------------------------------------------------------------- 
# Area of Interest (AOI) summary
def AOI_summary(data, AOI, savefile, var):
    
    if isinstance(AOI, str):
        aoi_gdf = gpd.read_file(AOI)
    elif isinstance(AOI, gpd.GeoDataFrame):
        aoi_gdf = AOI.copy()
    else:
        # Assume a shapely geometry
        aoi_gdf = gpd.GeoDataFrame(geometry=[AOI], crs='EPSG:4326')

    # Validate AOI
    if aoi_gdf.empty:
        raise ValueError('AOI contains no geometries.')
    if aoi_gdf.crs is None:
        raise ValueError('AOI has no CRS. Please assign a CRS before calling this function.')

    # Standardize to WGS84
    aoi_gdf = aoi_gdf.to_crs('EPSG:4326')

    # Dissolve to a single geometry and try to fix geometry if needed
    aoi_geom = aoi_gdf.geometry.unary_union
    try:
        aoi_geom = aoi_geom.buffer(0)
    except Exception:
        pass

    # Reproject to Web Mercator for basemap
    aoi_web = gpd.GeoDataFrame(geometry=[aoi_geom], crs='EPSG:4326').to_crs('EPSG:3857')

    # Plot AOI polygon and add basemap
    fig, ax = plt.subplots(figsize=(9, 9))
    aoi_web.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5, zorder=3)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs='EPSG:3857', reset_extent=False)

    # Decorate
    ax.set_axis_off()
    ax.set_title('Area Of Interest (AOI)')
    plt.tight_layout()
    plt.show()
    # Load plot data (CSV path or DataFrame)
    df = pd.read_csv(data, low_memory=False) if isinstance(data, str) else data.copy()

    # Check required coordinate fields
    if not {'LON', 'LAT'}.issubset(df.columns):
        raise ValueError('Input data must contain "LON" and "LAT" columns.')

    # Load AOI (shapefile path, GeoDataFrame, or single geometry)
    if isinstance(AOI, str):
        aoi_gdf = gpd.read_file(AOI)
    elif isinstance(AOI, gpd.GeoDataFrame):
        aoi_gdf = AOI.copy()
    else:
        # Assume a shapely geometry
        aoi_gdf = gpd.GeoDataFrame(geometry=[AOI], crs='EPSG:4326')

    if aoi_gdf.empty:
        raise ValueError('AOI contains no geometries.')

    # Ensure AOI is in WGS84 (to match LON/LAT)
    if aoi_gdf.crs is None:
        raise ValueError('AOI has no CRS. Please assign a CRS before calling this function.')
    aoi_gdf = aoi_gdf.to_crs('EPSG:4326')

    # Build a single AOI geometry
    aoi_geom = aoi_gdf.geometry.unary_union
    try:
        aoi_geom = aoi_geom.buffer(0)
    except Exception:
        pass

    # Create GeoDataFrame from plot data
    plot_geo = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['LON'], df['LAT']),
        crs='EPSG:4326'
    )

    # Select plots whose points intersect AOI (includes boundary)
    mask = plot_geo.geometry.intersects(aoi_geom)
    aoi_plots = plot_geo.loc[mask].copy()

    # Ensure output folder exists
    out_dir = os.path.dirname(savefile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save filtered plots to desired format
    ext = os.path.splitext(savefile)[1].lower()
    if ext == '.csv':
        aoi_plots.drop(columns='geometry', errors='ignore').to_csv(savefile, index=False)
    elif ext == '.shp':
        aoi_plots.to_file(savefile, driver='ESRI Shapefile')
    elif ext in ('.gpkg',):
        aoi_plots.to_file(savefile, layer='AOI_Plots', driver='GPKG')
    elif ext in ('.geojson', '.json'):
        aoi_plots.to_file(savefile, driver='GeoJSON')
    else:
        # Default to CSV if extension is unrecognized
        aoi_plots.drop(columns='geometry', errors='ignore').to_csv(savefile, index=False)

    # Prepare the target variable
    if var not in aoi_plots.columns:
        raise ValueError(f'Variable "{var}" not found in filtered AOI plots.')

    s = pd.to_numeric(aoi_plots[var], errors='coerce').dropna()
    if s.empty:
        raise ValueError(f'No valid numeric values for "{var}" within the AOI.')

    # Plot histogram of the variable
    plt.figure(figsize=(8, 5))
    plt.hist(s.values, bins='auto', alpha=0.75, edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.title(f'Histogram of {var} (AOI)')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Compute summary statistics
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    mean = s.mean()
    std = s.std(ddof=1)
    cv = (std / mean) if mean not in (0, np.nan) else np.nan
    iqr = q.loc[0.75] - q.loc[0.25]
    skew = s.skew()
    kurt = s.kurtosis()

    # Print summary
    print(f'AOI Summary For {var}:')
    print(f'Count: {int(s.count())}')
    print(f'Mean: {mean:.4f}')
    print(f'Std: {std:.4f}')
    print(f'Coefficient Of Variation: {cv:.4f}')
    print(f'Min: {s.min():.4f}')
    print(f'5%: {q.loc[0.05]:.4f}')
    print(f'25%: {q.loc[0.25]:.4f}')
    print(f'Median: {q.loc[0.5]:.4f}')
    print(f'75%: {q.loc[0.75]:.4f}')
    print(f'95%: {q.loc[0.95]:.4f}')
    print(f'Max: {s.max():.4f}')
    print(f'IQR: {iqr:.4f}')
    print(f'Skewness: {skew:.4f}')
    print(f'Kurtosis: {kurt:.4f}')

    # Return filtered table and basic stats dictionary
    return aoi_plots.drop(columns='geometry', errors='ignore'), {
        'count': int(s.count()),
        'mean': float(mean),
        'std': float(std),
        'cv': float(cv) if np.isfinite(cv) else np.nan,
        'min': float(s.min()),
        'p05': float(q.loc[0.05]),
        'p25': float(q.loc[0.25]),
        'median': float(q.loc[0.5]),
        'p75': float(q.loc[0.75]),
        'p95': float(q.loc[0.95]),
        'max': float(s.max()),
        'iqr': float(iqr),
        'skewness': float(skew),
        'kurtosis': float(kurt)
    }