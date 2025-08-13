# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:40:21 2023
@author: xinyuan.wei
Updated 8/4/2025
"""
#########################################################################
### PyFIA State Plot-Level Statistical Functions ###
#########################################################################
import pandas as pd
import numpy as np

###----------------------------------------------------------------------------
# Statewide plot-level statistics for DBH (cm) and height (m) (Mean, STD, and max)
def plot_D_H_stats(FIA_Data, inv_yr, n_years, tree_size, tree_spp, tree_spg):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    cond_data = FIA_Data['cond_data']
    
    # Create a list of inventory years
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # Filter the tree data for the inventory period
    tree_data = tree_data[tree_data['INVYR'].isin(inv_yrs)].copy()
    
    # Keep only the necessary columns
    columns_keep = [
        'INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'SUBP', 'TREE', 'CONDID',
        'STATUSCD', 'SPCD', 'SPGRPCD', 'DIA', 'HT'
    ]
    tree_data = tree_data[columns_keep]
    
    # Filter based on diameter threshold and optional species or group codes
    tree_size = tree_size * 0.393701
    conditions = (tree_data['DIA'] > tree_size)
    if tree_spp:
        conditions &= tree_data['SPCD'].isin(tree_spp)
    if tree_spg:
        conditions &= tree_data['SPGRPCD'].isin(tree_spg)
    tree_data = tree_data[conditions].copy()
    
    # Convert DBH from inches to centimeters and height from feet to meters
    tree_data['DIA_cm'] = tree_data['DIA'] * 2.54
    tree_data['HT_m'] = tree_data['HT'] * 0.30482
    
    # Filter the plot data and drop duplicate entries per plot-year
    plot_data = plot_data[plot_data['INVYR'].isin(inv_yrs)]
    plot_data = plot_data.drop_duplicates(subset=['PLOT', 'INVYR'])
    
    # Merge latitude/longitude into tree_data
    tree_data = pd.merge(
        tree_data,
        plot_data[['PLOT', 'LAT', 'LON', 'INVYR']],
        on=['PLOT', 'INVYR'],
        how='left'
    )
    
    # Group and apply the function
    plot_stats = (
        tree_data
        .groupby(['INVYR', 'PLOT'])
        .agg({
            'LAT':   'first',
            'LON':   'first',
            'DIA_cm': ['mean', 'std', 'max'],
            'HT_m':  ['mean', 'std', 'max']
        })
        .reset_index()
    )
    
    # Flatten the MultiIndex columns
    plot_stats.columns = [
        'INVYR', 'PLOT',
        'LAT', 'LON',
        'Mean_DBH_cm', 'STD_DBH_cm', 'Max_DBH_cm',
        'Mean_H_m', 'STD_H_m', 'Max_H_m',
    ]
    
    # Filter condition data (FORTYPCD, STDAGE, and COUNTYCD) and merge
    cond_filtered = cond_data[cond_data['INVYR'].isin(inv_yrs)].copy()
    cond_keep = ['PLOT', 'INVYR', 'FORTYPCD', 'STDAGE', 'COUNTYCD']
    cond_filtered = cond_filtered[cond_keep].drop_duplicates(subset=['PLOT', 'INVYR'])
    
    plot_stats = pd.merge(
        plot_stats,
        cond_filtered,
        on=['PLOT', 'INVYR'],
        how='left'
    )
    
    return plot_stats

###----------------------------------------------------------------------------
# Statewide plot-level structure metrics (top height and DBH)
def plot_str_metrics(FIA_Data, inv_yr, n_years, tree_size, tree_spp, tree_spg, top_percents):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    cond_data = FIA_Data['cond_data']
    
    # Default top percents if not provided
    if top_percents is None:
        top_percents = [10, 25, 40, 50, 60, 75, 90]
        
    # Keep unique and sorted for consistent output (largest first is useful for naming)
    top_percents = sorted(set(top_percents))
    
    # Create a list of inventory years
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # Filter tree data for inventory years
    tree_data = tree_data[tree_data['INVYR'].isin(inv_yrs)].copy()
    
    # Keep only the necessary columns
    columns_keep = [
        'INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'SUBP', 'TREE', 'CONDID',
        'STATUSCD', 'SPCD', 'SPGRPCD', 'DIA', 'HT', 'TPA_UNADJ'
    ]
    tree_data = tree_data[columns_keep]
    
    # Filter based on diameter threshold and optional species or group codes
    tree_size = tree_size * 0.393701
    conditions = (tree_data['DIA'] > tree_size)
    if tree_spp:
        conditions &= tree_data['SPCD'].isin(tree_spp)
    if tree_spg:
        conditions &= tree_data['SPGRPCD'].isin(tree_spg)
    tree_data = tree_data[conditions].copy()
    
    # Convert DBH from inches to centimeters and height from feet to meters
    tree_data['HT_m'] = tree_data['HT'] * 0.30482
    tree_data['DIA_cm'] = tree_data['DIA'] * 2.54
    
    # Filter the plot data and drop duplicate entries per plot-year
    plot_data = plot_data[plot_data['INVYR'].isin(inv_yrs)]
    plot_data = plot_data.drop_duplicates(subset=['PLOT', 'INVYR'])
    
    # Merge location info into tree_data
    tree_data = pd.merge(
        tree_data,
        plot_data[['PLOT', 'INVYR', 'LAT', 'LON']],
        on=['PLOT', 'INVYR'],
        how='left'
    )
    
    # Function to compute top proportion metrics for one group
    def summarize(df):
        heights = df['HT_m']
        dbh = df['DIA_cm']
        lat = df['LAT'].iloc[0] if not df.empty else None
        lon = df['LON'].iloc[0] if not df.empty else None
        
        out = {'LAT': lat, 'LON': lon}
        
        if heights.empty or dbh.empty:
            # Zero out all requested metrics
            for p in top_percents:
                out[f'top{p}_height_count'] = 0
                out[f'avg_top{p}_height_m'] = 0
                out[f'top{p}_dbh_count'] = 0
                out[f'avg_top{p}_dbh_cm'] = 0
            return pd.Series(out)
        
        # Precompute quantiles: threshold quantile for top p% is 1 - p/100
        height_qs = heights.quantile([1 - p / 100 for p in top_percents])
        dbh_qs = dbh.quantile([1 - p / 100 for p in top_percents])
        
        # Mapping from percentile to quantile key
        for p in top_percents:
            h_thresh = height_qs.get(1 - p / 100, None)
            d_thresh = dbh_qs.get(1 - p / 100, None)
            
            if h_thresh is None or pd.isna(h_thresh):
                out[f'top{p}_height_count'] = 0
                out[f'avg_top{p}_height_m'] = 0
            else:
                selected_h = heights[heights >= h_thresh]
                out[f'top{p}_height_count'] = selected_h.count()
                out[f'avg_top{p}_height_m'] = selected_h.mean() if not selected_h.empty else 0
            
            if d_thresh is None or pd.isna(d_thresh):
                out[f'top{p}_dbh_count'] = 0
                out[f'avg_top{p}_dbh_cm'] = 0
            else:
                selected_d = dbh[dbh >= d_thresh]
                out[f'top{p}_dbh_count'] = selected_d.count()
                out[f'avg_top{p}_dbh_cm'] = selected_d.mean() if not selected_d.empty else 0
        
        return pd.Series(out)
    
    # Group and apply the function
    plot_metrics = (
        tree_data
        .groupby(['INVYR', 'PLOT'])
        .apply(summarize)
        .reset_index()
    )
    
    # Filter condition data (FORTYPCD, STDAG. and COUNTYCD) and merge
    cond_filtered = cond_data[cond_data['INVYR'].isin(inv_yrs)].copy()
    cond_keep = ['PLOT', 'INVYR', 'FORTYPCD', 'STDAGE', 'COUNTYCD']
    cond_filtered = cond_filtered[cond_keep].drop_duplicates(subset=['PLOT', 'INVYR'])
    
    plot_metrics = pd.merge(
        plot_metrics,
        cond_filtered,
        on=['PLOT', 'INVYR'],
        how='left'
    )
    
    return plot_metrics

###----------------------------------------------------------------------------
# Statewide plot-level forest structure indices
def plot_str_indices(FIA_Data, inv_yr, n_years, tree_size, tree_spp, tree_spg):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    cond_data = FIA_Data['cond_data']
    
    # Create inventory year list
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # Filter tree data to inventory years and keep necessary columns
    tree_data = tree_data[tree_data['INVYR'].isin(inv_yrs)].copy()
    tree_cols = [
        'INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'SUBP', 'TREE', 'CONDID',
        'STATUSCD', 'SPCD', 'SPGRPCD', 'DIA', 'HT', 'TPA_UNADJ'
    ]
    tree_data = tree_data[tree_cols]
    
    # Filter based on diameter threshold and optional species or group codes
    tree_size = tree_size * 0.393701
    conditions = (tree_data['DIA'] > tree_size)
    if tree_spp:
        conditions &= tree_data['SPCD'].isin(tree_spp)
    if tree_spg:
        conditions &= tree_data['SPGRPCD'].isin(tree_spg)
    tree_data = tree_data[conditions].copy()
    
    # Convert DBH from inches to centimeters and height from feet to meters
    tree_data['DIA_cm'] = tree_data['DIA'] * 2.54
    tree_data['HT_m'] = tree_data['HT'] * 0.30482
    
    # Prepare plot data and merge location
    plot_data = plot_data[plot_data['INVYR'].isin(inv_yrs)]
    plot_data = plot_data.drop_duplicates(subset=['PLOT', 'INVYR'])
    tree_data = pd.merge(
        tree_data,
        plot_data[['PLOT', 'INVYR', 'LAT', 'LON']],
        on=['PLOT', 'INVYR'],
        how='left'
    )
    
    # Constant plot area used in TPA (same as original: 4 subplots of radius 24 ft)
    plot_area_acres = 4 * (np.pi * (24 ** 2)) / 43560.0  # ft^2 to acres
    
    # Aggregation per plot-year
    def summarize(df):
        diam = df['DIA_cm']
        heights = df['HT_m']
        lat = df['LAT'].iloc[0] if not df.empty else None
        lon = df['LON'].iloc[0] if not df.empty else None
        tree_count = len(df)
        
        # Quadratic Mean Diameter (QMD)
        if diam.empty or tree_count == 0:
            qmd = 0
        else:
            qmd = ((diam ** 2).sum() / tree_count) ** 0.5
        
        # CRR Canopy Relief Ratio (CRR)
        if heights.empty:
            crr = 0
        else:
            h_mean = heights.mean()
            h_min = heights.min()
            h_max = heights.max()
            crr = (h_mean - h_min) / (h_max - h_min) if h_max > h_min else 0
        
        # Forest Vertical Structure Score (FVSS)
        if heights.empty or heights.mean() == 0:
            fvss = 0
        else:
            fvss = heights.std() / heights.mean()
        
        # Basal area (sum of individual tree basal areas), using DBH in cm to meters
        if diam.empty:
            ba = 0
        else:
            # radius in meters = (D_cm / 100) / 2
            radii_m = (diam / 100.0) / 2.0
            ba = np.pi * (radii_m ** 2)
            ba = ba.sum()
        
        # Trees per Acre (TPA)
        tpa = tree_count / plot_area_acres if plot_area_acres > 0 else 0
        
        return pd.Series({
            'LAT': lat,
            'LON': lon,
            'QMD_cm': qmd,
            'CRR': crr,
            'FVSS': fvss,
            'BA_m2': ba,
            'TPA': tpa,
            'TreeCount': tree_count
        })

    # Group and apply the function    
    plot_indices = (
        tree_data
        .groupby(['INVYR', 'PLOT'])
        .apply(summarize)
        .reset_index()
    )
    
    # Filter condition data (FORTYPCD, STDAGE, and COUNTYCD) and merge
    cond_filtered = cond_data[cond_data['INVYR'].isin(inv_yrs)].copy()
    cond_filtered = cond_filtered[['PLOT', 'INVYR', 'FORTYPCD', 
                                   'STDAGE', 'COUNTYCD']].drop_duplicates(subset=['PLOT', 'INVYR'])
    
    plot_indices = pd.merge(
        plot_indices,
        cond_filtered,
        on=['PLOT', 'INVYR'],
        how='left'
    )
    
    return plot_indices

###----------------------------------------------------------------------------
# Statewide plot-level forest composition (diversity, evenness, and dominant species)
def plot_composition(FIA_Data, inv_yr, n_years):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    cond_data = FIA_Data['cond_data']
    
    # Create a list of inventory years
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # Filter tree data for the inventory period
    tree_data = tree_data[tree_data['INVYR'].isin(inv_yrs)].copy()
    
    # Keep only necessary columns
    columns_keep = [
        'INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'SUBP', 'TREE',
        'CONDID', 'STATUSCD', 'SPCD', 'DIA', 'HT', 'TPA_UNADJ'
    ]
    tree_data = tree_data[columns_keep]
    
    # Filter and deduplicate plot data, then merge location
    plot_data = plot_data[plot_data['INVYR'].isin(inv_yrs)]
    plot_data = plot_data.drop_duplicates(subset=['PLOT', 'INVYR'])
    tree_data = pd.merge(
        tree_data,
        plot_data[['PLOT', 'INVYR', 'LAT', 'LON']],
        on=['PLOT', 'INVYR'],
        how='left'
    )
    
    # Compute basal area per tree (convert DIA in inches to meters radius)
    tree_data['DIA_cm'] = tree_data['DIA'] * 2.54
    tree_data['radius_m'] = (tree_data['DIA_cm'] / 100.0) / 2.0
    tree_data['BA_m2'] = np.pi * (tree_data['radius_m'] ** 2)
    
    # Group by inventory year and plot to compute metrics
    def summarize(df):
        species_counts = df['SPCD'].value_counts()
        species_richness = species_counts.size
        total_trees = species_counts.sum()
        
        # Shannon diversity index
        if species_richness <= 1 or total_trees == 0:
            shannon = 0.0
            evenness = 0.0
        else:
            p = species_counts / total_trees
            shannon = - (p * np.log(p)).sum()
            evenness = shannon / np.log(species_richness) if species_richness > 1 else 0.0
        
        # Dominant species by count
        if not species_counts.empty:
            dominant_spcd_count = species_counts.idxmax()
            dominant_count = species_counts.iloc[0]
            dominant_spcd_count_prop = dominant_count / total_trees if total_trees > 0 else 0.0
        else:
            dominant_spcd_count = None
            dominant_spcd_count_prop = 0.0
        
        # Dominant species by basal area
        ba_by_sp = df.groupby('SPCD')['BA_m2'].sum()
        total_ba = ba_by_sp.sum()
        if not ba_by_sp.empty:
            dominant_spcd_ba = ba_by_sp.idxmax()
            dominant_ba = ba_by_sp.loc[dominant_spcd_ba]
            dominant_spcd_ba_prop = dominant_ba / total_ba if total_ba > 0 else 0.0
        else:
            dominant_spcd_ba = None
            dominant_spcd_ba_prop = 0.0
            total_ba = 0.0
        
        lat = df['LAT'].iloc[0] if not df.empty else None
        lon = df['LON'].iloc[0] if not df.empty else None
        
        return pd.Series({
            'LAT': lat,
            'LON': lon,
            'Shannon': shannon,
            'Evenness': evenness,
            'Richness': species_richness,
            'Dominant_count': dominant_spcd_count,
            'Dominant_count_prop': dominant_spcd_count_prop,
            'Dominant_ba': dominant_spcd_ba,
            'Dominant_ba_prop': dominant_spcd_ba_prop,
            'BA_m2': total_ba
        })

    # Group and apply the function    
    plot_composition = (
        tree_data
        .groupby(['INVYR', 'PLOT'])
        .apply(summarize)
        .reset_index()
    )
    
    # Filter condition data (FORTYPCD, STDAGE, and COUNTYCD) and merge
    cond_filtered = cond_data[cond_data['INVYR'].isin(inv_yrs)].copy()
    cond_filtered = cond_filtered[['PLOT', 'INVYR', 'FORTYPCD', 
                                   'STDAGE', 'COUNTYCD']].drop_duplicates(subset=['PLOT', 'INVYR'])
    
    plot_composition = pd.merge(
        plot_composition,
        cond_filtered,
        on=['PLOT', 'INVYR'],
        how='left'
    )
    return plot_composition

###----------------------------------------------------------------------------
# Statewide plot-level fraction for given tree species
def plot_f_species(FIA_Data, inv_yr, n_years, SP):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    cond_data = FIA_Data['cond_data']
    
    # Create a list of inventory years (an inventory cycle)
    inv_yrs = [inv_yr - i for i in range(n_years)]
    
    # Filter the data for the inventory period
    tree_data = tree_data[tree_data['INVYR'].isin(inv_yrs)].copy()
    
    # Keep only necessary columns
    columns_keep = ['INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'SUBP', 'TREE', 
                    'CONDID', 'STATUSCD', 'SPCD', 'DIA', 'HT', 'TPA_UNADJ']
    tree_data = tree_data[columns_keep]
    
    # Filter and drop duplicate plots in plot_data
    plot_data = plot_data[plot_data['INVYR'].isin(inv_yrs)]
    plot_data = plot_data.drop_duplicates(subset=['PLOT', 'INVYR'])
    
    # Add location information for each tree record by merging with plot_data
    tree_data = pd.merge(tree_data, plot_data[['PLOT', 'INVYR', 'LAT', 'LON']], 
                         on=['PLOT', 'INVYR'], how='left')
    
    # Calculate the stem area 
    tree_data['STEM_AREA'] = (tree_data['DIA'] / 2) ** 2 * np.pi
    
    # Filter tree species
    data_species = tree_data[tree_data['SPCD'].isin(SP)]

    # Group data by year, plot for total counts and sums
    total_count = tree_data.groupby(['INVYR', 'PLOT', 'LAT', 'LON', 'COUNTYCD'])['TREE'].count().reset_index(name='t_count')
    total_stem = tree_data.groupby(['INVYR', 'PLOT'])['STEM_AREA'].sum().reset_index(name='t_stem')
    
    # Group species by year and plot for total counts and sums
    sp_count = data_species.groupby(['INVYR', 'PLOT'])['TREE'].count().reset_index(name='sp_count')
    sp_stem = data_species.groupby(['INVYR', 'PLOT'])['STEM_AREA'].sum().reset_index(name='sp_stem')
    
    # Merge the data and calculate the fraction
    merged_data = pd.merge(total_count, sp_count, on=['INVYR', 'PLOT'], how='left').fillna(0)
    merged_data['f_tree'] = merged_data['sp_count'] / merged_data['t_count']
    merged_stem = pd.merge(total_stem, sp_stem, on=['INVYR', 'PLOT'], how='left').fillna(0)
    merged_stem['f_stem'] = merged_stem['sp_stem'] / merged_stem['t_stem']
    
    # Combine all data into one DataFrame
    plot_f_species = pd.merge(merged_data, merged_stem[['INVYR', 'PLOT', 'f_stem']], on=['INVYR', 'PLOT'])
    
    # Filter condition data (FORTYPCD, STDAGE, and COUNTYCD) and merge
    cond_filtered = cond_data[cond_data['INVYR'].isin(inv_yrs)].copy()
    cond_filtered = cond_filtered[['PLOT', 'INVYR', 'FORTYPCD', 
                                   'STDAGE']].drop_duplicates(subset=['PLOT', 'INVYR'])
    
    plot_f_species = pd.merge(
        plot_f_species,
        cond_filtered,
        on=['PLOT', 'INVYR'],
        how='left'
    )
    # Return the final DataFrame
    return plot_f_species

###----------------------------------------------------------------------------
# Statewide plot-level aboveground, belowground, and total biomass (kgC/m²)
def plot_biomass(FIA_Data, inv_yr, n_years, tree_size, tree_spp, tree_spg):
    tree_data = FIA_Data['tree_data']
    plot_data = FIA_Data['plot_data']
    cond_data = FIA_Data['cond_data']
    
    # Create list of inventory years
    inv_yrs = [inv_yr - i for i in range(n_years)]

    # Filter tree records for the inventory years
    tree_data = tree_data[tree_data['INVYR'].isin(inv_yrs)].copy()
    
    # Keep only necessary columns
    columns_keep = [
        'INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'SUBP', 'TREE', 'CONDID',
        'STATUSCD', 'SPCD', 'SPGRPCD', 'DIA', 'HT', 'CARBON_AG', 'CARBON_BG'
    ]
    tree_data = tree_data[columns_keep]
    
    # Apply filters for tree size and optional species/group
    tree_size = tree_size * 0.393701  # cm to inches
    conditions = (tree_data['DIA'] > tree_size)
    if tree_spp:
        conditions &= tree_data['SPCD'].isin(tree_spp)
    if tree_spg:
        conditions &= tree_data['SPGRPCD'].isin(tree_spg)
    tree_data = tree_data[conditions].copy()

    # Drop missing values
    tree_data.dropna(subset=['CARBON_AG', 'CARBON_BG'], inplace=True)

    # Sum CARBON_AG and CARBON_BG at the plot level (in lb)
    plot_carbon = (
        tree_data
        .groupby(['INVYR', 'PLOT'])
        .agg({
            'CARBON_AG': 'sum',
            'CARBON_BG': 'sum'
        })
        .reset_index()
    )
    
    # Calculate total carbon (lb)
    plot_carbon['CARBON_TOT'] = plot_carbon['CARBON_AG'] + plot_carbon['CARBON_BG']

    # Convert lb to kg: 1 lb = 0.45359237 kg
    plot_carbon['AG_kg'] = plot_carbon['CARBON_AG'] * 0.45359237
    plot_carbon['BG_kg'] = plot_carbon['CARBON_BG'] * 0.45359237
    plot_carbon['TOT_kg'] = plot_carbon['CARBON_TOT'] * 0.45359237

    # Total subplot area: 4 subplots × π × (24 ft)²
    # 1 ft = 0.3048 m → radius = 24 × 0.3048
    subplot_radius_m = 24 * 0.3048
    plot_area_m2 = 4 * np.pi * (subplot_radius_m ** 2)

    # Convert to kgC/m²
    plot_carbon['AG_kg_m2'] = plot_carbon['AG_kg'] / plot_area_m2
    plot_carbon['BG_kg_m2'] = plot_carbon['BG_kg'] / plot_area_m2
    plot_carbon['TOT_kg_m2'] = plot_carbon['TOT_kg'] / plot_area_m2

    # Merge LAT, LON
    plot_location = plot_data[['PLOT', 'INVYR', 'LAT', 'LON']].drop_duplicates()
    plot_carbon = plot_carbon.merge(plot_location, on=['PLOT', 'INVYR'], how='left')

    # Merge FORTYPCD and STDAGE
    cond_info = cond_data[cond_data['INVYR'].isin(inv_yrs)][['PLOT', 'INVYR', 'FORTYPCD', 'STDAGE']].drop_duplicates()
    plot_carbon = plot_carbon.merge(cond_info, on=['PLOT', 'INVYR'], how='left')

    # Select final columns
    plot_biomass = plot_carbon[[
        'INVYR', 'PLOT', 'LAT', 'LON', 'FORTYPCD', 'STDAGE',
        'AG_kg_m2', 'BG_kg_m2', 'TOT_kg_m2'
    ]].copy()

    # Rename columns for clarity
    plot_biomass.rename(columns={
        'AG_kg_m2': 'AGB_kgC/m2',
        'BG_kg_m2': 'BGB_kgC/m2',
        'TOT_kg_m2': 'TB_kgC/m2'
    }, inplace=True)

    return plot_biomass

###----------------------------------------------------------------------------
# Statewide plot-level harvested volume
def plot_harvest(FIA_Data, sta_yr, end_yr, land, etype, units):
    tree_data = FIA_Data['tree_data']
    tgrm_data = FIA_Data['tgrm_data']
    cond_data = FIA_Data['cond_data']
    
    # Filter harvesting data based on harvesting information
    tgrm_data = tgrm_data[(tgrm_data['LAND_BASIS'] == land) & 
                          (tgrm_data['ESTN_TYPE'] == etype) & 
                          (tgrm_data['ESTN_UNITS'] == units)]
    
    # Create a subset of cond_data with the fields 'PLT_CN' and 'PLOT'
    cond_subset = cond_data[['PLT_CN', 'PLOT']]

    # Merge tgrm_data with cond on 'PLT_CN' to get the corresponding 'PLOT'
    tgrm_data = tgrm_data.merge(cond_subset, on='PLT_CN', how='left')

    # Filter data based on the year range (INVYR)
    fyr_tree_data = tree_data[(tree_data['INVYR'] >= sta_yr) & (tree_data['INVYR'] <= end_yr)].copy()
    fyr_tgrm_data = tgrm_data[(tgrm_data['INVYR'] >= sta_yr) & (tgrm_data['INVYR'] <= end_yr)]

    # Sum plot volume (VOLCFNET) by grouping on both 'INVYR' and 'PLOT'
    fyr_tree_data['mvol'] = fyr_tree_data['VOLCFNET'] * fyr_tree_data['TPA_UNADJ']
    plot_vol = fyr_tree_data.groupby(['INVYR', 'PLOT'])[['mvol']].sum().reset_index()
    plot_vol = plot_vol.rename(columns={'mvol': 'Volume(ft3)'})

    # Sum harvested volume (REMOVALS) by grouping on both 'INVYR' and 'PLOT'
    plot_har = fyr_tgrm_data.groupby(['INVYR', 'PLOT'])[['REMOVALS']].sum().reset_index()
    plot_har = plot_har.rename(columns={'REMOVALS': 'Harvested(ft3)'})

    # Merge the two summary tables on 'INVYR' and 'PLOT'
    plot_harvest = pd.merge(plot_vol, plot_har, on=['INVYR', 'PLOT'], how='outer')

    # Compute the harvested fraction: fhar = plot_har / plot_vol
    plot_harvest['Fraction'] = plot_harvest['Harvested(ft3)'] / plot_harvest['Volume(ft3)']

     # Return the final DataFrame
    return plot_harvest

###----------------------------------------------------------------------------
# Statewide plot-level forest type and type group 
def plot_type(FIA_Data, sta_yr, end_yr):

    # Function converts FORTYPCD (Forest Type) to TYPGRPCD (Group Code)
    def map_fortypcd_to_typgrp(x):
        try:
            x = float(x)
        except:
            return None
        if 100 <= x <= 120:
            return 100
        elif 121 <= x <= 140:
            return 120
        elif 141 <= x <= 150:
            return 140
        elif 151 <= x <= 160:
            return 150
        elif 161 <= x <= 170:
            return 160
        elif 171 <= x <= 180:
            return 170
        elif 181 <= x <= 200:
            return 200
        elif 201 <= x <= 220:
            return 220
        elif 221 <= x <= 240:
            return 240
        elif 241 <= x <= 260:
            return 260
        elif 261 <= x <= 280:
            return 280
        elif 281 <= x <= 300:
            return 300
        elif 301 <= x <= 320:
            return 320
        elif 321 <= x <= 340:
            return 340
        elif 341 <= x <= 360:
            return 360
        elif 361 <= x <= 370:
            return 370
        elif 371 <= x <= 380:
            return 380
        elif 381 <= x <= 390:
            return 390
        elif 391 <= x <= 400:
            return 400
        elif 401 <= x <= 500:
            return 500
        elif 501 <= x <= 600:
            return 600
        elif 601 <= x <= 700:
            return 700
        elif 701 <= x <= 800:
            return 800
        elif 801 <= x <= 900:
            return 900
        elif 901 <= x <= 910:
            return 910
        elif 911 <= x <= 920:
            return 920
        elif 921 <= x <= 940:
            return 940
        elif 941 <= x <= 960:
            return 960
        elif 961 <= x <= 970:
            return 970
        elif 971 <= x <= 980:
            return 980
        elif 981 <= x <= 990:
            return 990
        elif x == 999:
            return 999
        else:
            return None

    # Get condition data.
    cond_data = FIA_Data['cond_data']
    
    # Filter data based on the year range (INVYR)
    filtered_data = cond_data[(cond_data['INVYR'] >= sta_yr) & (cond_data['INVYR'] <= end_yr)].copy()
    
    # Create the pivot table for FORTYPCD
    foresttype = filtered_data.pivot_table(index='PLOT', columns='INVYR', 
                                           values='FORTYPCD', aggfunc='first')
    
    # Reset index to include 'PLOT' as a column
    foresttype = foresttype.reset_index()
    
    # Create a new column 'TYPGRPCD' based on FORTYPCD
    filtered_data['TYPGRPCD'] = filtered_data['FORTYPCD'].apply(map_fortypcd_to_typgrp)
    
    # Create a pivot table for TYPGRPCD
    foresttype_group = filtered_data.pivot_table(index='PLOT', columns='INVYR', 
                                                 values='TYPGRPCD', aggfunc='first')
    
    # Reset index to include 'PLOT' as a column
    foresttype_group = foresttype_group.reset_index()
    
    # Rename columns for the TYPGRPCD pivot table (except 'PLOT') to distinguish them
    foresttype_group = foresttype_group.rename(columns=lambda x: f"TYPGRPCD_{x}" if x != "PLOT" else x)
    
    # Merge the two pivot tables on 'PLOT' so the final table includes both FORTYPCD and TYPGRPCD.
    plot_type = foresttype.merge(foresttype_group, on='PLOT')
    
    # Return the final DataFrame
    return plot_type
