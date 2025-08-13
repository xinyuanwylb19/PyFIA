# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 08:34:18 2024
@author: xinyuan.wei
"""
import pandas as pd
import geopandas as gpd
import _pyFIA_Tools as tl


# State and FIPS
State = 'GA'
FIPS = '13'

# Load the FIA Tree, Condition, Plot, County, Pop_stratum, Pop_Stratum_Assgn data
FIA_Data = {
    'tree_data': pd.read_csv(f'{State}_CSV/{State}_TREE.csv', low_memory=False),
    'cond_data': pd.read_csv(f'{State}_CSV/{State}_COND.csv', low_memory=False),
    'plot_data': pd.read_csv(f'{State}_CSV/{State}_PLOT.csv', low_memory=False),
    'coty_data': pd.read_csv(f'{State}_CSV/{State}_COUNTY.csv', low_memory=False),
    'tgrm_data': pd.read_csv(f'{State}_CSV/{State}_TREE_GRM_ESTN.csv', low_memory=False),
    'pops_data': pd.read_csv(f'{State}_CSV/{State}_POP_STRATUM.csv', low_memory=False),
    'popp_data': pd.read_csv(f'{State}_CSV/{State}_POP_PLOT_STRATUM_ASSGN.csv', low_memory=False)
}

# Load the US basemap (shapefile)
basemap = gpd.read_file('_US_Boundary/us_boundary.shp')

###----------------------------------------------------------------------------
# PyFIA Tools
###----------------------------------------------------------------------------

# Recode Plot ID 'PLOT' to COUNTYCD + PLOT 
savefolder = f'_Results/{State}_CSV'
tl.recode_plot(FIA_Data=FIA_Data, savefolder=savefolder, state=State)

# Extract the land cover (NLCD) data for a given state
NLCD_file = '_NLCD/NLCD_2024.tif'
savefile  = f'_NLCD/NLCD_{State}.tif'
tl.state_NLCD(NLCD_file=NLCD_file,
              basemap=basemap,
              savefile=savefile,
              FIPS=FIPS)

# Merge data tables for further analysis
file1 = '_Results/ME_Plot_Biomass_Climate.csv'
file2 = '_Results/ME_Plot_Composition.csv'
file3 = '_Results/ME_Plot_Structure_Indices.csv'
savefile2 = '_Results/ME_Merged.csv'
savefile3 = '_Results/ME_Merged.csv'
# Merge two files
merged = tl.data_merge(file1, file2, savefile=savefile2)
# Merge many files
merged = tl.data_merge([file1, file2, file3], savefile=savefile3)

# Analysis the relationship between variables
data     = '_Results/ME_Plot_Biomass_Climate.csv'
filters  = {'STDAGE': (0, 250), 'AGB_kgC/m2': (5, None)}
ind_var  = 'STDAGE'
dep_var  = 'AGB_kgC/m2'
regression = tl.regre(data=data,
                      ind_var=ind_var,
                      dep_var=dep_var,
                      filters=filters,
                      max_poly=3)
for k, v in regression.items():
    print(k, '=>', v)
