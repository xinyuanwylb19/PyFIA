# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 08:34:18 2024
@author: xinyuan.wei
"""
import pandas as pd
import geopandas as gpd
import _pyFIA_Plot_Stats as ps
import _pyFIA_Regional_Stats as rs
import _pyFIA_Visualization as vl
import _pyFIA_Interpolation as ip
import _pyFIA_Climate as cl
import _pyFIA_Bookkeeping as bk
import _pyFIA_Tools as tl


# State and FIPS
State = 'ME'
FIPS = '23'

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
# Statewide plot-level forest attributes analyses
###----------------------------------------------------------------------------
'''
# Statewide plot-level statistics for DBH (cm) and height (m)
savefile = f'_Results/{State}_Plot_DBH_H_Stats.csv'
plot_D_H_stats = ps.plot_D_H_stats(FIA_Data=FIA_Data, 
                                   inv_yr=2023,          # The ending inventory year
                                   n_years=5,            # Inventory cycle (years) 
                                   tree_size=0,          # Minimum tree diameter (cm)
                                   tree_spp=[],          # Species (empty-all species)
                                   tree_spg=[])          # species group (empty-all species)  
plot_D_H_stats.to_csv(savefile, index=False)
print(plot_D_H_stats.head())

# Statewide plot-level structure metrics
savefile = f'_Results/{State}_Plot_Structure_Metrics.csv'
plot_metrics = ps.plot_str_metrics(FIA_Data=FIA_Data, 
                                   inv_yr=2023,          # The ending inventory year
                                   n_years=5,            # Inventory cycle (years) 
                                   tree_size=0,          # Minimum tree diameter (cm)
                                   tree_spp=[],          # Species (empty-all species)
                                   tree_spg=[],          # species group (empty-all species) 
                                   top_percents=None)    # Top percent (None or [15])    
plot_metrics.to_csv(savefile, index=False)
print(plot_metrics.head())

# Statewide plot-level forest structure indices
savefile = f'_Results/{State}_Plot_Structure_Indices.csv'
plot_indices = ps.plot_str_indices(FIA_Data=FIA_Data, 
                                   inv_yr=2023,          # The ending inventory year
                                   n_years=5,            # Inventory cycle (years) 
                                   tree_size=0,          # Minimum tree diameter (cm)
                                   tree_spp=[],          # Species (empty-all species)
                                   tree_spg=[])          # species group (empty-all species)   
plot_indices.to_csv(savefile, index=False)
print(plot_indices.head())

# Statewide plot-level forest composition
savefile = f'_Results/{State}_Plot_Composition.csv'
plot_composition = ps.plot_composition(FIA_Data=FIA_Data,
                                       inv_yr=2023,      # The ending inventory year
                                       n_years=5)        # Inventory cycle (years)
plot_composition.to_csv(savefile, index=False) 
print(plot_composition.head())

# Statewide plot-level fraction for given tree species
savefile = f'_Results/{State}_Plot_Species_Proportion.csv'
plot_f_species = ps.plot_f_species(FIA_Data=FIA_Data,
                                   inv_yr=2023,          # The ending inventory year
                                   n_years=5,            # Inventory cycle (years)
                                   SP=[12])              # Species list
plot_f_species.to_csv(savefile, index=False) 
print(plot_f_species.head())

# Statewide plot-level aboveground, belowground, and total biomass (kgC/m²)
savefile = f'_Results/{State}_Plot_Biomass.csv'
plot_biomass = ps.plot_biomass(FIA_Data=FIA_Data, 
                               inv_yr=2023,          # The ending inventory year
                               n_years=5,            # Inventory cycle (years) 
                               tree_size=0,          # Minimum tree diameter (cm)
                               tree_spp=[],          # Species (empty-all species)
                               tree_spg=[])          # species group (empty-all species)   
plot_biomass.to_csv(savefile, index=False)
print(plot_biomass.head())

# Statewide plot-level harvested volume
savefile = f'_Results/{State}_Plot_Harvested_Volume.csv'
plot_harvest = ps.plot_harvest(FIA_Data=FIA_Data, 
                               sta_yr=2000,        # The starting year 
                               end_yr=2023,        # The ending year
                               land='FORESTLAND',  # Land type
                               etype='GS',         # Use growing stock removals  
                               units='CF')         # Use cubic feet
plot_harvest.to_csv(savefile, index=False)
print(plot_harvest.head())

# Statewide plot-level forest type and type group
savefile = f'_Results/{State}_Plot_ForestType.csv'
plot_type = ps.plot_type(FIA_Data=FIA_Data,
                         sta_yr=2019,          # The starting year
                         end_yr=2023)          # The ending year
plot_type.to_csv(savefile, index=False)
print(plot_type.head())
'''
###----------------------------------------------------------------------------
# Reginal forest bimass and other attributes summary
###----------------------------------------------------------------------------
'''
# State forest biomass/carbon storage summary
start_yr = 2019
end_yr   = 2023
state_biomass = rs.state_biomass(FIA_Data=FIA_Data,
                                 inv_yr=end_yr,          # The ending inventory year
                                 n_years=5,              # Inventory cycle (years)                           
                                 tree_size=0,            # Minimum tree diameter (cm)
                                 tree_spp=[],            # Species (empty-all species)
                                 tree_spg=[])            # species group (empty-all species)
 
print(f'{State} State Forest Biomass/Carbon Summary ({start_yr}–{end_yr}):')
for key, value in state_biomass.items():
    print(f'{key}: {value}')

# County forest biomass/carbon density
county_biomass = rs.county_biomass(FIA_Data=FIA_Data,
                                   inv_yr=2023,            # The ending inventory year
                                   n_years=5,              # Inventory cycle (years)
                                   tree_size=0,            # Minimum tree diameter (cm)
                                   tree_spp=[],            # Species (empty-all species)
                                   tree_spg=[])            # species group (empty-all species)

savefile = f'_Results/{State}_County_Biomass_Density.csv'
county_biomass.to_csv(savefile, index=False)   
print(f'{State} County Level Biomass/Carbon Density:')
print(county_biomass)

# State-level summary for a forest variable
data     = '_Results/ME_Plot_Structure_Indices.csv'
var      = 'BA_m2' 
state_summary = rs.state_summary(data, var)
    
# Area of Interest (AOI) summary
data     = '_Results/ME_Plot_Structure_Indices.csv'
AOI      = '_AOI/Coastal_Region.shp'
var      = 'BA_m2' 
savefile = '_Results/AOI_Coastal_Data.csv'
AOI_summary = rs.AOI_summary(data=data, 
                             AOI=AOI,
                             savefile=savefile,
                             var=var)
'''
###----------------------------------------------------------------------------
# Forest attributes visualization
###----------------------------------------------------------------------------
'''
# Visualize all the invotory plots for a given state
savefile = f'_Results/{State}_FIA_Plots.png'
vl.plot_map(FIA_Data=FIA_Data, 
            basemap=basemap, 
            FIPS=FIPS, 
            savefile=savefile)

# Statewide plot-level basal area visualization
savefile = f'_Results/{State}_TPA.png'
data     = '_Results/ME_Plot_Structure_Indices.csv'
var      = 'BA_m2'
vl.plot_forest_map(data=data,
                   var=var,
                   basemap=basemap,
                   FIPS=FIPS,
                   savefile=savefile)

# Statewide plot-level TPA visualization
savefile = f'_Results/{State}_TPA.png'
data     = '_Results/ME_Plot_Structure_Indices.csv'
var      = 'TPA'
vl.plot_forest_map(data=data,
                   var=var,
                   basemap=basemap,
                   FIPS=FIPS,
                   savefile=savefile)

# Statewide plot-level AGB visualization
savefile = f'_Results/{State}_AGB.png'
data     = '_Results/ME_Plot_Biomass.csv'
var      = 'AGB_kgC/m2'
vl.plot_forest_map(data=data,
                   var=var,
                   basemap=basemap,
                   FIPS=FIPS,
                   savefile=savefile)

# County-level forest carbon density visualization
savefile = f'_Results/{State}_County_Biomass_Density.png'
data     = '_Results/ME_County_Biomass_Density.csv'
var      = 'AG(kg/m2)'
vl.county_biomass_map(data=data,
                      var=var,
                      basemap=basemap,
                      FIPS=FIPS,
                      savefile=savefile)
'''
###----------------------------------------------------------------------------
# Wall-to-wall interpolated map 
###----------------------------------------------------------------------------
'''
# Interpolate the state forest map with land cover
land_cover = '_NLCD/NLCD_ME.tif'
data       = '_Results/ME_Plot_Biomass.csv'
var        = 'AGB_kgC/m2'
savefile   = f'_Results/{State}_Biomass_Map.tif'
ip.map_interpolate(data=data,
                   var=var,
                   basemap=basemap,       
                   FIPS=FIPS,
                   resolution = 0.01,       # Spatial resolution (interpolation)
                   interp_method='nearest', # Interpolation method
                   land_cover=land_cover,
                   savefile=savefile)
'''
###----------------------------------------------------------------------------
# Climate data acquisition and analysis
###----------------------------------------------------------------------------
'''
# Get the climate data for each inventory plot (select an existing data table)
savefile = f'_Results/{State}_Plot_Climate.csv'
data     = '_Results/ME_Plot_Biomass.csv'
cl.climate_data(data=data,                          
                climate_data='_ClimateNA',
                savefile=savefile) 
'''
###----------------------------------------------------------------------------
# Bookkeeping forest biomass tracking
###----------------------------------------------------------------------------
'''
# Bookkeeping model: derive stand growth model parameters
data = '_Results/ME_Plot_Biomass.csv'
bk.stand_growth(data=data, 
                var = 'AGB_kgC/m2',     # Biomass AGB
                inv_yr=2023,            # The ending inventory year
                n_years=5,              # Inventory cycle (years)
                max_age=140,            # Maximum stand age
                forest_type=[801])      # Forest Type

# Bookkeeping model: track the next year forest biomass 
bk.bookkeeping(base_map = '_BK_Biomass/EXP_Biomass_2004.tif', # Base biomass map
               next_map = '_BK_Biomass/EXP_Biomass_2005.tif', # The next year biomass map
               frtp_map = '_BK_Biomass/EXP_Foresttype.tif',   # Forest type map
               dist_map = '_BK_Biomass/EXP_Disturbance.tif',  # Forest disturbance map 
               yr       = 0o4,                                # The distuabnce year
               par_file = '_BK_Biomass/EXP_paras.csv')
'''