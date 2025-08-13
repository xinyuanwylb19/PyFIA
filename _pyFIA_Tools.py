# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:07:10 2025

@author: xinyuan.wei
"""

#########################################################################
### PyFIA State Biomass Function ###
#########################################################################
import os
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import contextily as ctx
from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

###----------------------------------------------------------------------------###----------------------------------------------------------------------------
def recode_plot(FIA_Data, savefolder, state):
    # Unpack
    tree_data = FIA_Data.get('tree_data')
    plot_data = FIA_Data.get('plot_data')
    coty_data = FIA_Data.get('coty_data')
    cond_data = FIA_Data.get('cond_data')
    popp_data = FIA_Data.get('popp_data')
    pops_data = FIA_Data.get('pops_data')
    tgrm_data = FIA_Data.get('tgrm_data')

    # Ensure output folder exists
    os.makedirs(savefolder, exist_ok=True)

    # Map dict keys -> desired output filenames
    name_map = {
        'tree_data': f'{state}_TREE.csv',
        'cond_data': f'{state}_COND.csv',
        'plot_data': f'{state}_PLOT.csv',
        'coty_data': f'{state}_COUNTY.csv',
        'tgrm_data': f'{state}_TREE_GRM_ESTN.csv',
        'pops_data': f'{state}_POP_STRATUM.csv',
        'popp_data': f'{state}_POP_PLOT_STRATUM_ASSGN.csv',
    }

    tables = {
        'tree_data': tree_data,
        'cond_data': cond_data,
        'plot_data': plot_data,
        'coty_data': coty_data,
        'tgrm_data': tgrm_data,
        'pops_data': pops_data,
        'popp_data': popp_data,
    }

    # Helper: get the first column named `col` as a Series, even if duplicated
    def first_col(df, col):
        if col not in df.columns:
            return None
        obj = df[col]
        return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj

    recoded = {}

    for key, df in tables.items():
        if df is None:
            continue

        df = df.copy()

        if ('COUNTYCD' in df.columns) and ('PLOT' in df.columns):
            has_plot_old = 'PLOT_OLD' in df.columns

            if not has_plot_old:
                # Rename only the FIRST 'PLOT' to 'PLOT_OLD'
                plot_positions = [i for i, c in enumerate(df.columns) if c == 'PLOT']
                first_plot_idx = plot_positions[0]
                cols = list(df.columns)
                cols[first_plot_idx] = 'PLOT_OLD'
                df.columns = cols

                # Insert new 'PLOT' column after 'PLOT_OLD'
                df.insert(first_plot_idx + 1, 'PLOT', pd.NA)

                # Old plot series is the newly renamed 'PLOT_OLD'
                plot_old_s = first_col(df, 'PLOT_OLD')
            else:
                # Reuse existing PLOT_OLD; do NOT create another
                plot_old_s = first_col(df, 'PLOT_OLD')

                # If no PLOT column exists (rare), create one right after first PLOT_OLD
                plot_positions = [i for i, c in enumerate(df.columns) if c == 'PLOT']
                if not plot_positions:
                    first_plot_old_idx = [i for i, c in enumerate(df.columns) if c == 'PLOT_OLD'][0]
                    df.insert(first_plot_old_idx + 1, 'PLOT', pd.NA)

            county_s = first_col(df, 'COUNTYCD')

            # Build mask safely on Series
            mask = county_s.notna() & plot_old_s.notna()

            # Make strings: COUNTYCD_PLOT (drop decimals)
            county_str = pd.to_numeric(county_s[mask], errors='coerce').astype('Int64').astype(str)
            plot_str   = pd.to_numeric(plot_old_s[mask], errors='coerce').astype('Int64').astype(str)
            new_vals   = county_str.str.cat(plot_str, sep='_')

            # Assign into the (single) 'PLOT' column
            df['PLOT'] = df['PLOT'].astype('object')
            df.loc[mask, 'PLOT'] = new_vals

        # Save out
        out_path = os.path.join(savefolder, name_map.get(key, f'{key}.csv'))
        df.to_csv(out_path, index=False)
        recoded[key] = df

    return recoded

###----------------------------------------------------------------------------
# Extract a state's NLCD from the national NLCD GeoTIFF and display the map
def state_NLCD(NLCD_file, basemap, savefile, FIPS):
    # Filter to target state
    state_map = basemap[basemap['STATEFP'] == FIPS].copy()
    if state_map.empty:
        raise ValueError(f'No features found for STATEFP={FIPS}')

    # Ensure output folder exists (if provided)
    out_dir = os.path.dirname(savefile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    forest_classes=(41, 42, 43, 90)
    
    # Open NLCD and clip
    with rasterio.open(NLCD_file) as src:
        # Reproject state geometry to raster CRS
        state_proj = state_map.to_crs(src.crs)

        # Dissolve into a single geometry, fix invalids
        geom = state_proj.geometry.unary_union
        try:
            geom = geom.buffer(0)
        except Exception:
            pass

        shapes = [mapping(geom)]

        # Mask/clip
        nodata_out = src.nodata if src.nodata is not None else 0
        out_img, out_transform = mask(src, shapes, crop=True, nodata=nodata_out)

        # Keep only forest classes; set others to nodata
        band = out_img[0] if out_img.ndim == 3 else out_img
        keep_mask = np.isin(band, np.array(forest_classes, dtype=band.dtype))
        band_filtered = np.where(keep_mask, band, nodata_out).astype(band.dtype)

        # Update metadata for single-band output
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': band_filtered.shape[0],
            'width': band_filtered.shape[1],
            'transform': out_transform,
            'compress': 'deflate',
            'tiled': True,
            'BIGTIFF': 'IF_SAFER',
            'count': 1,
            'nodata': nodata_out
        })

    # Write clipped, forest-only raster
    with rasterio.open(savefile, 'w', **meta) as dst:
        dst.write(band_filtered, 1)

    # Show the forest-only NLCD with labeled legend and a basemap
    with rasterio.open(savefile) as src:
        arr = src.read(1)
        nodata = src.nodata if src.nodata is not None else 0

        # Reproject raster to Web Mercator for basemap compatibility
        dst_crs = 'EPSG:3857'
        transform_3857, width_3857, height_3857 = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        arr_3857 = np.full((height_3857, width_3857), nodata, dtype=arr.dtype)
        reproject(
            source=arr,
            destination=arr_3857,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_3857,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=nodata
        )

        # Build discrete colormap / legend for present classes
        label_map = {41:'Deciduous Forest', 42:'Evergreen Forest', 43:'Mixed Forest', 90:'Woody Wetlands'}
        color_map = {41:'#c5e1a5', 42:'#1b5e20', 43:'#43a047', 90:'#41b6c4'}

        present = sorted(int(v) for v in np.unique(arr_3857) if v != nodata)
        idx = np.full(arr_3857.shape, -1, dtype='int32')
        for i, cls in enumerate(present):
            idx[arr_3857 == cls] = i
        idx = np.ma.masked_less(idx, 0)
        cmap = ListedColormap([color_map.get(cls, '#cccccc') for cls in present])

        # Compute correct extent from transform
        left, bottom, right, top = array_bounds(height_3857, width_3857, transform_3857)

        fig, ax = plt.subplots(figsize=(9, 9))

        # Plot NLCD first so it stays on top after we add the basemap (we'll keep extent)
        ax.imshow(
            idx,
            cmap=cmap,
            interpolation='nearest',
            extent=(left, right, bottom, top),
            zorder=3
        )

        # Add basemap UNDER the raster, keep current extent
        ctx.add_basemap(
            ax,
            source=ctx.providers.OpenStreetMap.Mapnik,
            crs=dst_crs,
            reset_extent=False
        )

        # Overlay state boundary on top
        basemap[basemap['STATEFP'] == FIPS].to_crs(dst_crs).boundary.plot(
            ax=ax, edgecolor='black', linewidth=0.8, zorder=4
        )

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_title('Land Cover')
        ax.set_axis_off()

        if present:
            handles = [Patch(facecolor=color_map.get(c, '#cccccc'), edgecolor='none', label=label_map.get(c, str(c)))
                       for c in present]
            ax.legend(handles=handles, title='NLCD class', loc='upper left', frameon=True)

        plt.tight_layout()
        plt.show()

###----------------------------------------------------------------------------
# Merge data tables for further analysis
def data_merge(*datasets, savefile=None):
    # Accept one or many inputs; each can be a CSV path or a DataFrame
    if len(datasets) == 1 and isinstance(datasets[0], (list, tuple)):
        datasets = tuple(datasets[0])
    if len(datasets) == 0:
        raise ValueError('Provide at least one CSV path or DataFrame.')

    # Load inputs to DataFrames
    def _to_df(d):
        return pd.read_csv(d, low_memory=False) if isinstance(d, str) else d.copy()
    frames = [_to_df(d) for d in datasets]

    # Ensure join keys exist and harmonize dtypes
    on = ['INVYR', 'PLOT']
    for i, df in enumerate(frames, start=1):
        missing = [k for k in on if k not in df.columns]
        if missing:
            raise ValueError(f'Merge key(s) {missing} not found in table #{i}.')
        df['INVYR'] = pd.to_numeric(df['INVYR'], errors='coerce').astype('Int64')
        df['PLOT'] = df['PLOT'].astype('string')

    # If only one table, optionally save and return
    if len(frames) == 1:
        merged_file = frames[0]
        if savefile:
            out_dir = os.path.dirname(savefile)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            merged_file.to_csv(savefile, index=False)
        return merged_file

    # Iterative merge using inner join on ['INVYR','PLOT']
    merged_file = frames[0]
    for i, df in enumerate(frames[1:], start=2):
        merged_file = pd.merge(
            merged_file, df,
            how='inner',
            on=on,
            suffixes=('', f'_tbl{i}')
        )

    # Save if requested
    if savefile:
        out_dir = os.path.dirname(savefile)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        merged_file.to_csv(savefile, index=False)

    return merged_file
    
###----------------------------------------------------------------------------   
# Regression analysis between variables (univariate)
# - Fits linear, polynomial (up to max_poly), log-x (x>0), exp-y (y>0), power (x>0 & y>0)
# - Plots scatter and all fitted curves with equations and R² in the legend
def regre(data, ind_var, dep_var, filters=None, max_poly=3, plot=True):
    # Load data
    df = pd.read_csv(data)

    # Apply data filters if provided
    if filters:
        for col, cond in filters.items():
            if isinstance(cond, tuple):
                lo = cond[0] if len(cond) > 0 else None
                hi = cond[1] if len(cond) > 1 else None
                if lo is not None:
                    df = df[df[col] >= lo]
                if hi is not None:
                    df = df[df[col] <= hi]
            else:
                df = df[df[col] == cond]

    # Ensure single independent variable
    if isinstance(ind_var, (list, tuple)):
        if len(ind_var) != 1:
            raise ValueError('Provide exactly one independent variable (univariate analysis).')
        ind_var = ind_var[0]

    # Drop missing rows on required columns
    df = df.dropna(subset=[ind_var, dep_var]).copy()

    # Extract arrays
    x = df[ind_var].to_numpy(dtype=float)
    y = df[dep_var].to_numpy(dtype=float)

    # Guard against not enough data
    if x.size < 2:
        raise ValueError('Not enough data after filtering.')

    # Define helpers
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def poly_eq_str(coefs, xname):
        # coefs: highest degree first (np.polyfit order)
        terms = []
        deg = len(coefs) - 1
        for i, a in enumerate(coefs):
            p = deg - i
            if p == 0:
                term = f'{a:.3g}'
            elif p == 1:
                term = f'{a:.3g}·{xname}'
            else:
                term = f'{a:.3g}·{xname}^{p}'
            terms.append(term)
        return ' + '.join(terms).replace('+ -', '- ')

    # Prepare x-grid for smooth curves
    xr = np.linspace(np.min(x), np.max(x), 400)

    # Collect fits
    fits = []

    # Linear: y = a1·x + a0
    c_lin = np.polyfit(x, y, 1)  # [a1, a0]
    yhat_lin = np.polyval(c_lin, x)
    r2_lin = r2(y, yhat_lin)
    fits.append({
        'name': 'Linear',
        'yfit': np.polyval(c_lin, xr),
        'label': f'Linear: y = {poly_eq_str(c_lin, ind_var)}  (R²={r2_lin:.3f})'
    })

    # Polynomial degrees 2..max_poly
    for d in range(2, max_poly + 1):
        try:
            c_poly = np.polyfit(x, y, d)  # length d+1
            yhat_poly = np.polyval(c_poly, x)
            r2_poly = r2(y, yhat_poly)
            fits.append({
                'name': f'Poly{d}',
                'yfit': np.polyval(c_poly, xr),
                'label': f'Poly{d}: y = {poly_eq_str(c_poly, ind_var)}  (R²={r2_poly:.3f})'
            })
        except Exception:
            continue

    # Log-x: y = a0 + a1·ln(x)  (x>0)
    if np.all(x > 0):
        try:
            Xlog = np.vstack([np.log(x), np.ones_like(x)]).T
            a1, a0 = np.linalg.lstsq(Xlog, y, rcond=None)[0]
            yhat_logx = a0 + a1 * np.log(x)
            r2_logx = r2(y, yhat_logx)
            fits.append({
                'name': 'Log-X',
                'yfit': a0 + a1 * np.log(xr.clip(min=np.min(x[x>0]))),
                'label': f'Log-X: y = {a0:.3g} + {a1:.3g}·ln({ind_var})  (R²={r2_logx:.3f})'
            })
        except Exception:
            pass

    # Exp-y: y = exp(a0 + a1·x)  (y>0)
    if np.all(y > 0):
        try:
            Ylog = np.log(y)
            Xlin = np.vstack([x, np.ones_like(x)]).T
            a1e, a0e = np.linalg.lstsq(Xlin, Ylog, rcond=None)[0]
            yhat_expy = np.exp(a0e + a1e * x)
            r2_expy = r2(y, yhat_expy)
            fits.append({
                'name': 'Exp-Y',
                'yfit': np.exp(a0e + a1e * xr),
                'label': f'Exp-Y: y = exp({a0e:.3g} + {a1e:.3g}·{ind_var})  (R²={r2_expy:.3f})'
            })
        except Exception:
            pass

    # Power: y = A·x^b  (x>0, y>0)
    if np.all(x > 0) and np.all(y > 0):
        try:
            Ylog = np.log(y)
            Xlog = np.vstack([np.log(x), np.ones_like(x)]).T
            b, lnA = np.linalg.lstsq(Xlog, Ylog, rcond=None)[0]
            A = np.exp(lnA)
            yhat_pow = A * (x ** b)
            r2_pow = r2(y, yhat_pow)
            fits.append({
                'name': 'Power',
                'yfit': A * (xr ** b),
                'label': f'Power: y = {A:.3g}·{ind_var}^{b:.3g}  (R²={r2_pow:.3f})'
            })
        except Exception:
            pass

    # Plot
    if plot:
        plt.figure(figsize=(7, 7))
        plt.scatter(x, y, alpha=0.6, label='Data')
        for f in fits:
            plt.plot(xr, f['yfit'], linewidth=2, label=f['label'])
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(f'Regression Fits: {dep_var} vs {ind_var}')
        plt.grid(True)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    # Summarize results
    regre = {f['name']: f['label'] for f in fits}
    return regre