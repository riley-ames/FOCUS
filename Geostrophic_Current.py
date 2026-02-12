# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:01:34 2026

@author: ripti
"""

#----------------------------------------------------------------ABANDON ALL HOPE, YE WHO ENTER HERE----------------------------------------------------------------

#import packages
import math, os, gsw, glob, re, sys, warnings, calendar
import numpy as np
import matplotlib.pyplot as plt 
import cmocean as cmo
import pandas as pd
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import integrate
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter
from datetime import datetime
from scipy.integrate import cumulative_trapezoid


# Read CTD data from a file 'filename' and return it to a dictionary
def ReadCTDData(filename, nheaders):
    with open(filename, 'r') as fobjs:
        linelist = fobjs.readlines()
    
    data_lines = linelist[nheaders:]
    ndata = len(data_lines)

    pres, temp, salt, oxy, lon, lat, time = ndata * [0.0], ndata * [0.0], ndata * [0.0], ndata * [0.0], ndata * [0.0], ndata * [0.0], ndata * [0.0]
    
    for i, line in enumerate(data_lines):
        words = line.strip().split(',')  # Split columns by comma
        
        # Extract data and check for 'no data' codes in CTD data
        #no data in ctd shown as -9.990e-29
        try:
            pres[i] = float(words[4]) if float(words[4]) not in [0, -9.990e-29] else None
            temp[i] = float(words[7]) if float(words[7]) not in [0, -9.990e-29] else None
            salt[i] = float(words[17]) if float(words[17]) not in [0, -9.990e-29] else None
            oxy[i] = float(words[19]) if float(words[19]) not in [0, -9.990e-29] else None  # ml/l
            lon[i] = float(words[2]) if float(words[2]) not in [0, -9.990e-29] else None
            lat[i] = float(words[1]) if float(words[1]) not in [0, -9.990e-29] else None
            time[i] = float(words[3]) if float(words[3]) not in [0, -9.990e-29] else None

        except (ValueError, IndexError) as e:
            print(f"Error processing line {i + nheaders}: {line.strip()} - {e}")
    
    # Exclude points where we have no data from the cast
    cast = {
        'pressure': [value for value in pres if value is not None],
        'salt': [value for value in salt if value is not None],
        'temperature': [value for value in temp if value is not None],
        'oxygen': [value for value in oxy if value is not None],
        'longitude': [value for value in lon if value is not None],
        'latitude': [value for value in lat if value is not None],
        'time': [value for value in time if value is not None]
    }

    # Labels extraction from the header (first line of the file)
    labels = linelist[0].strip().split(',')
    for label in labels:
        if label not in cast:
            cast[label] = None  # Add header keys to dictionary with initial None values

    return cast

#process multiple CTD files, sorts them, reads data, and interpolates onto pressure grid
def interpData(ctdfiles):
    lons = []
    lats = []
    times = []
    p = np.arange(1, 701)
    sigma0_grid = np.arange(22, 28, 0.0075)
    tlev = np.arange(5, 31, 5)
    temp = np.empty((700, 9))
    salt = np.empty((700, 9))
    oxy = np.empty((700, 9))
    cast_nums = []
    
    ctdfiles_sorted = sorted(ctdfiles, key=lambda x: ReadCTDData(x, 1).get('cast', 0))
            
    for i, file in enumerate(ctdfiles_sorted):
        filename = file
        
        cast = ReadCTDData(filename, 1)

        try: #gets longitude and latitude values from each cast
            lon = cast.get("longitude")
            lat = cast.get("latitude")
            if lon is not None:
                lon = [value if value not in [-9.990e-29, 0] else None for value in lon]
            if lat is not None:
                lat = [value if value not in [-9.990e-29, 0] else None for value in lat]
        except TypeError:
        # Handle invalid data by setting None
            lon, lat = None, None

    #add longitudes and latitudes to a list
        lons.append(lon)
        lats.append(lat)

    for i, file in enumerate(ctdfiles_sorted):
        filename = file
        
        cast = ReadCTDData(filename, 1)

        try: #gets longitude and latitude values from each cast
            time = cast.get("time")
            if time is not None:
                time = [value if value not in [-9.990e-29, 0] else None for value in time]
        except TypeError:
        # Handle invalid data by setting None
            time = None
      
        times.append(time)

        cast_num = cast.get('cast')
        cast_nums.append(cast_num)

        cast_pres = cast.get("pressure")
        cast_temp = cast.get("temperature")
        cast_salt = cast.get("salt")
        cast_oxy = cast.get("oxygen")

        temp[:,i] = np.interp(p, cast_pres, cast_temp, left=np.nan, right=np.nan)
        salt[:,i] = np.interp(p, cast_pres, cast_salt, left=np.nan, right=np.nan)
        oxy[:,i] = np.interp(p, cast_pres, cast_oxy, left=np.nan, right=np.nan)
     
    return p, lons, temp, salt, oxy, cast_nums, lats, times


#add interpolated data to dataset dictionary
def create_dataset(dataset, lons, temp, salt, oxy, cast_nums, lats, times):
    
    dataset['lons'].append(lons)
    dataset['temperature'].append(temp)
    dataset['salinity'].append(salt)
    dataset['oxygen'].append(oxy)
    dataset['cast_nums'].append(cast_nums)
    dataset['lats'].append(lats)
    dataset['times'].append(times)

    return dataset

#find largest common pressure between sets of casts for use later as p_ref
def find_largest_common_pressure(cast1, cast2):
    # Extract pressure values from the two casts
    pres1 = set(cast1.get('pressure', []))  # Use set to remove duplicates and make comparison easier
    pres2 = set(cast2.get('pressure', []))
    
    # Find the common pressures between both casts
    common_pressures = pres1.intersection(pres2)
    
    # If there are no common pressures, return None
    if not common_pressures:
        return None
    
    # Return the largest common pressure
    return max(common_pressures)


#gets data from one specific cast for use in gsw calculations
def GetCastData(ctdfiles, cast_index):
    filename = ctdfiles[cast_index]
    cast_data = ReadCTDData(filename, 1)

    SP = np.asarray(cast_data['salt'])
    p  = np.asarray(cast_data['pressure'])
    t  = np.asarray(cast_data['temperature'])

    lons_list = [x for x in cast_data['longitude'] if x is not None]
    lats_list = [x for x in cast_data['latitude'] if x is not None]
    times_list = [x for x in cast_data['time'] if x is not None]

    if not lons_list or not lats_list or not times_list:
        raise ValueError(f"Cast {cast_index} has no valid lon/lat/time data")

    average_lons = float(np.mean(lons_list))
    average_lats = float(np.mean(lats_list))
    total_time = float(times_list[-1] - times_list[0])

    # p_ref
    p_ref = None
    if cast_index + 1 < len(ctdfiles):
        next_cast_data = ReadCTDData(ctdfiles[cast_index + 1], 1)
        p_ref = find_largest_common_pressure(cast_data, next_cast_data)

    return SP, p, t, p_ref, lons_list, lats_list, average_lons, average_lats, times_list, total_time


def get_cast_metadata(ctdfiles):
    #extract data needed for geostrophy (deepest pressure, longitude, and latitude)

    n_casts = len(ctdfiles)

    p_deep = np.full((n_casts, 1), np.nan)
    lon_cast = np.full((n_casts, 1), np.nan)
    lat_cast = np.full((n_casts, 1), np.nan)

    for i in range(n_casts):
        _, p, _, _, lon_list, lat_list, avg_lon, avg_lat, _, _ = GetCastData(ctdfiles, i)

        p_deep[i, 0] = np.nanmax(p)
        lon_cast[i, 0] = avg_lon
        lat_cast[i, 0] = avg_lat

    return p_deep, lon_cast, lat_cast


def GeostrophicStream_all_casts_array(ctdfiles, n_levels=700, p_min=1):
    # M = # depth levels, N = # stations 
    # use common pressure grid n=700
    
    n_casts = len(ctdfiles)
    cast_depths = []

    # max depth across all casts
    max_depth = 0
    for i in range(n_casts):
        _, p, _, _, _, _, _, _, _, _ = GetCastData(ctdfiles, i)
        max_depth = max(max_depth, np.nanmax(p))
        cast_depths.append(p)

    # common pressure grid
    p_common = np.linspace(p_min, max_depth, n_levels)

    # allocate arrays
    geo_strf_all = np.full((n_levels, n_casts), np.nan)
    p_ref = np.full(n_casts, np.nan)
    SA_all = np.full((n_levels, n_casts), np.nan)
    CT_all = np.full((n_levels, n_casts), np.nan)

    # loop over casts
    for i in range(n_casts):
        SP, p, t, _, _, _, avg_lon, avg_lat, _, _ = GetCastData(ctdfiles, i)

        # Only valid data
        valid = np.isfinite(p) & np.isfinite(SP) & np.isfinite(t)
        if np.sum(valid) < 3:
            continue

        # sort by pressure
        sort_idx = np.argsort(p[valid])
        p_valid = p[valid][sort_idx]
        SP_valid = SP[valid][sort_idx]
        t_valid = t[valid][sort_idx]

        # compute SA and CT
        SA = gsw.SA_from_SP(SP_valid, p_valid, avg_lon, avg_lat)
        CT = gsw.CT_from_t(SA, t_valid, p_valid)

        # determine p_ref (deepest common level between pairs)
        if i < n_casts - 1:
            _, p_next, _, _, _, _, _, _, _, _ = GetCastData(ctdfiles, i + 1)
            candidate_ref = min(np.nanmax(p_valid), np.nanmax(p_next))
        else:
            candidate_ref = np.nanmax(p_valid)

        # clip to cast range
        candidate_ref = np.clip(candidate_ref, np.nanmin(p_valid), np.nanmax(p_valid))
        p_ref[i] = candidate_ref
        SA_i = gsw.SA_from_SP(SP_valid, p_valid, avg_lon, avg_lat)
        CT_i = gsw.CT_from_t(SA_i, t_valid, p_valid)
        # compute dynamic height
        dyn_height = gsw.geo_strf_dyn_height(SA, CT, p_valid, candidate_ref)

        # interpolate dyn_height onto common grid, only within cast depth
        in_range = (p_common >= np.nanmin(p_valid)) & (p_common <= np.nanmax(p_valid))
        geo_strf_all[in_range, i] = np.interp(p_common[in_range], p_valid, dyn_height)
        
        SA_all[in_range, i] = np.interp(p_common[in_range], p_valid, SA_i)
        CT_all[in_range, i] = np.interp(p_common[in_range], p_valid, CT_i)
        
    return SA_all, CT_all, geo_strf_all, p_ref, p_common, cast_depths






# set up ctdfiles
foldername = 'Downcast'
ctdfiles = [os.path.join(foldername, f) for f in os.listdir(foldername) if f.endswith('.asc')]
ctdfiles.sort()  # optional: sort files if needed

# prepare total dataset
total_dataset = {'lons': [], 'temperature': [], 'salinity': [], 'oxygen': [],
                 'cast_nums': [], 'pressure': [], 'lats':[], 'times':[]}

# interpolate and create dataset
p, lons, temp, salt, oxy, cast_nums, lats, times = interpData(ctdfiles)
total_dataset = create_dataset(total_dataset, lons, temp, salt, oxy, cast_nums, lats, times)

labels = [f"Cast Number: {i}" for i in range(1, len(ctdfiles) + 1)]

#get metadata (can print for quality check)
p_deep, lon_cast, lat_cast = get_cast_metadata(ctdfiles)

# compute dynamic height and SA and CT
SA, CT, geo_strf_all, p_ref, p_common, cast_depth = GeostrophicStream_all_casts_array(ctdfiles, n_levels=700, p_min=1)

# number of casts
n_casts = len(ctdfiles)

# initialize 1D arrays for average longitudes and latitudes (for gsw.geostrophic_velocity)
avg_lons = np.full(n_casts, np.nan)
avg_lats = np.full(n_casts, np.nan)

# loop through each cast and fill averages
for i in range(n_casts):
    _, _, _, _, lon_list, lat_list, avg_lon, avg_lat, _, _ = GetCastData(ctdfiles, i)
    avg_lons[i] = avg_lon
    avg_lats[i] = avg_lat

#quality check lons and lats to make sure shape is correct
print("avg_lons shape:", avg_lons.shape)  # (N,)
print("avg_lats shape:", avg_lats.shape)  # (N,)

#check shape of geo_strf
print(geo_strf_all)

# quality check: count number of NaNs in geo_strf_all
num_nans = np.sum(np.isnan(geo_strf_all))
print(f"Number of NaNs in geo_strf_all: {num_nans}")
total_elements = geo_strf_all.size
nan_fraction = num_nans / total_elements
print(f"Fraction of NaNs: {nan_fraction:.2%}")

#plot CT
fig, ax = plt.subplots(figsize=(10, 6))
pcm = ax.pcolormesh(
    avg_lons,
    p_common,
    CT,
    cmap=cmo.cm.thermal,
    shading='nearest'
)
ax.invert_yaxis()
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Pressure (dbar)')
ax.set_title('CT Cross Section')
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('CT (C)')
plt.tight_layout()
plt.show()

#plot AS
fig, ax = plt.subplots(figsize=(10, 6))
pcm = ax.pcolormesh(
    avg_lons,
    p_common,
    SA,
    cmap=cmo.cm.haline,
    shading='nearest'
)
ax.invert_yaxis()
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Pressure (dbar)')
ax.set_title('AS Cross Section')
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('AS (psu)')
plt.tight_layout()
plt.show()


#plot dynamic height
fig, ax = plt.subplots(figsize=(10, 6))
pcm = ax.pcolormesh(
    avg_lons,
    p_common,
    geo_strf_all,
    cmap=cmo.cm.balance,
    shading='nearest'
)
ax.invert_yaxis()
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Pressure (dbar)')
ax.set_title('Geostrophic Stream Cross Section')
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Geostrophic Stream (m)')
plt.tight_layout()
plt.show()

#OLD WAY, MxN array
#geovel, lon, lat = gsw.geostrophic_velocity(geo_strf_all, avg_lons, avg_lats)


#NEW WAY, do it pairwise, see if it changes, print everything for checking

n_levels, n_casts = geo_strf_all.shape

# allocate velocity array (between casts)
geo_vel_pair = np.full((n_levels, n_casts - 1), np.nan)

# midpoint lon/lat for each cast pair
#that's fine

lon_mid = 0.5 * (avg_lons[:-1] + avg_lons[1:])
lat_mid = 0.5 * (avg_lats[:-1] + avg_lats[1:])

for i in range(n_casts - 1):
    #check
    print("\n-------------------------------------------")
    print(f"Cast Pair: {i}  &  {i+1}")
    print(f"Lon Pair: {avg_lons[i]:.4f}, {avg_lons[i+1]:.4f}")
    print(f"Lat Pair: {avg_lats[i]:.4f}, {avg_lats[i+1]:.4f}")

    # dynamic height for the cast pair (M x 2)
    dyn_pair = geo_strf_all[:, i:i+2]

    # lon/lat for the cast pair
    lon_pair = np.array([avg_lons[i], avg_lons[i + 1]])
    lat_pair = np.array([avg_lats[i], avg_lats[i + 1]])

    # count valid depth levels
    valid_levels = np.sum(np.isfinite(dyn_pair).all(axis=1))
    print(f"Valid overlapping depth levels: {valid_levels}")

    # compute geostrophic velocity
    v_pair, _, _ = gsw.geostrophic_velocity(
        dyn_pair,
        lon_pair,
        lat_pair
    )

    #check
    print(f"Velocity shape: {v_pair.shape}")
    print(f"Min velocity: {np.nanmin(v_pair):.5f} m/s")
    print(f"Max velocity: {np.nanmax(v_pair):.5f} m/s")
    print(f"Mean velocity: {np.nanmean(v_pair):.5f} m/s")

    # OPTIONAL: print full vertical profile
    print("Velocity profile (first 20 levels):")
    print(v_pair[:20, 0])

    # store result
    geo_vel_pair[:, i] = v_pair[:, 0]


print("geo_vel_pair shape:", geo_vel_pair.shape)
# (n_levels, n_casts - 1)

fig, ax = plt.subplots(figsize=(10, 6))
pcm = ax.pcolormesh(
    lon_mid,
    p_common,
    geo_vel_pair,
    cmap=cmo.cm.balance,
    shading='nearest'
)
ax.invert_yaxis()
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Pressure (dbar)')
ax.set_title('Pairwise Geostrophic Velocity (GSW)')
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Geostrophic Velocity (m/s)')
plt.tight_layout()
plt.show()