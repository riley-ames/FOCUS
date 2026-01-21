# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:10:22 2026

@author: ripti
"""


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

#finds p_ref between each pair of casts
def compare_consecutive_casts(ctdfiles):
    results = []
    # Read CTD data from all the files
    casts = [ReadCTDData(file, 1) for file in ctdfiles]
    
    # Compare consecutive casts (Cast 1 with Cast 2, Cast 2 with Cast 3, etc.)
    for i in range(len(casts) - 1):  # Loop until second-to-last cast
        cast1 = casts[i]
        cast2 = casts[i + 1]
        
        # Find the largest common pressure between these two casts
        largest_common_pressure = find_largest_common_pressure(cast1, cast2)
        
        if largest_common_pressure is not None:
            # Save the result: cast indices and largest common pressure
            results.append((i + 1, i + 2, largest_common_pressure))  # +1 for human-readable cast numbers
    
    return results

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



def GeostrophicStream(ctdfiles, cast_index, p_ref=None):
    
    #compute dynamic height (geostrophic streamfunction) for a single cast.

    # get cast data
    SP, p, t, _, lon_list, lat_list, avg_lon, avg_lat, _, _ = GetCastData(ctdfiles, cast_index)

    # convert to Absolute Salinity and Conservative Temperature
    SA = gsw.SA_from_SP(SP, p, avg_lon, avg_lat)
    CT = gsw.CT_from_t(SA, t, p)

    # set reference pressure for dynamic height
    if p_ref is None:
        p_ref = np.max(p)

    geo_strf = gsw.geo_strf_dyn_height(SA, CT, p, p_ref)

    return geo_strf, avg_lon, avg_lat, p


def GeostrophicVelocity(ctdfiles, cast_index1, cast_index2, p_common=None):
    
    #compute geostrophic velocity between two casts
    
    #get dynamic heights for both casts
    geo1, lon1, lat1, p1 = GeostrophicStream(ctdfiles, cast_index1)
    geo2, lon2, lat2, p2 = GeostrophicStream(ctdfiles, cast_index2)

    #find common pressure grid
    if p_common is None:
        min_p = max(np.min(p1), np.min(p2))
        max_p = min(np.max(p1), np.max(p2))
        p_common = np.linspace(min_p, max_p, 500)

    #interpolate dynamic heights onto common pressure grid
    geo1_interp = np.interp(p_common, p1, geo1)
    geo2_interp = np.interp(p_common, p2, geo2)

    #lats constant, use actual longitude difference
    lons = np.array([lon1, lon2])
    lats = np.array([np.mean(lat1), np.mean(lat1)])  # latitude constant
    geo_velocity = gsw.geostrophic_velocity(np.vstack([geo1_interp, geo2_interp]), lons, lats)

    return geo_velocity, p_common




#visualize data in contour plot
def ContourPlot(X, Y, Z, cmap, xlabel, ylabel, title, cbarlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(X, Y, Z, levels=120, cmap=cmap)
    #ax.invert_yaxis()  # Depth increasing downward
    #ax.set_ylim(Y.max(), Y.min())
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical')
    cbar.set_label(cbarlabel)
    plt.tight_layout()
    plt.show()

def PlotVelocityProfile(velocity, depth, cast_pair_num, lon):
    plt.figure(figsize=(6, 8))
    plt.plot(velocity, depth)       # x = velocity, y = depth
    plt.gca().invert_yaxis()        # depth increases downward
    plt.xlabel('Geostrophic Velocity (m/s)')
    plt.ylabel('Pressure (dbar)')
    plt.title(f'Geostrophic Velocity Profile\nCast {cast_pair_num} at Lon {lon:.2f}°')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



#set up arrays, process CTD files, create final dataset for plotting    
foldername = 'Downcast'

temps = np.arange(4, 32, 0.5)
salts = np.arange(33, 38, 0.2)
T, S = np.meshgrid(temps, salts)

total_dataset = {'lons': [], 'temperature': [], 'salinity': [], 'oxygen': [], 'cast_nums': [], 'pressure': [], 'lats':[], 'times':[]}

ctdfiles = []
ctdfiles_sorted = []

files = os.listdir(foldername)
for filename in files:
    if filename.endswith('.asc'):
        filepath = os.path.join(foldername, filename)
        ctdfiles.append(filepath)

#reverse the order so that the last cast becomes first, not used
#ctdfiles = ctdfiles[::-1]

    
p, lons, temp, salt, oxy, cast_nums, lats, times = interpData(ctdfiles)
                                         
total_dataset = create_dataset(total_dataset, lons, temp, salt, oxy, cast_nums, lats, times)

labels = [f"Cast Number: {i}" for i in range(1, len(ctdfiles) + 1)]

# Get results for consecutive pairs of casts
#results = compare_consecutive_casts(ctdfiles)
#print(results)

'''
#quality check, comment out

SP4, p4, t4, p_ref4, lons4, lats4, average_lons4, average_lats4, times, total_time = GetCastData(ctdfiles, 8)

SP1, p1, t1, p_ref1, lons1, lats1, avg_lon1, avg_lat1, times1, _ = GetCastData(ctdfiles, 0)
SP2, p2, t2, p_ref2, lons2, lats2, avg_lon2, avg_lat2, times2, _ = GetCastData(ctdfiles, 1)

print("p_ref1:", p_ref1)
print("pressure range cast1:", (min(p1), max(p1)))
print("pressure range cast2:", (min(p2), max(p2)))

print(SP4)
print(p_ref4)
print(average_lons4)
print(p4)
print(t4) 
'''

geostrf1, lons1, lats1, p1 = GeostrophicStream(ctdfiles, 0)
geostrf2, lons2, lats2, p2 = GeostrophicStream(ctdfiles, 1)
geostrf3,lons3,lats3, p3 = GeostrophicStream(ctdfiles, 2)
geostrf4,lons4,lats4, p4 = GeostrophicStream(ctdfiles, 3)
geostrf5,lons5,lats5, p5 = GeostrophicStream(ctdfiles, 4)
geostrf6,lons6,lats6, p6 = GeostrophicStream(ctdfiles, 5)
geostrf7,lons7,lats7, p7 = GeostrophicStream(ctdfiles, 6)
geostrf8,lons8,lats8, p8 = GeostrophicStream(ctdfiles, 7)
geostrf9,lons9,lats9, p9 = GeostrophicStream(ctdfiles, 8)

geovel1, depth1 = GeostrophicVelocity(ctdfiles, 0, 1)
geovel2, depth2 = GeostrophicVelocity(ctdfiles, 1, 2)
geovel3, depth3 = GeostrophicVelocity(ctdfiles, 2, 3)
geovel4, depth4 = GeostrophicVelocity(ctdfiles, 3, 4)
geovel5, depth5 = GeostrophicVelocity(ctdfiles, 4, 5)
geovel6, depth6 = GeostrophicVelocity(ctdfiles, 5, 6)
geovel7, depth7 = GeostrophicVelocity(ctdfiles, 6, 7)
geovel8, depth8 = GeostrophicVelocity(ctdfiles, 7, 8)

geo1 = geovel1[0]
geo2 = geovel2[0]
geo3 = geovel3[0]
geo4 = geovel4[0]
geo5 = geovel5[0]
geo6 = geovel6[0]
geo7 = geovel7[0]
geo8 = geovel8[0]

lon1 = geovel1[1]
lon2 = geovel2[1]
lon3 = geovel3[1]
lon4 = geovel4[1]
lon5 = geovel5[1]
lon6 = geovel6[1]
lon7 = geovel7[1]
lon8 = geovel8[1]


#make list of lons for plotting
lons = []
lons.append(lons1)
lons.append(lons2)
lons.append(lons3)
lons.append(lons4)
lons.append(lons5)
lons.append(lons6)
lons.append(lons7)
lons.append(lons8)

geo_strfs = [geostrf1, geostrf2, geostrf3, geostrf4, geostrf5, geostrf6, geostrf7, geostrf8, geostrf9]
depths = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
lons = [lons1, lons2, lons3, lons4, lons5, lons6, lons7, lons8, lons9]

'''
#quality check
print(lons)
print(depths)
print(geostrf1)
'''

# define common pressure grid
min_depth = min([np.min(d) for d in depths])
max_depth = max([np.max(d) for d in depths])
common_p = np.linspace(min_depth, max_depth, 500)  # 500 points vertical

#longitude array for plotting, one per cast
LON = np.array([lons])  # pick first lon of each pair


#interpolate dynamic height profiles onto same common pressure grid
geo_strf_matrix = np.full((len(common_p), len(geo_strfs)), np.nan)

for i, (dh, depth) in enumerate(zip(geo_strfs, depths)):
    dh = np.array(dh).flatten()
    depth = np.array(depth).flatten()
    
    mask = np.isfinite(dh) & np.isfinite(depth)
    if np.sum(mask) > 3:
        geo_strf_matrix[:, i] = np.interp(common_p, depth[mask], dh[mask], left=np.nan, right=np.nan)

#compute average longitude per cast
lons_avg = lons

#contour plot
ContourPlot(
    X=lons_avg,
    Y=common_p,
    Z=geo_strf_matrix,
    cmap=cmo.cm.balance,
    xlabel='Longitude (°E)',
    ylabel='Pressure (dbar)',
    title='Dynamic Height Cross Section',
    cbarlabel='Dynamic Height (m²/s²)'
)



#same as above but for geostrophic velocity

geos = [geo1, geo2, geo3, geo4, geo5, geo6, geo7, geo8]
lon = [lon1, lon2, lon3, lon4, lon5, lon6, lon7, lon8]
depth = [depth1, depth2, depth3, depth4, depth5, depth6, depth7, depth8]
lon = [float(x) for x in lon]

#print(lon)
min_depth = min([np.min(d) for d in depth])
max_depth = max([np.max(d) for d in depth])
common_p = np.linspace(min_depth, max_depth, 500)

geo_vel_matrix = np.full((len(common_p), len(geos)), np.nan)

for i, (vel, dep) in enumerate(zip(geos, depth)):
    vel = np.asarray(vel).flatten()
    dep = np.asarray(dep).flatten()
    min_len = min(len(vel), len(dep))
    vel, dep = vel[:min_len], dep[:min_len]

    mask = np.isfinite(vel) & np.isfinite(dep)
    if np.sum(mask) > 3:
        sorted_idx = np.argsort(dep[mask])
        geo_vel_matrix[:, i] = np.interp(
            common_p,
            dep[mask][sorted_idx],
            vel[mask][sorted_idx],
            left=np.nan,
            right=np.nan
        )

ContourPlot(
    X=np.array(lon),
    Y=common_p,
    Z=geo_vel_matrix,
    cmap=cmo.cm.balance,
    xlabel='Longitude (°E)',
    ylabel='Pressure (dbar)',
    title='Geostrophic Velocity Cross Section',
    cbarlabel='Velocity (m/s)'
)
