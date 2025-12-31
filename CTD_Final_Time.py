# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 15:54:27 2025

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
    # Ensure the cast_index is valid (not out of range)
   if cast_index < 0 or cast_index >= len(ctdfiles):
       raise ValueError("Invalid cast_index. It must be within the range of available CTD files.")

   # Extract data for the specified cast
   filename = ctdfiles[cast_index]  # Get the filename for the specified cast
   cast_data = ReadCTDData(filename, 1)  # Read data for this cast

   # Extract relevant data
   SP = cast_data['salt']  # Salinity
   p = cast_data['pressure']  # Pressure
   lons = cast_data['longitude']  # Longitude
   lats = cast_data['latitude']  # Latitude
   t = cast_data['temperature']  # Temperature
   times = cast_data['time']  # Longitude


   average_lons=float(sum(lons)/len(lons))
   average_lats=float(sum(lats)/len(lats))
   total_time = float((times[-1])-(times[0])) 
   # Set p_ref to the deepest common pressure between the specified cast and the next cast
   p_ref = None
   
   # If there's a next cast, compare with it to find the largest common pressure
   if cast_index + 1 < len(ctdfiles):
       # Read the next cast's data
       next_cast_data = ReadCTDData(ctdfiles[cast_index + 1], 1)
       
       # Find the largest common pressure between the two casts
       p_ref = find_largest_common_pressure(cast_data, next_cast_data)
       lons_next = next_cast_data['longitude']  # Longitude
       

   return SP, p, t, p_ref, lons,lats,average_lons, average_lats, times, total_time

#takes info from pair of casts and returns geostrophic velocity (shear)
def GeostrophicVelocity(ctdfiles, cast_number, cast_number2):
    SP4, p4, t4, p_ref4, lons4, lats4, average_lons4, average_lats4, times4, total_time4 = GetCastData(ctdfiles, cast_number)
    SA4 = gsw.SA_from_SP(SP4, p4, average_lons4, average_lats4) #absolute salinity
    CT4 = gsw.CT_from_t(SA4, t4, p4) #conservative temperature
    SA4 = np.array(SA4)
    CT4 = np.array(CT4)
    p4 = np.array(p4)
    geo_strf_dyn_height_4 = gsw.geo_strf_dyn_height(SA4, CT4, p4, p_ref4) #geostrophic streamfunction(dynamic height)
    
    SP5, p5, t5, p_ref5, lons5, lats5, average_lons5, average_lats5, times5, total_time5 = GetCastData(ctdfiles, cast_number2)
    SA5 = gsw.SA_from_SP(SP5, p5, average_lons5, average_lats5)
    CT5 = gsw.CT_from_t(SA5, t5, p5)
    SA5 = np.array(SA5)
    CT5 = np.array(CT5)
    p5 = np.array(p5)
    geo_strf_dyn_height_5 = gsw.geo_strf_dyn_height(SA5, CT5, p5, p_ref4)
    
    
   
    #if max(p4) < 200 or max(p5) < 200: #dont use casts too shallow for integration with p_ref 150
        #print(f"Skipping cast pair {cast_number}-{cast_number2}: insufficient depth")
        
        
    common_depths_45 = np.intersect1d(p4, p5)
    #print("Common depths:", common_depths_45)

    if common_depths_45.size > 0:
        # Find indices in p4 and p5 where values match common_depths_45
        p4_common_indices = np.where(np.isin(p4, common_depths_45))[0]
        p5_common_indices = np.where(np.isin(p5, common_depths_45))[0]

        # Trim p and geo_strf arrays to those indices
        p4_trimmed = p4[p4_common_indices]
        p5_trimmed = p5[p5_common_indices]
        geo_strf_dyn_height_4_trimmed = geo_strf_dyn_height_4[p4_common_indices]
        geo_strf_dyn_height_5_trimmed = geo_strf_dyn_height_5[p5_common_indices]

        # Combine trimmed dynamic height values into a 2D array (each row: [cast4_value, cast5_value])
        #geo_strf_dyn_height_45_array = np.column_stack((geo_strf_dyn_height_4_trimmed, geo_strf_dyn_height_5_trimmed))
        geo_strf = np.array([geo_strf_dyn_height_4_trimmed, geo_strf_dyn_height_5_trimmed])
        
        lons = np.array([average_lons4, average_lons5])
        lats = np.array([average_lats4, average_lats5])
        #print(lons)
        geostrophic_velocity = gsw.geostrophic_velocity(geo_strf, lons, lats)
        return geostrophic_velocity,lons,lats, p4_trimmed, p_ref4, geo_strf_dyn_height_4_trimmed
    #print(f"Common depths between cast {cast_number} and {cast_number2}: min={np.min(common_depths_45)}, max={np.max(common_depths_45)}")
    return

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
    plt.plot(velocity, depth)
    #plt.gca().invert_yaxis()
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
    
p, lons, temp, salt, oxy, cast_nums, lats, times = interpData(ctdfiles)
                                         
total_dataset = create_dataset(total_dataset, lons, temp, salt, oxy, cast_nums, lats, times)

labels = [f"Cast Number: {i}" for i in range(1, len(ctdfiles) + 1)]

# Get results for consecutive pairs of casts
#results = compare_consecutive_casts(ctdfiles)
#print(results)

#quality check, comment out
SP4, p4, t4, p_ref4, lons4, lats4, average_lons4, average_lats4, times, total_time = GetCastData(ctdfiles, 1)



print(SP4)
print(p_ref4)
print(average_lons4)
print(p4)
print(t4)    
#get shear, lons, lats, depth, and p_ref for each cast   
geostrophicvel1,lons1,lats1, p4_trimmed1, p_ref41, geo_strf1=GeostrophicVelocity(ctdfiles, 0, 1)       
geostrophicvel2,lons2,lats2, p4_trimmed2, p_ref42, geo_strf2=GeostrophicVelocity(ctdfiles, 1, 2)     
geostrophicvel3,lons3,lats3, p4_trimmed3, p_ref43, geo_strf3=GeostrophicVelocity(ctdfiles, 2, 3)     
geostrophicvel4,lons4,lats4, p4_trimmed4, p_ref44, geo_strf4=GeostrophicVelocity(ctdfiles, 3, 4)
geostrophicvel5,lons5,lats5, p4_trimmed5, p_ref45, geo_strf5=GeostrophicVelocity(ctdfiles, 4, 5)  
geostrophicvel6,lons6,lats6, p4_trimmed6, p_ref46, geo_strf6=GeostrophicVelocity(ctdfiles, 5, 6)
geostrophicvel7,lons7,lats7, p4_trimmed7, p_ref47, geo_strf7=GeostrophicVelocity(ctdfiles, 6, 7)
geostrophicvel8,lons8,lats8, p4_trimmed8, p_ref48, geo_strf8=GeostrophicVelocity(ctdfiles, 7, 8)             

#print(np.min(p4_trimmed1))
#print(np.max(p4_trimmed1))

#just take geostrophic shear from above results
geo1 = (geostrophicvel1[0])
geo2 = (geostrophicvel2[0])
geo3 = (geostrophicvel3[0])
geo4 = (geostrophicvel4[0])
geo5 = (geostrophicvel5[0])
geo6 = (geostrophicvel6[0])
geo7 = (geostrophicvel7[0])
geo8 = (geostrophicvel8[0])

#print(geo1.shape) #499
#print(geo6)

#just take lons from above results
lons1 = (geostrophicvel1[1])
lons2 = (geostrophicvel2[1])
lons3 = (geostrophicvel3[1])
lons4 = (geostrophicvel4[1])
lons5 = (geostrophicvel5[1])
lons6 = (geostrophicvel6[1])
lons7 = (geostrophicvel7[1])
lons8 = (geostrophicvel8[1])

print(lons1)

#make list of lons for plotting
lons = []
lons.append(lons1[0])
lons.append(lons2[0])
lons.append(lons3[0])
lons.append(lons4[0])
lons.append(lons5[0])
lons.append(lons6[0])
lons.append(lons7[0])
lons.append(lons8[0])

# Define each set of geostrophic velocity data and metadata for plotting
geos = [geo1, geo2, geo3, geo4, geo5, geo6, geo7, geo8]
geo_strfs = [geo_strf1, geo_strf2, geo_strf3, geo_strf4, geo_strf5, geo_strf6, geo_strf7, geo_strf8]
depths_trimmed = [p4_trimmed1, p4_trimmed2, p4_trimmed3, p4_trimmed4,
                  p4_trimmed5, p4_trimmed6, p4_trimmed7, p4_trimmed8]
lons_pairs = [lons1, lons2, lons3, lons4, lons5, lons6, lons7, lons8]

# Plot each geostrophic velocity profile as its own figure
for i in range(len(geos)):
    Z = geos[i]                # (N,) geostrophic velocity values
    Y = depths_trimmed[i]      # (N,) pressure
    lon = lons_pairs[i][0]     # single longitude

    PlotVelocityProfile(Z, Y, i+1, lon)



lon_midpoints = []
for pair_lons in lons_pairs:
    lon_midpoints.append(np.mean(pair_lons))

# Step 2: Define a common pressure grid (depth axis)
common_p = np.arange(0, 701, 1)  # adjust to your data's max depth

# Step 3: Interpolate all geostrophic velocity profiles onto the common pressure grid
geo_matrix = np.full((len(common_p), len(geos)), np.nan)

for i, (vel, depth) in enumerate(zip(geos, depths_trimmed)):
    # Ensure velocity and depth are 1-D numpy arrays
    vel = np.array(vel).flatten()
    depth = np.array(depth).flatten()

# Mask out NaNs safely
    mask = np.isfinite(vel) & np.isfinite(depth)

    if np.sum(mask) > 3:
        geo_interp = np.interp(common_p, depth[mask], vel[mask], left=np.nan, right=np.nan)
        geo_matrix[:, i] = geo_interp


# Step 4: Create meshgrid for plotting (X = lon, Y = depth)
LON, DEPTH = np.meshgrid(lon_midpoints, common_p)

# Step 5: Plot using your existing ContourPlot() function
ContourPlot(
    X=LON,
    Y=DEPTH,
    Z=geo_matrix,
    cmap=cmo.cm.balance,  # balanced for +/− velocities
    xlabel='Longitude (°E)',
    ylabel='Pressure (dbar)',
    title='Geostrophic Velocity Cross Section',
    cbarlabel='Geostrophic Velocity (m/s)'
)

# Step 3B: Interpolate dynamic height profiles onto the common pressure grid
geo_strf_matrix = np.full((len(common_p), len(geo_strfs)), np.nan)

for i, (dh, depth) in enumerate(zip(geo_strfs, depths_trimmed)):

    dh = np.array(dh).flatten()
    depth = np.array(depth).flatten()

    mask = np.isfinite(dh) & np.isfinite(depth)

    if np.sum(mask) > 3:
        geo_strf_matrix[:, i] = np.interp(common_p, depth[mask], dh[mask], 
                                          left=np.nan, right=np.nan)

ContourPlot(
    X=LON,
    Y=DEPTH,
    Z=geo_strf_matrix,
    cmap=cmo.cm.balance,
    xlabel='Longitude (°E)',
    ylabel='Pressure (dbar)',
    title='Dynamic Height Cross Section',
    cbarlabel='Dynamic Height (m²/s²)'
)
