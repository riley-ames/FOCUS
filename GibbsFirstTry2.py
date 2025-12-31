# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:55:20 2025

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

    pres, temp, salt, oxy, lon, lat = ndata * [0.0], ndata * [0.0], ndata * [0.0], ndata * [0.0], ndata * [0.0], ndata * [0.0]
    
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
        except (ValueError, IndexError) as e:
            print(f"Error processing line {i + nheaders}: {line.strip()} - {e}")
    
    # Exclude points where we have no data from the cast
    cast = {
        'pressure': [value for value in pres if value is not None],
        'salt': [value for value in salt if value is not None],
        'temperature': [value for value in temp if value is not None],
        'oxygen': [value for value in oxy if value is not None],
        'longitude': [value for value in lon if value is not None],
        'latitude': [value for value in lat if value is not None]
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

      
        cast_num = cast.get('cast')
        cast_nums.append(cast_num)

        cast_pres = cast.get("pressure")
        cast_temp = cast.get("temperature")
        cast_salt = cast.get("salt")
        cast_oxy = cast.get("oxygen")
     

        temp[:,i] = np.interp(p, cast_pres, cast_temp, left=np.nan, right=np.nan)
        salt[:,i] = np.interp(p, cast_pres, cast_salt, left=np.nan, right=np.nan)
        oxy[:,i] = np.interp(p, cast_pres, cast_oxy, left=np.nan, right=np.nan)
     

    return p, lons, temp, salt, oxy, cast_nums, lats


#add interpolated data to dataset dictionary
def create_dataset(dataset, lons, temp, salt, oxy, cast_nums, lats):
    
    dataset['lons'].append(lons)
    dataset['temperature'].append(temp)
    dataset['salinity'].append(salt)
    dataset['oxygen'].append(oxy)
    dataset['cast_nums'].append(cast_nums)
    dataset['lats'].append(lats)
    
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

   average_lons=float(sum(lons)/len(lons))
   average_lats=float(sum(lats)/len(lats))
    
   # Set p_ref to the deepest common pressure between the specified cast and the next cast
   p_ref = None
   
   # If there's a next cast, compare with it to find the largest common pressure
   if cast_index + 1 < len(ctdfiles):
       # Read the next cast's data
       next_cast_data = ReadCTDData(ctdfiles[cast_index + 1], 1)
       
       # Find the largest common pressure between the two casts
       p_ref = find_largest_common_pressure(cast_data, next_cast_data)
       lons_next = next_cast_data['longitude']  # Longitude
       

   return SP, p, t, p_ref, lons,lats,average_lons, average_lats

#takes info from pair of casts and returns geostrophic velocity (shear)
def GeostrophicVelocity(ctdfiles, cast_number, cast_number2):
    SP4, p4, t4, p_ref4, lons4, lats4, average_lons4, average_lats4 = GetCastData(ctdfiles, cast_number)
    SA4 = gsw.SA_from_SP(SP4, p4, average_lons4, average_lats4) #absolute salinity
    CT4 = gsw.CT_from_t(SA4, t4, p4) #conservative temperature
    SA4 = np.array(SA4)
    CT4 = np.array(CT4)
    p4 = np.array(p4)
    geo_strf_dyn_height_4 = gsw.geo_strf_dyn_height(SA4, CT4, p4, p_ref4) #geostrophic streamfunction(dynamic height)
    
    SP5, p5, t5, p_ref5, lons5, lats5, average_lons5, average_lats5 = GetCastData(ctdfiles, cast_number2)
    SA5 = gsw.SA_from_SP(SP5, p5, average_lons5, average_lats5)
    CT5 = gsw.CT_from_t(SA5, t5, p5)
    SA5 = np.array(SA5)
    CT5 = np.array(CT5)
    p5 = np.array(p5)
    geo_strf_dyn_height_5 = gsw.geo_strf_dyn_height(SA5, CT5, p5, p_ref4)
   
    if max(p4) < 200 or max(p5) < 200: #dont use casts too shallow for integration with p_ref 150
        print(f"Skipping cast pair {cast_number}-{cast_number2}: insufficient depth")

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
        geo_strf_dyn_height_45_array = np.column_stack((geo_strf_dyn_height_4_trimmed, geo_strf_dyn_height_5_trimmed))

        lons = np.array([average_lons4, average_lons5])
        lats = np.array([average_lats4, average_lats4])
        #print(lons)
        geostrophic_velocity = gsw.geostrophic_velocity(geo_strf_dyn_height_45_array, lons, lats)
        return geostrophic_velocity,lons,lats, p4_trimmed, p_ref4
    #print(f"Common depths between cast {cast_number} and {cast_number2}: min={np.min(common_depths_45)}, max={np.max(common_depths_45)}")
    return

#visualize data in contour plot
def ContourPlot(X, Y, Z, cmap, xlabel, ylabel, title, cbarlabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(X, Y, Z, levels=120, cmap=cmap)
    ax.invert_yaxis()  # Depth increasing downward
    #ax.set_ylim(Y.max(), Y.min())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical')
    cbar.set_label(cbarlabel)
    plt.tight_layout()
    plt.show()


#set up arrays, process CTD files, create final dataset for plotting    
foldername = 'Downcast'

temps = np.arange(4, 32, 0.5)
salts = np.arange(33, 38, 0.2)
T, S = np.meshgrid(temps, salts)

total_dataset = {'lons': [], 'temperature': [], 'salinity': [], 'oxygen': [], 'cast_nums': [], 'pressure': [], 'lats':[]}

ctdfiles = []
ctdfiles_sorted = []

files = os.listdir(foldername)
for filename in files:

    if filename.endswith('.asc'):
        filepath = os.path.join(foldername, filename)
        ctdfiles.append(filepath)
    
p, lons, temp, salt, oxy, cast_nums, lats = interpData(ctdfiles)
                                         
total_dataset = create_dataset(total_dataset, lons, temp, salt, oxy, cast_nums, lats)

labels = [f"Cast Number: {i}" for i in range(1, len(ctdfiles) + 1)]

# Get results for consecutive pairs of casts
results = compare_consecutive_casts(ctdfiles)
print(results)

#quality check, comment out
SP4, p4, t4, p_ref4, lons4, lats4, average_lons4, average_lats4 = GetCastData(ctdfiles, 0)
print(average_lons4)
#print(p4)
    
#get shear, lons, lats, depth, and p_ref for each cast   
geostrophicvel1,lons1,lats1, p4_trimmed1, p_ref41=GeostrophicVelocity(ctdfiles, 0, 1)       
geostrophicvel2,lons2,lats2, p4_trimmed2, p_ref42=GeostrophicVelocity(ctdfiles, 1, 2)     
geostrophicvel3,lons3,lats3, p4_trimmed3, p_ref43=GeostrophicVelocity(ctdfiles, 2, 3)     
geostrophicvel4,lons4,lats4, p4_trimmed4, p_ref44=GeostrophicVelocity(ctdfiles, 3, 4)
geostrophicvel5,lons5,lats5, p4_trimmed5, p_ref45=GeostrophicVelocity(ctdfiles, 4, 5)  
geostrophicvel6,lons6,lats6, p4_trimmed6, p_ref46=GeostrophicVelocity(ctdfiles, 5, 6)
geostrophicvel7,lons7,lats7, p4_trimmed7, p_ref47=GeostrophicVelocity(ctdfiles, 6, 7)
#geostrophicvel8,lons8,lats8, p4_trimmed8, p_ref48=GeostrophicVelocity(ctdfiles, 7, 8)             

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
#geo8 = (geostrophicvel8[0])

#print(geo1.shape) #499
#print(geo1)

#just take lons from above results
lons1 = (geostrophicvel1[1])
lons2 = (geostrophicvel2[1])
lons3 = (geostrophicvel3[1])
lons4 = (geostrophicvel4[1])
lons5 = (geostrophicvel5[1])
lons6 = (geostrophicvel6[1])
lons7 = (geostrophicvel7[1])
#lons8 = (geostrophicvel8[1])

#make list of lons for plotting
lons = []
lons.append(lons1[0])
lons.append(lons2[0])
lons.append(lons3[0])
lons.append(lons4[0])
lons.append(lons5[0])
lons.append(lons6[0])
lons.append(lons7[0])
#lons.append(lons8[0])

#ADCP BEGIN

#open netCDF file using xarray
adcp_data = xr.open_dataset('focus_adcp.nc')

#print(f"Number of variables: {len(adcp_data.data_vars)}")
#print("Variable names:", list(adcp_data.data_vars))

#Now we can explore the dataset and see what variables are available 
#access variables, get information about each one
adcp_data['time']
adcp_data['lat']
adcp_data['lon']
adcp_data['tr_temp']
adcp_data['v']
adcp_data['u']
adcp_data['depth']


#Get ADCP "cast" at a target longitude (matches ctd casts)
def GetADCPCasts(target_lon, vel_component='v'):
    lon_values = adcp_data['lon'].values

    if lon_values.ndim == 1:
        idx = np.abs(lon_values - target_lon).argmin()

        velocity_var = adcp_data[vel_component]
        depth_var = adcp_data['depth']

        if 'time' in velocity_var.dims:
            velocity = velocity_var.isel(time=idx)
        else:
            raise ValueError(f"{vel_component} does not have a 'time' dimension.")

        if 'time' in depth_var.dims:
            depth = depth_var.isel(time=idx)
        elif 'depth' in depth_var.dims and depth_var.ndim == 1:
            depth = depth_var
        else:
            raise ValueError("Unexpected structure in 'depth' variable.")

    else:
        raise ValueError("Longitude is multidimensional. Please inspect its shape with adcp_data['lon'].shape.")

    # Convert to numpy arrays
    velocity_values = velocity.values
    depth_values = depth.values

    # Ensure equal length and filter out any NaNs
    valid_mask = ~np.isnan(velocity_values) & ~np.isnan(depth_values)
    velocity_values = velocity_values[valid_mask]
    depth_values = depth_values[valid_mask]

    # Combine into array: each row [depth, velocity]
    vel_depth_array = np.column_stack((depth_values, velocity_values))

    return velocity, depth, vel_depth_array

#qualoity check, comment out
#vel, depth, vel_depth_array = GetADCPCasts(lons1[0])
#print("Velocity values:\n", vel.values)
#print("Depth values:\n", depth.values)
#print("Velocity-Depth pairs:\n", vel_depth_array)


#integrate adcp velocity from p_ref to 150 dbar (or other)
def integrate_adcp_velocity_single(lon, p_ref, vel_component='v'):
    """
    Computes the integrated ADCP velocity profile from p_ref to 25 dbar for a single cast,
    using depth-aligned velocity-depth array.
    """
    print(f"Integrating ADCP velocity at lon={lon}, p_ref={p_ref}")

    try:
        vel, depth, vel_depth_array = GetADCPCasts(lon, vel_component=vel_component)

        # Unpack the array
        all_depths = vel_depth_array[:, 0]
        all_velocities = vel_depth_array[:, 1]

        if p_ref is None or np.isnan(p_ref):
            nan_arr = np.full_like(all_depths, np.nan)
            return nan_arr, all_depths, np.nan, np.nan, nan_arr

        # Filter valid depth range: from p_ref (deep) to 150 dbar (shallow)
        mask = (all_depths <= p_ref) & (all_depths >= 150)

        if not np.any(mask):
            nan_arr = np.full_like(all_depths, np.nan)
            return nan_arr, all_depths, np.nan, np.nan, nan_arr

        valid_depths = all_depths[mask]
        valid_velocities = all_velocities[mask]

        # Remove NaNs (already should be removed, but double-check)
        valid_mask = ~np.isnan(valid_depths) & ~np.isnan(valid_velocities)
        valid_depths = valid_depths[valid_mask]
        valid_velocities = valid_velocities[valid_mask]

        if len(valid_depths) < 2:
            nan_arr = np.full_like(valid_depths, np.nan)
            return nan_arr, valid_depths, np.nan, np.nan, nan_arr

        # Sort by increasing depth (if not already sorted)
        sort_idx = np.argsort(valid_depths)
        valid_depths = valid_depths[sort_idx]
        valid_velocities = valid_velocities[sort_idx]

        # Compute cumulative integral using trapezoidal rule
        cumulative_integrated = cumulative_trapezoid(valid_velocities, valid_depths, initial=0)


        # Mean velocity profile (pointwise mean up to each depth)
        depth_interval = valid_depths - valid_depths[0]
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_velocity_profile = np.where(depth_interval != 0,
                                             cumulative_integrated / depth_interval,
                                             np.nan)

        # Total integrated value
        total_integrated = np.trapz(valid_velocities, valid_depths)

        # Mean velocity
        depth_range = valid_depths[-1] - valid_depths[0]
        mean_velocity = total_integrated / depth_range if depth_range != 0 else np.nan

        print(f"Depth range in ADCP data: min={np.min(valid_depths)}, max={np.max(valid_depths)}")

        return cumulative_integrated, valid_depths, total_integrated, mean_velocity, mean_velocity_profile

    except Exception as e:
        print(f"Error processing lon={lon}: {e}")
        return np.full(1, np.nan), np.full(1, np.nan), np.nan, np.nan, np.full(1, np.nan)

# Example input values
lon_example = lons7[0]       # or just a float like -149.0
p_ref_example = p_ref47      # corresponding pressure reference value

# Call the function
cumulative_integrated, valid_depths, total_integrated, mean_velocity, mean_velocity_profile = integrate_adcp_velocity_single(
    lon_example, p_ref_example, vel_component='v'
)

# Combine depths and integrated velocity values into a 2D array
integrated_array = np.column_stack((valid_depths, cumulative_integrated))

# Print result
#print("Depth (dbar) and Cumulative Integrated Velocity (m/s):")
print(integrated_array)

#print("Depth profile (Cast 1):", depth_profile)
#print("Cumulative integrated velocity (Cast 1):", cumulative_profile)
#print("Total integrated velocity (Cast 1):", total_integrated)
#print("Mean velocity over range (Cast 1):", mean_velocity)
print(cumulative_integrated)

#print(mean_velocity_profile)


def GeostrophicFinal(cast_number, p_ref4, p4_trimmed, geo):
    lon_cast = lons[cast_number]
    p_ref_cast = p_ref4

    # Call the integration function to get the mean velocity profile
    _, valid_depths, _, _, mean_velocity_profile = integrate_adcp_velocity_single(
        lon_cast, p_ref_cast, vel_component='v'
    )

    if len(valid_depths) < 2 or np.all(np.isnan(mean_velocity_profile)):
        print(f"Insufficient valid integrated ADCP data for cast {cast_number}")
        return np.full((1, 2), np.nan)

    # Get geo values aligned to ADCP depths (nearest depth match)
    geo1_depths = np.array(p4_trimmed)
    geo1_values = np.ravel(geo)

    closest_geo_values = np.array([
        geo1_values[np.abs(geo1_depths - d).argmin()] for d in valid_depths
    ])

    # Geostrophic velocity difference: Integrated ADCP mean profile - Geostrophic
    velocity_difference = mean_velocity_profile - closest_geo_values

    # Optionally, reconstruct full profile (if needed)
    reconstructed_velocity = velocity_difference + closest_geo_values

    # Return paired array: [depth, velocity difference]
    result_array = np.column_stack((valid_depths, velocity_difference))
    return result_array


#quality check, comment out
#geostrophic_result = GeostrophicFinal(0, p_ref41, p4_trimmed1, geo1)
#print("Depth (dbar) and Geostrophic Velocity Difference (m/s):")
#print(geostrophic_result)

# Call GeostrophicFinal for each adcp "cast"
updated_difference1 = GeostrophicFinal(0, p_ref41, p4_trimmed1, geo1)
updated_difference2 = GeostrophicFinal(1, p_ref42, p4_trimmed2, geo2)
updated_difference3 = GeostrophicFinal(2, p_ref43, p4_trimmed3, geo3)
updated_difference4 = GeostrophicFinal(3, p_ref44, p4_trimmed4, geo4)
updated_difference5 = GeostrophicFinal(4, p_ref45, p4_trimmed5, geo5)
updated_difference6 = GeostrophicFinal(5, p_ref46, p4_trimmed6, geo6)
updated_difference7 = GeostrophicFinal(6, p_ref47, p4_trimmed7, geo7)
#updated_difference8 = GeostrophicFinal(7, p_ref48, p4_trimmed8, geo8)
#print(updated_difference3)

all_results = [
    updated_difference1,
    updated_difference2,
    updated_difference3,
    updated_difference4,
    updated_difference5,
    updated_difference6,
    updated_difference7,
    #updated_difference8
]

# Extract depths and velocities separately and pad to same length
depth_lists = []
velocity_lists = []


for arr in all_results:
    depths = arr[:, 0]
    velocities = arr[:, 1]
    depth_lists.append(depths)
    velocity_lists.append(velocities)

# Find max length of depth arrays
max_len = max(len(d) for d in depth_lists)
#print(velocity_lists)


# Initialize arrays with NaNs for padding
depth_array = np.full((max_len, len(depth_lists)), np.nan)
velocity_array = np.full((max_len, len(velocity_lists)), np.nan)

# Fill arrays
for i, (d, v) in enumerate(zip(depth_lists, velocity_lists)):
    length = len(d)
    depth_array[:length, i] = d
    velocity_array[:length, i] = v

# For X axis: use your longitude array (make sure length matches casts)
lons_array = np.array(lons[:len(all_results)])

# For Y axis: depths — use depth_array's rows (depth varies by cast, so choose average or min depth per row)
# We'll just use the depth_array directly for plotting (same shape as velocity_array)

# Create meshgrid for contour plot
X, Y = np.meshgrid(lons_array, np.arange(max_len))

# Because depths vary by cast, to plot by depth instead of index:
# Let's approximate Y with mean depth per row ignoring NaNs for better depth axis
mean_depths = np.nanmean(depth_array, axis=1)

# Replace Y with mean depths
Y = np.tile(mean_depths[:, np.newaxis], (1, len(lons_array)))

#print((velocity_array))

# Plot
ContourPlot(
    X, Y, velocity_array,
    cmap=cmo.cm.speed,
    xlabel='Longitude',
    ylabel='Depth (dbar)',
    title='Geostrophic Velocity Cross Section',
    cbarlabel='Velocity (m/s)'
)

print(Y.min(), Y.max())

'''        

# Print the results
'''
'''
for cast1_idx, cast2_idx, largest_pressure in results:
    print(f"Cast {cast1_idx} and Cast {cast2_idx} have the largest common pressure: {largest_pressure}")

#cast 1
SP1, p1, t1, p_ref1,lons1,lats1, average_lons1, average_lats1, average_lons_next_cast1=GetCastData(ctdfiles,0)
#print(average_lons1)
SA1 = gsw.SA_from_SP(SP1, p1, average_lons1, average_lats1)
CT1=gsw.CT_from_t(SA1,t1,p1)
SA1 = np.array(SA1)
CT1 = np.array(CT1)
p1 = np.array(p1)

geo_strf_dyn_height_1 = gsw.geo_strf_dyn_height(SA1, CT1, p1, p_ref1)
#print(geo_strf_dyn_height_1)
# After calculating geo_strf_dyn_height_1
#cast 2
SP2, p2, t2, p_ref2,lons2,lats2, average_lons2, average_lats2, average_lons_next_cast2=GetCastData(ctdfiles,1)
#print(lons2)
SA2 = gsw.SA_from_SP(SP2, p2, average_lons2, average_lats2)
CT2=gsw.CT_from_t(SA2,t2,p2)
SA2 = np.array(SA2)
CT2 = np.array(CT2)
p2 = np.array(p2)

# Compute dynamic height relative to p_ref1
geo_strf_dyn_height_2 = gsw.geo_strf_dyn_height(SA2, CT2, p2, p_ref1)

# Find the last common value in p1 and p2
common_depths = np.intersect1d(p1, p2)

if common_depths.size > 0:
    last_common_value = common_depths[-1]

    # Find the indices where the values in p1 and p2 are less than or equal to the last common value
    p1_trimmed_indices = np.where(p1 <= last_common_value)[0]
    p2_trimmed_indices = np.where(p2 <= last_common_value)[0]

    # Trim the arrays to the same length
    p1_trimmed = p1[p1_trimmed_indices]
    p2_trimmed = p2[p2_trimmed_indices]

    # Trim geo_strf_dyn_height_1 and geo_strf_dyn_height_2 to match the trimmed depths
    geo_strf_dyn_height_1_trimmed = geo_strf_dyn_height_1[:len(p1_trimmed)]
    geo_strf_dyn_height_2_trimmed = geo_strf_dyn_height_2[:len(p2_trimmed)]

    print("Number of values in geo_strf_dyn_height_1_trimmed:", geo_strf_dyn_height_1_trimmed.size)
    print("Number of values in geo_strf_dyn_height_2_trimmed:", geo_strf_dyn_height_2_trimmed.size)
    print("Number of values in p1_trimmed:", p1_trimmed.size)
    print("Number of values in p2_trimmed:", p2_trimmed.size)

# Create a 2D array where each row corresponds to a depth, 
# with the first column being geo_strf_dyn_height_1_trimmed and 
# the second column being geo_strf_dyn_height_2_trimmed
geo_strf_dyn_height_2D = np.column_stack((geo_strf_dyn_height_1_trimmed, geo_strf_dyn_height_2_trimmed))


#print(geo_strf_dyn_height_2)
#print(p2)
#print(p2_clipped)
# Optional: print shape or data to verify
#print("Shape of combined array:", geo_strf_dyn_height_combined.shape)
#print(geo_strf_dyn_height_combined)
#print("Number of values in geo_strf_dyn_height_2:", geo_strf_dyn_height_2.size)

#geo_strf_dyn_height = np.column_stack([geo_strf_dyn_height_1, geo_strf_dyn_height_2])  # shape: (n_levels, 2)
# Ensure both casts are aligned on the same pressure levels for a valid 2D array
# We’ll interpolate geo_strf_dyn_height_2 to p1_clipped if p2_clipped differs

# Create a 2D array: rows = [cast1, cast2]; columns = pressure levels
#geo_strf_dyn_height_2D = np.vstack([geo_strf_dyn_height_1_aligned, geo_strf_dyn_height_2_aligned]).T

#print(geo_strf_dyn_height_2D)

lons = np.array([average_lons1, average_lons2])
lats = np.array([average_lats1, average_lats1])
print(lons)
print(lats)
#print(lons)
#print(lats)

#lon=np.array([average_lons1, average_lons2])
#lon = np.full_like(p, average_lons)  # Create an array of shape (591,) with repeated longitude values
#lats = [float(average_lats), float(average_lats)]
#print(lon)
#print(lats1)

#lons_final=generate_evenly_spaced(lons1)
#print(lons_final)

#lats_final=repeat_number(lats1)

geostrophic_velocity = gsw.geostrophic_velocity(geo_strf_dyn_height_2D, lons, lats)
print(geostrophic_velocity)

#print(geo_strf_dyn_height)
#print(p_ref)
#print(CT)
#print(SA)
#print(average_lons)
#print(lon)
#print(p)
#GeoVel1=GibbsSeawater(ctdfiles,0)
#print(GeoVel1)
   
# cast 4
SP4, p4, t4, p_ref4, lons4, lats4, average_lons4, average_lats4, average_lons_next_cast4 = GetCastData(ctdfiles, 3)
# %%
SA4 = gsw.SA_from_SP(SP4, p4, average_lons4, average_lats4)

CT4 = gsw.CT_from_t(SA4, t4, p4)
SA4 = np.array(SA4)
CT4 = np.array(CT4)
p4 = np.array(p4)

geo_strf_dyn_height_4 = gsw.geo_strf_dyn_height(SA4, CT4, p4, p_ref4)

# cast 5
SP5, p5, t5, p_ref5, lons5, lats5, average_lons5, average_lats5, average_lons_next_cast5 = GetCastData(ctdfiles, 4)
SA5 = gsw.SA_from_SP(SP5, p5, average_lons5, average_lats5)
CT5 = gsw.CT_from_t(SA5, t5, p5)
SA5 = np.array(SA5)
CT5 = np.array(CT5)
p5 = np.array(p5)

# Compute dynamic height relative to p_ref4
geo_strf_dyn_height_5 = gsw.geo_strf_dyn_height(SA5, CT5, p5, p_ref4)
#print(geo_strf_dyn_height_5)
# Find common depths
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

    #print("Number of common depths:", len(common_depths_45))
    #print(len(geo_strf_dyn_height_4_trimmed))
    #print(len(geo_strf_dyn_height_5_trimmed))
    #print("geo_strf_dyn_height_4_trimmed:", geo_strf_dyn_height_4_trimmed)
    #print("geo_strf_dyn_height_5_trimmed:", geo_strf_dyn_height_5_trimmed)

    # Combine trimmed dynamic height values into a 2D array (each row: [cast4_value, cast5_value])
    geo_strf_dyn_height_45_array = np.column_stack((geo_strf_dyn_height_4_trimmed, geo_strf_dyn_height_5_trimmed))

    #print("2D array of corresponding dynamic heights (cast 4 and cast 5):")
    #print(geo_strf_dyn_height_45_array)
    #print("Shape of 2D array:", geo_strf_dyn_height_45_array.shape)
    
    lons = np.array([average_lons4, average_lons5])
    lats = np.array([average_lats4, average_lats4])
    print(lons)
    print(lats)
    
    geostrophic_velocity = gsw.geostrophic_velocity(geo_strf_dyn_height_45_array, lons, lats)
    print(geostrophic_velocity)
'''
'''
# Create a 2D array where each row corresponds to a depth,
# with the first column being geo_strf_dyn_height_4_trimmed and
# the second column being geo_strf_dyn_height_5_trimmed
geo_strf_dyn_height_2D_45 = np.column_stack((geo_strf_dyn_height_4_trimmed, geo_strf_dyn_height_5_trimmed))

# Optional: print shape or data to verify
#print("Shape of combined array:", geo_strf_dyn_height_2D_45.shape)
#print(geo_strf_dyn_height_2D_45)

lons_45 = np.array([average_lons4, average_lons5])
lats_45 = np.array([average_lats4, average_lats5])
print(lons_45)
print(lats_45)

# Compute geostrophic velocity for the new cast pair
geostrophic_velocity_45 = gsw.geostrophic_velocity(geo_strf_dyn_height_2D_45, lons_45, lats_45)
print(geostrophic_velocity_45)

#geostrophic_velocity = gsw.geostrophic_velocity(geo_strf_dyn_height_2D, lons, lats)
#print(geostrophic_velocity)

'''

'''
# Read the data from the file, skipping the header lines
file_path = "FS.bath"
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=8, names=["Distance_km", "Depth_m"])

# Extract distance and depth values
distance_km = data["Distance_km"].values
depth_m = data["Depth_m"].values

# Given origin coordinates
origin_lat = 27.000  # degrees
origin_lon = -80.088333  # degrees

# Convert distance (km) to longitude assuming 1 degree longitude ≈ 111.32 km at the equator
lon_conversion_factor = 111.32 * np.cos(np.radians(origin_lat))  # km per degree longitude at this latitude

# Compute longitudes from distances (assuming eastward direction)
longitudes = origin_lon + (distance_km / lon_conversion_factor)

# Sort data by longitude to maintain bathymetric profile
sorted_indices = np.argsort(longitudes)
lon_bathy = longitudes[sorted_indices]
depth_bathy = depth_m[sorted_indices]

ax.plot(lon_bathy, depth_bathy, c='lightgrey')
ax.fill_between(lon_bathy, depth_bathy, np.max(depth), color = "lightgrey")
'''
