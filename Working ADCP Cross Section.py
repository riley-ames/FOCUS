# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:42:33 2025

@author: ripti
"""


#import packages
import numpy as np
import cmocean as cmo
import pandas as pd
import xarray as xr
import math, os, gsw, glob, re, sys, warnings, calendar

import matplotlib.pyplot as plt 
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

import matplotlib as mpl #RA

import gsw #RA

#CTD

#read CTD data from a file 'filename' and return it to a dictonary
    #opens file 'filename', reads the lines, processes data into arrays for p, t, sal, and O2
def ReadCTDData(filename, nheaders):
    fobjs = open(filename, 'r')
    linelist = fobjs.readlines()
    fobjs.close()
    
    data_lines = linelist[nheaders:]
    ndata = len(data_lines)

    pres, temp, salt, oxy, lon, lat = ndata*[0.0], ndata*[0.0], ndata*[0.0], ndata*[0.0], ndata*[0,0], ndata*[0,0]
    
    for i,line in enumerate(linelist[nheaders:]):
        words=line.strip().split(',') #split columns comma deliminated
        #extract data and know that 0 and -9.990e-29 are code for no data in CTD data
        try:
            pres[i]= float(words[4]) if float(words[4]) not in [0, -9.990e-29] else None
            temp[i]= float(words[7]) if float(words[7]) not in [0, -9.990e-29] else None
            salt[i]= float(words[17]) if float(words[17]) not in [0, -9.990e-29] else None
            oxy[i] = float(words[19]) if float(words[19]) not in [0, -9.990e-29] else None#ml/l 
            lon[i] = float(words[2]) 
            lat[i] = float(words[1])
        except (ValueError, IndexError):
            print('Error')
    #exclude points where we have no data from cast    
    cast = {
        'pressure': [value for value in pres if value is not None],
        'salt': [value for value in salt if value is not None],
        'temperature': [value for value in temp if value is not None],
        'oxygen': [value for value in oxy if value is not None],
        'longitude': [value for value in lon if value is not None],
        'latitude': [value for value in lat if value is not None]
    }

    
    labels = linelist[0].strip().split(',')
    for label in labels:
       cast[label] = None  # Add header keys to dictionary with initial None values

        
    return (cast)

def calculatelons(lons): #function to calculate the average longitudes for each cast, use later for graphing
    average_lons = []
    for lon_array in lons:
        if lon_array:
            valid_lons = [value for value in lon_array if isinstance(value, (int, float)) and value not in [0, -9.990e-29]]
            
            # Calculate the average if there are valid longitudes
            if valid_lons:
                avg_lon = sum(valid_lons) / len(valid_lons)
                average_lons.append(avg_lon)
        else:
            continue
    
    return average_lons

#process multiple CTD files, sorts them, reads data, and interpolates onto pressure grid
def interpData(ctdfiles):
    lons = []
    lats = []
    p = np.arange(1, 801)
    sigma0_grid = np.arange(22, 28, 0.0075)
    tlev = np.arange(5, 31, 5)
    temp = np.empty((800, 9))
    salt = np.empty((800, 9))
    oxy = np.empty((800, 9))
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
     
    average_lons = calculatelons(lons) #pas the list of longitudes into the average longitudes function
    print("Average longitudes for each cast:", average_lons)

    return p, lons, temp, salt, oxy, cast_nums, lats, average_lons


#add interpolated data to dataset dictionary
def create_dataset(dataset, lons, temp, salt, oxy, cast_nums, lats):
    
    dataset['lons'].append(lons)
    dataset['temperature'].append(temp)
    dataset['salinity'].append(salt)
    dataset['oxygen'].append(oxy)
    dataset['cast_nums'].append(cast_nums)
    dataset['lats'].append(lats)
    
    return dataset

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
    
p, lons, temp, salt, oxy, cast_nums, lats, average_lons = interpData(ctdfiles)

                                         
total_dataset = create_dataset(total_dataset, lons, temp, salt, oxy, cast_nums, lats)

if average_lons: #quality check
    print("Calculated Average Longitudes:", average_lons)
else:
    print("No valid average longitudes found.")

#create two new lists 
valid_lons = []
valid_lats = []

#adds each longitude to valid_lons list if the longitude exists in the data
for lon_array in lons:
    if lon_array is not None:
        valid_lons.extend([value for value in lon_array if value is not None])
if valid_lons: #finds the minimum and maximum longitude if valid longitudes exist
    min_lon = min(valid_lons)
    max_lon = max(valid_lons)
    print(f"Min Longitude: {min_lon}, Max Longitude: {max_lon}")
else:
    print("No valid longitudes")
#do same as above for latitudes
for lat_array in lats:
    if lat_array is not None:
        valid_lats.extend([value for value in lat_array if value is not None])
if valid_lats:
    min_lat = min(valid_lats)
    max_lat = max(valid_lats)
    print(f"Min Latitude: {min_lat}, Max Latitude: {max_lat}")
else:
    print("No valid latitudes")

labels = [f"Cast Number: {i}" for i in range(1, len(ctdfiles) + 1)]

'''
plotting CTD data
def ContourPlot(x,y,z, cmap, xlabel, ylabel,title,cbarlabel): #plot cross sections of temp, oxy, and salt
    #create figure
    fig, ax = plt.subplots(figsize = (8, 6))
    
    #create plot    
    plot = ax.contourf(x, y, z, cmap = cmap, levels = 120)
    
    #add contour lines (can remove)
    #contour_lines = ax.contour(x, y, z, colors='black', linewidths=0.5, levels=10) # Adjust levels as needed
    #ax.clabel(contour_lines, fmt='%1.1f', fontsize=8)
    
    #invert y axis so depth increases, add labels, title, and tick marks
    ax.invert_yaxis()
    ticks = np.arange(0, 801, 100)
    ax.set(xlabel=xlabel)
    ax.set(ylabel=ylabel)
    fig.suptitle(title)
    
    #add colorbar, label colorbar
    cbar=fig.colorbar(plot, ax=ax, orientation='vertical')
    cbar.set_label(cbarlabel)
    
    #show plot
    plt.show()
    
    return

#invoke function to graph three cross sections with contour for temp, salinity, and oxygen related to depth and longitude
ContourPlot(average_lons,p,temp, cmo.cm.thermal, 'Longitude', 'Depth(m)', 'Focus CTD Data Temperature', 'Temperature(C)')
ContourPlot(average_lons,p,salt,cmo.cm.haline, 'Longitude', 'Depth(m)', 'Focus CTD Data Salinity', 'Salinity')
ContourPlot(average_lons,p,oxy, cmo.cm.oxy, 'Longitude', 'Depth(m)', 'Focus CTD Data Oxygen', 'Oxygen (kg/mol')
'''

#set font properties
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 20}
mpl.rc('font', **font)

#define hlelper functions
    #rot_ticks function rotates x-axis tick labels to rot angle and ha horizontal alignment
def rot_ticks(axs,rot,ha):
    for xlabels in axs.get_xticklabels():
                xlabels.set_rotation(rot)
                xlabels.set_ha(ha)

#add land, rivers, lakes                
def add_features(ax):
    
    ax.add_feature(cfeature.LAND,   facecolor='0.8',edgecolor='k',zorder=3)
    ax.add_feature(cfeature.RIVERS, edgecolor='w'  ,zorder=3)
    ax.add_feature(cfeature.LAKES,  facecolor='w'  ,zorder=3)
    ax.set(xlabel='',ylabel='')
#add gridlines    
    gls = ax.gridlines(crs=ccrs.PlateCarree(), 
                        draw_labels=True,
                        x_inline=False, 
                        y_inline=False,
                        linewidth=0.75,
                        alpha=0.75, 
                        linestyle='--',
                        lw=0,
                        color='k',
                        ylocs=mpl.ticker.MultipleLocator(base=2),
                        xlocs=mpl.ticker.MultipleLocator(base=2))

    gls.top_labels = False #disables top gridlines
    gls.bottom_labels = True #enables bottom gridlines
    gls.right_labels = False    #disables right side gridlines
    gls.left_labels = True #enables left side gridlines
    gls.xpadding=10 #sets padding between x-axis gridline labels and axis to 10 units
    gls.ypadding=10 #sets padding between y-axis gridline labels and axis to 10 units
    for k, spine in ax.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(10)

#opens netCDF file gebco_bathy using xarray
#interpolates data to new grid with specified longitude and latitude ranges and intervals
bathy = xr.open_dataset('gebco_bathy.nc')
bathy = bathy.interp(lon=np.arange(min_lon, max_lon,.05),lat=np.arange(min_lat, max_lat,.05)) #creates grid where minimum and maximum longitudes are th calculated CTD minimum and maximum longitudes
#opens netCDF file 
ds = xr.open_dataset("os75nb.nc")          

#can only run this once
#set depth variable by selecting first time step, makes depth independent of time
#swaps dimensions to use depth instead of depth_cell
ds['depth'] = ds['depth'].isel(time=0)
ds = ds.swap_dims({"depth_cell":"depth"})

# Extract longitude values for each time step
lon_times = ds['lon'].isel(time=slice(None)).values  # Assuming longitude varies with time

# Add longitude values as a new coordinate associated with 'time'
ds = ds.assign_coords(lon=('time', lon_times))

ds = ds.sortby('lon')

#plot ADCP cross section from minimum to maximum CTD longitude
fig, (ax, bx) = plt.subplots(2, 1, figsize=(20,10), sharex= True, constrained_layout = True)

var = ['u', 'v']
labels = ["Zonal Velocity (ms-1)", "Meridional Velocity (ms-1)"]

for i, vari in enumerate(var):
    ds[vari].plot(x='lon', y="depth", ylim=(800,0), vmin=-2, vmax=2, 
                  ax=[ax, bx][i],
                  cmap="cmo.balance",
                  cbar_kwargs={'pad':0.01,'label':labels[i]})
    
if min_lon is not None and max_lon is not None:
    [ax, bx][i].set_xlim(min_lon, max_lon)

#ds["amp"].plot(y="depth", ylim=(1000,0), cmap="cmo.dense", ax=cx)
#creates figure with two vertical subplots sharing x-axis
#plots u and v against depth using colormap
rot_ticks(bx, 0, 'center')
#set axis labels
ax.set(xlabel=None)
bx.set(xlabel='Longitude')
#cx.set(xlabel=None)
#set figure title
fig.suptitle("FOCUS Ship ADCP Data")

plt.show()

print(ds)
