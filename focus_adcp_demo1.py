# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:45:14 2025

@author: ripti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:04:44 2023

@author: palomacartwright
"""
#import nesseccary packages
    #xarray is for netCDF files, catopy is for mapping, cmocean is for colormaps
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

#define font properties
font = {'family' : 'Avenir',
        'weight' : 'normal',
        'size'   : 25}

#open netCDF file using xarray
adcp_data = xr.open_dataset('os75nb.nc')



#Now we can explore the dataset and see what variables are available 
#access variables, get information about each one
adcp_data['time']
adcp_data['lat']
adcp_data['lon']
adcp_data['tr_temp']
#Plot the adcp data 
'''
#creates figure with two vertical sublots sharing the x-axis
fig1, (ax1, bx1) = plt.subplots(2, 1, figsize=(10,8), sharex = True)
#define var as list containing variables u and v, which are zonial and meridonial velocity
var = ['u', 'v']
#loop through var list, plotting each velocity against depth with color limits, colormap, and y-axis limits
for i, vari in enumerate(var): 
    adcp_data[vari].plot(y='depth', ylim=(800,0), vmin=-2, vmax=2, cmap='cmo.phase', ax=[ax1, bx1][i])
plt.show()
######################
'''
#creates map plot
    #single sublot, plate carree projection
fig, ax = plt.subplots(figsize=(20,11), subplot_kw={'projection':ccrs.PlateCarree()})
#add land, coastline, and ocean features to map
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.OCEAN)
#set map extent to specific latitude and longitude coordinates
ax.set_extent([-80.5,-78.5,27.5,25])
#add gridlines to map
gl = ax.gridlines(lw =0, 
                  draw_labels=True)
#plots longitude and latitude in black
ax.plot(adcp_data.lon, adcp_data.lat, c='k')
plt.show()
'''
#creates a quiver plot (vector field) of mean velocities u and v averaged over depth (0-100m) at selected time points (every 8th time point)
quivplt = adcp_data.isel(time=np.arange(0, len(adcp_data.time), 8),
                           depth = slice(0,100)).mean('depth').plot.quiver(x='lon',
                                                                           y='lat',
                                                                           u='u', v='v', 
                                                                           scale=8, #sets vector scaling factor
                                                                           pivot='tail', #specifies vector pivot point
                                                                           hue='tr_temp', #colors vectors based on tr_temp variable                                                                        
                                                                           cmap='spring_r') #sets color map                                                                        
plt.show()                                                                                                                                                   
                                                                           

#Bathymetry
              

bathy = xr.open_dataset('gebco_bathy.nc')
bathy = bathy.interp(lon=np.arange(-80.5,-78.5,.05),lat=np.arange(25,27.5,.05))    

roi_lon = slice(-82, -78)
roi_lat = slice(24, 28)

#Extract bathymetry data for the ROI
land_mask = bathy['elevation'] > 0
bathymetry_roi = bathy['elevation'].sel(lon=roi_lon, lat=roi_lat).where(~land_mask)                                                        

fig, ax = plt.subplots()
bathymetry_roi.plot.contourf(ax=ax, levels=15, cmap='cool_r')
                                                                           
plt.show()                                                                
'''                                                                          




