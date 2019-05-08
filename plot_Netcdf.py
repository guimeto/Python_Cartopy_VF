from netCDF4 import Dataset
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import cartopy.crs as ccrs
from cmocean import cm as cmo

# conda install cmocean
# tuto http://earthpy.org/tag/cartopy.html
import sys
import os
from cartopy.util import add_cyclic_point

flf = Dataset('./temperature_annual_1deg.nc')
lat = flf.variables['lat'][:]
lon = flf.variables['lon'][:]

temp = flf.variables['t_an'][0,0,:,:]
temp_cyc, lon_cyc = add_cyclic_point(temp, coord=lon)

plt.figure(figsize=(13,6.2))
    
ax = plt.subplot(111, projection=ccrs.PlateCarree())

mm = ax.pcolormesh(lon_cyc,\
                   lat,\
                   temp_cyc,\
                   vmin=-2,\
                   vmax=30,\
                   transform=ccrs.PlateCarree(),\
                   cmap=cmo.balance )
ax.coastlines();

# Looks fine. Now we will use default cartopy backgroung:
plt.figure(figsize=(13,6.2))
    
ax = plt.subplot(111, projection=ccrs.PlateCarree())

ax.set_extent([30,-70,-30,70])

mm = ax.pcolormesh(lon_cyc,\
                   lat,\
                   temp_cyc,\
                   vmin=-2,\
                   vmax=30, \
                   transform=ccrs.PlateCarree(),\
                   cmap=cmo.balance )


ax.coastlines(resolution='110m');

ax.stock_img();

#Let's setup cartopy for use of the custom background. 
#You have to have a folder that contain background images and a json file that describes images (see explination below).
# Then you have to create environment variable that contains the path to the folder:

#os.environ["CARTOPY_USER_BACKGROUNDS"] = "K:/Code Python/Cartopy/"

# Now you can specify name of the image in it's resolution in the background_img() method:

#plt.figure(figsize=(13,6.2))
    
#ax = plt.subplot(111, projection=ccrs.PlateCarree())
#ax.background_img(name='BM', resolution='low')
#mm = ax.pcolormesh(lon_cyc,\
#                   lat,\
#                   temp_cyc,\
#                   vmin=-2,\
#                   vmax=30,\
#                   transform=ccrs.PlateCarree(),\
#                   cmap=cmo.balance )
#ax.coastlines(resolution='110m');