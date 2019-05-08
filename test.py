
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def make_map(projection=ccrs.PlateCarree(), extent=[-42, 0, -32, 0.5]):
    subplot_kw = dict(projection=projection)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)
    ax.set_extent(extent)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax

url = 'http://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'

layers = ['BlueMarble_NextGeneration', 'VIIRS_CityLights_2012',
          'Reference_Features', 'Sea_Surface_Temp_Blended', 'MODIS_Terra_Aerosol',
          'Coastlines', 'BlueMarble_ShadedRelief', 'BlueMarble_ShadedRelief_Bathymetry']

for layer in layers:
    fig, ax = make_map()
    ax.add_wmts(url, layer)
    ax.set_title(layer)
