"""" 
Function  : Route plot function
Title     : Route_Plot
Written by: Alejandro Rios
Date      : April/2019
Language  : Python
Aeronautical Institute of Technology / Airbus Brazil
"""
########################################################################################
"""Importing Modules"""
########################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import haversine

from mpl_toolkits.basemap import Basemap
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import StandardScaler
from haversine import haversine, Unit

from sklearn import metrics
from sklearn import preprocessing
from collections import Counter
from collections import OrderedDict
from scipy import interpolate
from itertools import cycle
from itertools import islice
from datetime import datetime
########################################################################################
"""Importing Data"""
########################################################################################

print('[0] Load dataset.\n')
    
# df = pd.read_csv('AIRP1AIRP2_5day.csv', header=0, delimiter=',')
df = pd.read_csv('Data4Clustering01.csv', header=0, delimiter=',')
df_head = df.head()

Nflights = df.sort_index().query('count == 0')

# Number of flights
Numflights = len(Nflights)

print('- Number of flights: \n', Numflights )

#########################################################################################
"""Airport coordinates"""
########################################################################################
# FRA: lat: 50.110924 | lon: 8.682127
# FCO: lat: 41.7997222222 | lon: 12.2461111111
# CDG: lat: 49.009722 | lon: 2.547778
# LHR: lat: 51.4775 | lon: -0.461389
########################################################################################


#########################################################################################
"""Base map definition"""
########################################################################################

fig, ax = plt.subplots()
m = Basemap(resolution='i', projection='merc', llcrnrlat=35, urcrnrlat=60, llcrnrlon=-15, urcrnrlon=30)
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='1.0',lake_color='aqua')
m.drawcoastlines()
m.drawcountries()
parallels = np.arange(0.,81,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False])
meridians = np.arange(10.,351.,20.)
m.drawmeridians(meridians,labels=[True,False,False,True])


#########################################################################################
"""Re-sizing data/spline """
########################################################################################

for i in range(len(Nflights)-1):
    # Define some parametres
    CHUNK_SIZE = 500 # Size of flight vector

    # Separate vector by flights
    flights = df.iloc[Nflights.index[i]:Nflights.index[i+1]]
    
    # Resizing vector of flights lat, lon and alt
    lat = flights['lat']
    xlat = np.arange(lat.size)
    new_xlat = np.linspace(xlat.min(), xlat.max(), CHUNK_SIZE)
    lat_rz = sp.interpolate.interp1d(xlat, lat, kind='slinear')(new_xlat)

    lon = flights['lon']
    xlon = np.arange(lon.size)
    new_xlon = np.linspace(xlon.min(), xlon.max(), CHUNK_SIZE)
    lon_rz = sp.interpolate.interp1d(xlon, lon, kind='slinear')(new_xlon)

    alt = flights['alt']
    xalt = np.arange(alt.size)
    new_xalt = np.linspace(xalt.min(), xalt.max(), CHUNK_SIZE)
    alt_rz = sp.interpolate.interp1d(xalt, alt, kind='slinear')(new_xalt)
    x,y = m(lon_rz,lat_rz)
    m.plot(*(x, y), color = 'b', linewidth=0.1)


    

    # for j in range(len(lon_rz)-1):

    #     # Defining cordinates of two points to messure distance      
    #     coordinates0 = (lat_rz[j],lon_rz[j])
    #     # Calculating haversine distance between two points in nautical miles
    #     distance_to_AIRP1 = float(haversine(coor_AIRP1,coordinates0,unit='nmi'))
    #     distance_to_AIRP2 = float(haversine(coor_AIRP2,coordinates0,unit='nmi'))
        
    #     if distance_to_AIRP1 > 60 and distance_to_AIRP2 > 60:

    #         lon_f = [lon_rz[j]]
    #         lat_f = [lat_rz[j]]
    #         alt_f = [alt_rz[j]]

    #         # lat_teste_f = [lat_teste[j]]
    #         # lon_teste_f = [lon_teste[j]]
            
    #         lat_ff.append(lat_f)
    #         lon_ff.append(lon_f)
    #         alt_ff.append(alt_f)

    #         # lat_teste_ff.append(lat_teste_f)
    #         # lon_teste_ff.append(lon_teste_f)

    

# lat_ff = np.asarray(lat_ff)
# lon_ff = np.asarray(lon_ff)

#########################################################################################
"""Plot Definition"""
########################################################################################
data = pd.read_csv('network_EU.csv', header=0, delimiter=',')
number_of_airports = len(data.APT)
cities=[i for i in range(number_of_airports)] # Creamos ciudades de la 0 a la 9  
arcs =[(i,j) for i in cities for j in cities if i!=j]

lon_coordinates = data.LON
lat_coordinates = data.LAT

x = lon_coordinates
y = lat_coordinates
x = x.values.tolist()
y = y.values.tolist()
# print(x)
names = data.APT

x,y = m(x,y)
m.scatter(x, y, 100, color="orange", marker="o",edgecolor="r", zorder=3)
for i in range(len(names)):
    plt.text(x[i], y[i], names[i], va="baseline",color='k', fontsize = 12,family="monospace", weight="bold")
##############################################################
def radius_for_tissot(dist_km):
    return np.rad2deg(dist_km/6367.)

# x,y=m(lon_AIRP2,lat_AIRP2)
# x2,y2 = m(lon_AIRP2,lat_AIRP2) 
# circle1 = plt.Circle((x, y), 170000, color='black',fill=False)
# ax.add_patch(circle1)

# print(y2-y)

# x,y=m(lon_AIRP1,lat_AIRP1)
# x2,y2 = m(lon_AIRP1,lat_AIRP1) 
# circle1 = plt.Circle((x, y), 170000, color='black',fill=False)
# ax.add_patch(circle1)

############################################
plt.show()










