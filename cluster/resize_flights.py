"""" 
Function  : spline_resize
Title     : Spline rezise function
Written by: Alejandro Rios
Date      : March/2020
Language  : Python
Aeronautical Institute of Technology - Airbus Brazil
"""
########################################################################################
"""Importing Modules"""
########################################################################################
import numpy as np
import scipy as sp

from scipy import interpolate
from scipy.interpolate import make_lsq_spline, BSpline
from scipy import signal

def resize(df,Nflights,chunk_size,cluster_type):
    
    coordinates_vec = []

    print(df.head())

    for i in range(0,len(Nflights)-1): 

        flights = df.iloc[Nflights.index[i]:Nflights.index[i+1]]

        # Resizing vector of flights time
        time= flights['times']
        xtime= np.arange(time.size)
        new_xtime = np.linspace(xtime.min(), xtime.max(), chunk_size)
        time_rz_1 = sp.interpolate.interp1d(xtime, time, kind='linear')(new_xtime)
        time_rz = time_rz_1-time_rz_1[0]
        # time_rz = sp.signal.medfilt(time_rz,51)
        
        # Resizing vector of flights lat, lon and alt
        lat = flights['lat']
        xlat = np.arange(lat.size)
        new_xlat = np.linspace(xlat.min(), xlat.max(), chunk_size)
        lat_rz = sp.interpolate.interp1d(xlat, lat, kind='linear')(new_xlat)
        # lat_rz = sp.signal.medfilt(lat_rz,51)

        lon = flights['lon']
        xlon = np.arange(lon.size)
        new_xlon = np.linspace(xlon.min(), xlon.max(), chunk_size)
        lon_rz = sp.interpolate.interp1d(xlon, lon, kind='linear')(new_xlon)
        # lon_rz = sp.signal.medfilt(lon_rz,51)

        alt = flights['alt']
        # alt = sp.signal.savgol_filter(alt, 11, 3)
        xalt = np.arange(alt.size)
        new_xalt = np.linspace(xalt.min(), xalt.max(), chunk_size)
        alt_rz = sp.interpolate.interp1d(xalt, alt, kind='linear')(new_xalt)
        # alt_rz = sp.signal.medfilt(alt_rz,51)
        # alt_rz = sp.signal.savgol_filter(alt_rz, 11, 2)

        spds= flights['speed']
        xspds= np.arange(spds.size)
        new_xspds = np.linspace(xspds.min(), xspds.max(), chunk_size)
        spds_rz = sp.interpolate.interp1d(xspds, spds, kind='linear')(new_xspds)
        # spds_rz = sp.signal.medfilt(spds_rz,51)

        rocs= flights['roc']
        xrocs= np.arange(rocs.size)
        new_xrocs = np.linspace(xrocs.min(), xrocs.max(), chunk_size)
        rocs_rz = sp.interpolate.interp1d(xrocs, rocs, kind='linear')(new_xrocs)
        # rocs_rz = sp.signal.medfilt(rocs_rz,51)

        tas_spds= flights['tas']
        xtasspds= np.arange(tas_spds.size)
        new_tas_xspds = np.linspace(xtasspds.min(), xtasspds.max(), chunk_size)
        tas_spds_rz = sp.interpolate.interp1d(xtasspds, tas_spds, kind='linear')(new_tas_xspds)

        machs= flights['mach']
        xmachs= np.arange(machs.size)
        new_xmachs = np.linspace(xmachs.min(), xmachs.max(), chunk_size)
        machs_rz = sp.interpolate.interp1d(xmachs, machs, kind='linear')(new_xmachs)
        # spds_rz = sp.signal.medfilt(spds_rz,51)



        if cluster_type == 0:
            coordinates = np.concatenate((time_rz[:,None], lon_rz[:,None],lat_rz[:,None],alt_rz[:,None],spds_rz[:,None],rocs_rz[:,None],tas_spds_rz[:,None],machs_rz[:,None]),axis=1)
        elif cluster_type == 1:
            coordinates = np.concatenate((time_rz[:,None], lon_rz[:,None],lat_rz[:,None],alt_rz[:,None],spds_rz[:,None],rocs_rz[:,None],tas_spds_rz[:,None],machs_rz[:,None]),axis=1)
        coordinates = tuple(map(tuple, coordinates))
        coordinates_vec.append(np.vstack(coordinates))

    return(coordinates_vec)
    
def resize_norm(df,Nflights,chunk_size,cluster_type):
    
    coordinates_vec = []
    # for i in range(6): 
    for i in range(len(Nflights)-1): 
    # for i in range(1000):

        # Separate vector by flights
        flights = df.iloc[Nflights.index[i]:Nflights.index[i+1]]

        # Resizing vector of flights lat and lon
        lat = flights['lat_norm']
        xlat = np.arange(lat.size)
        new_xlat = np.linspace(xlat.min(), xlat.max(), chunk_size)
        lat_rz = sp.interpolate.interp1d(xlat, lat, kind='slinear')(new_xlat)

        lon = flights['lon_norm']
        xlon = np.arange(lon.size)
        new_xlon = np.linspace(xlon.min(), xlon.max(), chunk_size)
        lon_rz = sp.interpolate.interp1d(xlon, lon, kind='slinear')(new_xlon)

        alt = flights['alt_norm']
        xalt = np.arange(alt.size)
        new_xalt = np.linspace(xalt.min(), xalt.max(), chunk_size)
        alt_rz = sp.interpolate.interp1d(xalt, alt, kind='slinear')(new_xalt)

        time = flights['times_norm']
        xtime = np.arange(time.size)
        new_xtime = np.linspace(xtime.min(), xtime.max(), chunk_size)
        time_rz = sp.interpolate.interp1d(xtime, time, kind='slinear')(new_xtime)

        # coordinates = np.concatenate((lon_rz[:,None],lat_rz[:,None],alt_rz[:,None]),axis=1)
        if cluster_type == 0:
            coordinates = np.concatenate((lon_rz[:,None],lat_rz[:,None]),axis=1)
        elif cluster_type == 1:
            # coordinates = alt_rz[:,None]
            coordinates = np.concatenate((lat_rz[:,None],alt_rz[:,None]),axis=1)

        coordinates = tuple(map(tuple, coordinates))
        coordinates_vec.append(np.vstack(coordinates))

    return(coordinates_vec)


def resize_4cluster(df,Nflights,chunk_size,cluster_type):
    coordinates_vec = []
    for i in range(0,len(Nflights)-1): 
        # Separate vector by flights
        flights = df.iloc[Nflights.index[i]:Nflights.index[i+1]]

        # flights = flights [~flights ['flight_phase'].isin(['GND','CL','DE','LVL','NA'])]
        flights = flights [~flights ['flight_phase'].isin(['CL','GND','DE','LVL','NA'])]

        flights.dropna(subset=['flight_phase'], inplace=True)

        # Resizing vector of flights time
        time= flights['times']
        xtime= np.arange(time.size)
        new_xtime = np.linspace(xtime.min(), xtime.max(), chunk_size)
        time_rz_1 = sp.interpolate.interp1d(xtime, time, kind='slinear')(new_xtime)
        time_rz = time_rz_1-time_rz_1[0]
        # time_rz = sp.signal.medfilt(time_rz,51)
        
        # Resizing vector of flights lat, lon and alt
        lat = flights['lat']
        xlat = np.arange(lat.size)
        new_xlat = np.linspace(xlat.min(), xlat.max(), chunk_size)
        lat_rz = sp.interpolate.interp1d(xlat, lat, kind='slinear')(new_xlat)
        # lat_rz = sp.signal.medfilt(lat_rz,51)

        lon = flights['lon']
        xlon = np.arange(lon.size)
        new_xlon = np.linspace(xlon.min(), xlon.max(), chunk_size)
        lon_rz = sp.interpolate.interp1d(xlon, lon, kind='slinear')(new_xlon)
        lon_rz = sp.signal.medfilt(lon_rz,51)

        alt = flights['alt']
        xalt = np.arange(alt.size)
        new_xalt = np.linspace(xalt.min(), xalt.max(), chunk_size)
        alt_rz = sp.interpolate.interp1d(xalt, alt, kind='cubic')(new_xalt)
        alt_rz = sp.signal.medfilt(alt_rz,51)
        # alt_rz = sp.signal.savgol_filter(alt_rz, 11, 2)

        spds= flights['speed']
        xspds= np.arange(spds.size)
        new_xspds = np.linspace(xspds.min(), xspds.max(), chunk_size)
        spds_rz = sp.interpolate.interp1d(xspds, spds, kind='slinear')(new_xspds)
        # spds_rz = sp.signal.medfilt(spds_rz,51)

        rocs= flights['roc']
        xrocs= np.arange(rocs.size)
        new_xrocs = np.linspace(xrocs.min(), xrocs.max(), chunk_size)
        rocs_rz = sp.interpolate.interp1d(xrocs, rocs, kind='slinear')(new_xrocs)
        # rocs_rz = sp.signal.medfilt(rocs_rz,51)

        # coordinates = np.concatenate((lon_rz[:,None],lat_rz[:,None],alt_rz[:,None]),axis=1)
        if cluster_type == 0:
            coordinates = np.concatenate((time_rz[:,None], lon_rz[:,None],lat_rz[:,None],alt_rz[:,None],spds_rz[:,None],rocs_rz[:,None]),axis=1)
        elif cluster_type == 1:
            # coordinates = alt_rz[:,None]
            coordinates = np.concatenate((time_rz[:,None], lon_rz[:,None],lat_rz[:,None],alt_rz[:,None],spds_rz[:,None],rocs_rz[:,None]),axis=1)

        coordinates = tuple(map(tuple, coordinates))
        coordinates_vec.append(np.vstack(coordinates))

    return(coordinates_vec)
    
def resize_norm_4cluster(df,Nflights,chunk_size,cluster_type):
    
    coordinates_vec = []

    for i in range(len(Nflights)-1): 

        # Separate vector by flights
        flights = df.iloc[Nflights.index[i]:Nflights.index[i+1]]

        # Resizing vector of flights lat and lon
        lat = flights['lat_norm']
        xlat = np.arange(lat.size)
        new_xlat = np.linspace(xlat.min(), xlat.max(), chunk_size)
        lat_rz = sp.interpolate.interp1d(xlat, lat, kind='slinear')(new_xlat)

        lon = flights['lon_norm']
        xlon = np.arange(lon.size)
        new_xlon = np.linspace(xlon.min(), xlon.max(), chunk_size)
        lon_rz = sp.interpolate.interp1d(xlon, lon, kind='slinear')(new_xlon)

        alt = flights['alt_norm']
        xalt = np.arange(alt.size)
        new_xalt = np.linspace(xalt.min(), xalt.max(), chunk_size)
        alt_rz = sp.interpolate.interp1d(xalt, alt, kind='slinear')(new_xalt)

        time = flights['times_norm']
        xtime = np.arange(time.size)
        new_xtime = np.linspace(xtime.min(), xtime.max(), chunk_size)
        time_rz = sp.interpolate.interp1d(xtime, time, kind='slinear')(new_xtime)

        # coordinates = np.concatenate((lon_rz[:,None],lat_rz[:,None],alt_rz[:,None]),axis=1)
        if cluster_type == 0:
            coordinates = np.concatenate((lon_rz[:,None],lat_rz[:,None]),axis=1)
        elif cluster_type == 1:
            # coordinates = alt_rz[:,None]
            coordinates = np.concatenate((lat_rz[:,None],alt_rz[:,None]),axis=1)

        coordinates = tuple(map(tuple, coordinates))
        coordinates_vec.append(np.vstack(coordinates))

    return(coordinates_vec)



