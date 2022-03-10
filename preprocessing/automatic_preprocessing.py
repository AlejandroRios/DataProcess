"""" 
Function  : Data cleaning/preprocessing
Title     : Data_Process
Written by: Alejandro Rios
Date      : April/2019
Language  : Python
Aeronautical Institute of Technology / Airbus Brazil
"""
########################################################################################
"""Importing Modules"""
########################################################################################
import pandas as pd
import numpy as np
import sklearn
# import haversine
import pickle
# from haversine import haversine, Unit
# from geopy.distance import distance

import flightphase
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import scipy as sp
from scipy import signal
from horizontal_inefficiency import horizontal_ineff
########################################################################################
"""Importing Data"""
########################################################################################

def data_preprocessing(departure,arrival):
    
    name = "allFlights_" + departure + arrival +".csv"
    print('[0] Load dataset.\n')
        
    # df = pd.read_csv('allFlights.csv',  header=None, delimiter=',')
    df = pd.read_csv(name, delimiter=',')
    df_head = df.head()
    print(df_head)

    ########################################################################################
    """Data manipulation and checks"""
    ########################################################################################
    print('[1] Data manipulation and checks.\n')

    # Calculate difference between current and previus timestep for latitude and store information in new column
    df['amount_diff'] = (df['lat'].diff().fillna(0)).abs()

    # Check if difference in latitude between timesteps is greater than 3. This will be used as another criteria to define a new flight 
    df['big_diff'] = df['amount_diff'].apply(lambda x: 'False' if x <= 3 else 'True')

    # Drop outliers declared by AirSense
    # df.drop(df[df['isoutlier'] == True].index, inplace=True)

    # Calculate time in seconds [s]
    df['posTime'] = df['posTime']/1000 

    # Calculate delta altitude [ft]
    df['delta_h'] = (df['alt'].diff().fillna(0))

    # Calculate delta time [s]
    df['delta_t'] = (df['posTime'].diff().fillna(0))

    # Calculate RoC [ft/min]
    df['roc'] = (df['delta_h']/df['delta_t'])*60

    ########################################################################################
    """Filtering lat and long data out of lat lon bounds"""
    ########################################################################################
    print('[2] First data filter.\n')

    # Drop outliers out of lat scale lat < -90, lat > 90
    df = df.drop(df[df.lat > 90].index)
    df = df.drop(df[df.lat < -90].index)
    # Drop outliers out of lon scale lon < -180, lon > 180
    df = df.drop(df[df.lon > 180].index)
    df = df.drop(df[df.lon < -180].index)

    # Drop rows containg NAN
    df = df[df['roc'].notna()]

    # Drop high values of roc 
    df = df.drop(df[df.roc < -5000].index)
    df = df.drop(df[df.roc > 5000].index)
    ########################################################################################
    """Accounting the number of flights"""
    ########################################################################################
    print('[3] Accounting number of flights by icao and call filters.\n')

    # query - takes all index with count = 0 of sorted data
    # df['count'] = df.groupby(((df['aircraftReg'] != df['aircraftReg'].shift(1)) | (df['targetId'] != df['targetId'].shift(1)) | (df['missionId'] != df['missionId'].shift(1))   |  (df['big_diff'].eq('True').shift(0))).cumsum()).cumcount()
    df['count'] = df.groupby(((df['targetId'] != df['targetId'].shift(1)) | (df['airsenseMissionId'] != df['airsenseMissionId'].shift(1))   |  (df['big_diff'].eq('True').shift(0))).cumsum()).cumcount()

    # df['count'] = df.groupby(((df['big_diff'].eq('True').shift(0))).cumsum()).cumcount()

    flights = df.sort_index().query('count == 0')

    # # Number of flights
    Numflights = len(flights)
    print('Number of flights: ', Numflights )

    df['FlightNum']=df['count'].eq(0).cumsum()

    df2 = df[['count','posTime','lat','lon','alt','speed','mach','tas',"heading","delta_h","delta_t","roc",'targetId','FlightNum']]

    ########################################################################################
    """Flight phase identification"""
    ########################################################################################
    print('[4] Performing phase identification for all flights.\n')

    data_gby = df2.groupby('FlightNum')

    df3=pd.DataFrame()
    data = pd.DataFrame()


    df4 =pd.DataFrame(columns=['HFE'])

    i = 0
    for key, data in data_gby:

        if len(data) > 100:

            times = np.array(data['posTime'])
            times = times - times[0]
            alts = np.array(data['alt'])
            spds = np.array(data['speed'])
            rocs = np.array(data['roc'])
            lat = np.array(data['lat'])
            lon = np.array(data['lon'])

            # df4.append({'HFE':horizontal_ineff(lat,lon)},ignore_index=True)
            df4.loc[i] = [horizontal_ineff(lat,lon)]
            # print(df4['HFE'])
            i = i+1
            # df4 = df4.append(HEF)

            labels = flightphase.fuzzylabels(times, alts, spds, rocs)

            data['flight_phase'] = labels
            data['times'] = times
            data['Phase_count']=(data['flight_phase'] != data['flight_phase'].shift(1)).cumsum()


            if (data.flight_phase.values == 'CR').sum() > 30:

                df3=df3.append(data,ignore_index=True)



            colormap = {'GND': 'black', 'CL': 'green', 'CR': 'blue',
                        'DE': 'yellow', 'LVL': 'purple', 'NA': 'red'}

            colors = [colormap[l] for l in labels]


            altspl = UnivariateSpline(times, alts)(times)
            # altspl = sp.signal.savgol_filter(altspl,21, 3)
            altspl = sp.signal.medfilt(altspl,51)
            spdspl = UnivariateSpline(times, spds)(times)
            # spdspl = sp.signal.savgol_filter(spdspl, 11, 2)
            rocspl = UnivariateSpline(times, rocs)(times)
            # rocspl = sp.signal.medfilt(rocspl,21)
            rocspl = sp.signal.savgol_filter(rocspl, 21, 3)
            

            if i < 5:
                plt.subplot(411)
                plt.plot(lat,lon , '-', color='k', alpha=0.5)
                plt.scatter(lat, lon, marker='.', c=colors, lw=0)
                plt.xlabel('lat (deg)')
                plt.ylabel('lon (deg)')
                plt.grid(True)

                plt.subplot(412)
                # plt.title('press any key to continue...')
                plt.plot(times, altspl, '-', color='k', alpha=0.5)
                plt.scatter(times, alts, marker='.', c=colors, lw=0)
                plt.ylabel('altitude (ft)')
                plt.grid(True)

                plt.subplot(413)
                plt.plot(times, spdspl, '-', color='k', alpha=0.5)
                plt.scatter(times, spds, marker='.', c=colors, lw=0)
                plt.ylabel('speed (kt)')
                plt.grid(True)

                plt.subplot(414)
                plt.plot(times, rocspl, '-', color='k', alpha=0.5)
                plt.scatter(times, rocs, marker='.', c=colors, lw=0)
                plt.ylabel('roc (fpm)')
                plt.xlabel('time')
                plt.grid(True)

                plt.tight_layout()
                plt.draw()
                plt.waitforbuttonpress(-1)
                plt.clf()

    # Store flights that contain more than 500 points of data
    df3 = df3[df3.groupby('FlightNum')['FlightNum'].transform('count').ge(500)]

    ########################################################################################
    """Saving processed data into new .csv"""
    ########################################################################################

    print('[5] Saving processed data into new .csv.\n')
    # "allFlights_" + departure + arrival +".csv"
    name_data_cluster = "Data4Clustering_" + departure + arrival + ".csv"
    name_data_HEF ="data_HEF_" + departure + arrival + ".csv"
    df3.to_csv(name_data_cluster)
    df4.to_csv(name_data_HEF)
    # df.to_pickle("Data4Clustering01.pkl")
    print('[6] All completed.\n')

    return


departure = "ZRH"
arrival = "LHR"
data_preprocessing(departure,arrival)