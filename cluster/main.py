"""" 
Function  : cluster_two_step
Title     : Two step cluster
Written by: Alejandro Rios
Date      : March/2020
Language  : Python
Aeronautical Institute of Technology - Airbus Brazil
"""
########################################################################################
"""Importing Modules"""
########################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter, defaultdict

from sklearn.cluster import DBSCAN
# from sklearn.cluster import KMeans
# from sklearn.cluster import OPTICS, cluster_optics_dbscan

from sklearn import metrics
from sklearn import preprocessing

# In-House functions
from resize_flights import resize, resize_norm, resize_4cluster, resize_norm_4cluster
from distances_matrix import distances

########################################################################################
"""Airport coordinates"""
########################################################################################
# FRA: lat: 50.110924 | lon: 8.682127
# FCO: lat: 41.7997222222 | lon: 12.2461111111
# CDG: lat: 49.009722 | lon: 2.547778
# LHR: lat: 51.4775 | lon: -0.461389
########################################################################################
# AIRP1 airport coordinates
lat_AIRP1 = 50.110924
lon_AIRP1 = 8.682127
coor_AIRP1 = (lat_AIRP1,lon_AIRP1)

# AIRP2 airport coordinates
lat_AIRP2 = 41.7997222222
lon_AIRP2 = 12.2461111111
coor_AIRP2 = (lat_AIRP2,lon_AIRP2)
########################################################################################
"""Inputs"""
########################################################################################
# Type horizontal = 0, vertical = 1
cluster_type = 0

# Cluster norm: Not norm = 0, norm = 1
cluster_norm = 0
########################################################################################
"""Importing Data"""
########################################################################################
print('[0] Load dataset.\n')

# CSV flights import    
df = pd.read_csv('Data4Clustering01.csv', header=0, delimiter=',')

# Separate flights aided by feature count
Nflights = df.sort_index().query('count == 0')

# Count number of flights
Numflights = len(Nflights)

print('- Number of flights: \n', Numflights )

df_head = df.head()
print('Database head: \n', df_head )

# Size of flight vector (for resizing)
chunk_size = 100 
########################################################################################
"""Data scaling"""
########################################################################################
print('--------------------------------------------------------------------------------\n')
print('[1] Data scaling (0-1).\n')

if cluster_norm == 1:
    mms_lat = preprocessing.MinMaxScaler(feature_range=(0, 1))
    df['lat_norm'] = mms_lat.fit_transform(df.lat.values.reshape((-1, 1)))

    mms_lon = preprocessing.MinMaxScaler(feature_range=(0, 1))
    df['lon_norm'] = mms_lon.fit_transform(df.lon.values.reshape((-1, 1)))

    mms_alt = preprocessing.MinMaxScaler(feature_range=(0, 100))
    df['alt_norm'] = mms_alt.fit_transform(df.alt.values.reshape((-1, 1)))

    mms_time = preprocessing.MinMaxScaler(feature_range=(0, 1000))
    mms_time = preprocessing.MinMaxScaler(feature_range=(0, 1000))
    df['times_norm'] = mms_time.fit_transform(df.time.values.reshape((-1, 1)))
    dt = mms_time.scale_ * 0.5 * 60 * 60   # time interval of 30 mins
########################################################################################
"""Re-sizing flight vectors"""
########################################################################################
print('--------------------------------------------------------------------------------\n')
print('[2] Re-sizing flight vectors (same size).\n')

# Resize each flight in the DB
if cluster_norm == 0:
    coordinates_vec = resize(df,Nflights,chunk_size,cluster_type)
    coordinates_vec_cluster = resize_4cluster(df,Nflights,chunk_size,cluster_type)

elif cluster_norm == 1:
    coordinates_vec = resize_norm(df,Nflights,chunk_size,cluster_type)
    coordinates_vec_cluster = resize_norm_4cluster(df,Nflights,chunk_size,cluster_type)

########################################################################################
""" Meassuring Hausdorff distance between trajectories  2"""
#######################################################################################
print('--------------------------------------------------------------------------------\n')
print('[3] Meassuring Hausdorff distance between trajectories.\n')

# Distance matrix exist? exist = 1, no exist = 0
d_matrix_exist = 0

# D is a matrix containing distance between trajectories A to B and B to A
if d_matrix_exist == 1:
    D = np.load('horizontal_D.npy')
else:
    D = distances(coordinates_vec_cluster,d_matrix_exist,cluster_type)
########################################################################################
"""Clustering"""
########################################################################################
print('--------------------------------------------------------------------------------\n')
print('[4] Start clustering.\n')

# Define inputs for DBScan clustering process
if cluster_norm == 0: # Not norm
    if cluster_type == 0:
        dbscan = DBSCAN(eps=3, min_samples=30)
    elif cluster_type == 1:
        dbscan = DBSCAN(eps=2000, min_samples=30)
        # dbscan = KMeans(n_clusters=3, random_state=0)
        # dbscan =  OPTICS(min_samples=50, xi=.1, min_cluster_size=.05)

elif cluster_norm == 1: # Norm
    if cluster_type == 0:
        dbscan = DBSCAN(eps=0.1, min_samples=50)
    elif cluster_type == 1:
        dbscan = DBSCAN(eps=0.2, min_samples=30)

cluster_lst = dbscan.fit_predict(D)

# Number of clusters in labels, ignoring noise if present.
labels = dbscan.labels_

core_samples = np.zeros_like(labels, dtype=bool)
core_samples[dbscan.core_sample_indices_] = True

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
unique_labels = set(labels)

print('- Number of clusters: \n', n_clusters_)
print('- Number of noise points: \n', n_noise_)
print('- Unique labels: \n', unique_labels)  
# print('- Labels: \n', labels)
print('- Silhouette Score: %0.3f' % metrics.silhouette_score(D,labels))
print(pd.Series(labels).value_counts())

# Definition of color list for cluster representation (plots)
color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'dimgray', 'indigo', 'khaki', 'teal', 'saddlebrown', 
                 'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'olive'])


num_Hclusters = n_clusters_

########################################################################################
"""Second step clustering"""
########################################################################################

def second_step_clustering(traj_lst, cluster_lst,n_clusters):

    print(len(traj_lst))
    print(len(cluster_lst))

    clusters = defaultdict(list)

    for i in range(n_clusters):
        for traj, cluster in zip(traj_lst, cluster_lst): 

            # print(cluster_lst)
            if cluster == i:
                clusters[i].append(np.vstack(traj))
    return clusters

vclusters = second_step_clustering(coordinates_vec, cluster_lst,num_Hclusters)

pd.Series(vclusters).head()

print(pd.Series(vclusters).head())

########################################################################################
""" Meassuring Hausdorff distance between trajectories  2"""
#######################################################################################
print('--------------------------------------------------------------------------------\n')
print('[5] Meassuring Hausdorff distance between trajectories.\n')

# Distance matrix exist? exist = 1, no exist = 0
# d_matrix_exist = 0
Ds = {}
d_matrix_exist = 0

for i in range(num_Hclusters):
    Ds[i] = distances(vclusters[i],d_matrix_exist,1)
    print('Ds cluster size:', i ,len(Ds[i]))

########################################################################################
"""Clustering"""
########################################################################################
vcluster_list = {}
num_Vclusters = {}
for i in range(num_Hclusters):

    print('--------------------------------------------------------------------------------\n')
    print('[6] Start clustering.\n')

    dbscan = DBSCAN(eps=4000, min_samples=10)
        
    vcluster_list[i] = dbscan.fit_predict(Ds[i])

    # Number of clusters in labels, ignoring noise if present.
    labels = dbscan.labels_

    core_samples = np.zeros_like(labels, dtype=bool)
    # core_samples[dbscan.core_sample_indices_] = True

    num_Vclusters[i] = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    unique_labels = set(labels)

    print('- Number of clusters: \n', num_Vclusters)
    print('- Number of noise points: \n', n_noise)
    print('- Unique labels: \n', unique_labels)  
    # print('- Labels: \n', labels)
    print('- Silhouette Score: %0.3f' % metrics.silhouette_score(Ds[i],labels))
    print(pd.Series(labels).value_counts())
########################################################################################
"""Centroids"""
########################################################################################
# def centroids_lst(vclus_traj_lst, vcluster_list,num_Vclusters,num_Hclusters):

#     clusters = defaultdict(list)
#     for i in range(num_Hclusters):
#         for traj, cluster in zip(vclus_traj_lst[i], vcluster_list[i]):
#             if cluster == i:
#                 clusters[i].append(np.vstack(traj))
        
#     return clusters


# for i in range(num_Hclusters):

#     vvclusters = centroids_lst(vclusters[i], vcluster_list[i])



########################################################################################
"""Plots"""
########################################################################################
def plot_cluster(traj_lst, cluster_lst):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    
    for traj, cluster in zip(traj_lst, cluster_lst):
        
        if cluster == -1:
            # Means it it a noisy trajectory, paint it black
            plt.plot(traj[:, 1], traj[:, 2], c='k', linestyle='dashed',linewidth=0.1)
        
        else:
            plt.plot(traj[:, 1], traj[:, 2], c=color_lst[cluster % len(color_lst)],linewidth=1,alpha=0.1)

    ax.set_xlabel('$Time [s]')
    ax.set_ylabel('$Altitude [ft]$')
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")


def plot_cluster3D(traj_lst, cluster_lst):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for traj, cluster in zip(traj_lst, cluster_lst):
        
        if cluster == -1:
            # Means it it a noisy trajectory, paint it black
            ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], c='k', linestyle='dashed',linewidth=0.1)
        
        else:
            ax.plot(traj[:, 1], traj[:, 2], traj[:, 3],c=color_lst[cluster % len(color_lst)],linewidth=1,alpha=0.1)

    ax.set_xlabel('Lon [deg]')
    ax.set_ylabel('Lat [deg]')
    ax.set_zlabel('Alt [ft]')
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="z", direction="in")


fig, ax = plt.subplots(1)
plot_cluster(coordinates_vec, cluster_lst)

for i in range(num_Hclusters):
    plot_cluster3D(vclusters[i], vcluster_list[i])
plt.show()





# %%
