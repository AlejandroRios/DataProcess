"""" 
Function  : Distances matrix
Title     : distances_matrix
Written by: Alejandro Rios
Date      : February/2020
Language  : Python
Aeronautical Institute of Technology - Airbus Brazil
"""
########################################################################################
"""Importing Modules"""
########################################################################################
import numpy as np
from cluster.hausdurff_distance import cmax
########################################################################################
""" Meassuring Hausdorff distance between trajectories  2"""
#######################################################################################

def hausdorff(A,B):
    H_distance = max(cmax(A, B),cmax(B,A))
    return(H_distance)

def distances(coordinates_vec,d_matrix_exist,cluster_type):
    print('Meassuring Hausdorff distance between trajectories.\n')

    # coordinates_vec=coordinates_vec[1][:,[1,2]]
    traj_count = len(coordinates_vec)
    D = np.zeros((traj_count, traj_count))

    # This may take a while
    for i in range(traj_count):
        for j in range(i + 1, traj_count):
            if cluster_type == 0:
                distance = hausdorff(coordinates_vec[i][:,[1,2]], coordinates_vec[j][:,[1,2]])
            elif cluster_type == 1:
                distance = hausdorff(coordinates_vec[i][:,[3]], coordinates_vec[j][:,[3]])
    
            D[i, j] = distance
            D[j, i] = distance

    if (d_matrix_exist == 0) & (cluster_type == 0):
        np.save('horizontal_D.npy', D)
        
        
    print('End.\n')

    return(D)

