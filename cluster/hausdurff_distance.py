"""" 
Function  : General Hausdorff distance calculator
Title     : Hausdorff_Distance
Written by: Alejandro Rios
Date      : February/2020
Language  : Python
Aeronautical Institute of Technology
"""
'Modules'
import numpy as np
import time
import numba
from math import sqrt, pow, cos, sin, asin



'Euclidean distance calculation'

@numba.jit(nopython=True, fastmath=True)
def euclidean(A,B):
    n = A.shape[0]
    ret = 0.
    for i in range(n):
        ret += (A[i]-B[i])**2

    distance = sqrt(ret)
    return distance



'direct hausdorff calculation' 
@numba.jit(nopython=True, fastmath=True)
def cmax(A,B):   
    n = len(A)
    cmax = 0
    for i in range(n):
        cmin = np.inf
        for j in range(n):
            d = euclidean(A[j,:],B[i,:])
            if d < cmax:
                cmin = 0
            cmin = min(cmin,d)
        cmax = max(cmax,cmin)

    return(cmax)



# '#################################################################'
# 'Test - uncomment'

# start = time.clock() 

# n = 100
# 'Ex'
# 'Curve 1'
# t = np.linspace((3/4)*np.pi, (5/4)*np.pi, n)
# x1 = np.cos(t)
# y1 = np.sin(t)
# z1 = t
# 'Curve 2'
# u = np.linspace((1/2)*np.pi, (3/2)*np.pi, n)
# x2 = 2*np.cos(u)
# y2 = np.sin(u)
# z2 = 2*u

# A = np.column_stack((x1,y1,z1))
# B = np.column_stack((x2,y2,z2))
# end = time.clock()  



# print(hausdorff(A, B))

# print("Processor time (in seconds):", end) 
