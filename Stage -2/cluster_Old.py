'''
Clustering algorithm to reduce the number of load patterns

Date of Creation : 02.09.2019 				Last Modified : 02.09.19

Author : Vasudevan B.               Mailid: vasubdevan@yahoo.com

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io as sio 
from sklearn import preprocessing

from similaritymeasures import Similarity

def ReadingData():
    data = pd.read_csv("Config.csv") 
    #print(data.head())
    #print(data.columns) # printing the column name
    ND = data.values[:,0:66]
    return ND

def DetermineCentroid(Centeroids, ND):
    mean = np.mean(ND, axis = 0)
    std = np.std(ND, axis = 0)
    Centeroids = (Centeroids*std) + mean
    return Centeroids

def AssignCentroids(Centroids, ND, k, c, n):
    newArray = np.random.randint(0, n, size=(k))
    print(newArray)
    for x in range(k):
        Centroids[x,:] = ND[newArray[x],:]
    return Centroids

def ComputeDist(Dist, ND, Centeroids, k, measures):
    for nc in range(k):
        A = Centeroids[nc]
        #print(A)
        for i in range(ND.shape[0]):   #ND.shape[0] # for all load combinations
            B = ND[i]
            #print(B)
            maxvar = 0;
            maxvar = measures.euclidean_distance(A, B)

            Dist[i][nc] = maxvar

            #print("EucDist[", i,"][",nc,"] =", EucDist[i][nc])
    return Dist

def deepcopy(centers_new, Centeroids):
    for i in range(Centeroids.shape[0]):
        centers_new[i,:] = Centeroids[i,:]
    return centers_new

def main():    
    ND = ReadingData()
    # print(ND.shape) #print(ND)
    measures = Similarity()
    
    k, n, c = 22, ND.shape[0], ND.shape[1]  # Assigning variables
    #print("Number of clusters = ", k, "No of training data are ", n, "and features are ", c)

    dist = 'Euclidean'
    Outfile1 = "Intial_{}_{}.csv".format(k, dist)
    Outfile2 = "Centroid_{}_{}.csv".format(k, dist)
    Outfile3 = "Cluster_{}_{}.mat".format(k, dist)

    '''
    # One way of obtaining Centroids
    Centeroids = np.random.randn(k,c)
    #print(Centeroids) 
    Centeroids = DetermineCentroid(Centeroids, ND)
    #print(Centeroids)
    '''

    # Another way to obtain initial centroid
    Centroids = np.zeros([k,c])
    Centeroids = AssignCentroids(Centroids, ND, k, c, n)
    #print(Centroids)

    np.savetxt(Outfile1, Centeroids, fmt='%.4f', delimiter=',', header=" P1, Q1, P2, Q2, P3, Q3, P4, Q4, P5, Q5, P6, Q6, P7, Q7, P8, Q8, P9, Q9, P10, Q10, P11, Q11,  P12, Q12,  P13, Q13, P14, Q14,P15, Q15, P16, Q16, P17, Q17, P18, Q18, P19, Q19, P20, Q20, P21, Q21, P22,  Q22, P23, Q23, P24, Q24, P25, Q25, P26, Q26, P27, Q27, P28, Q28, P29, Q29, P30, Q30, P31, Q31, P32, Q32, P33, Q33")
     
    Dist = np.zeros([n, k])
    #print(Dist.shape)
                
    centers_old = np.zeros(Centeroids.shape) # to store old centers
    clusters = np.zeros(n)
    #print(clusters)
    
    error = np.linalg.norm(Centeroids - centers_old)
    print("Initial error value is computed as ", error)
    
    
    while error != 0:
        Dist = ComputeDist(Dist, ND, Centeroids, k, measures)
        #print("Euclidean distances is as follows : \n", EucDist)
        
        clusters = np.argmin(Dist, axis = 1)
        #print("Clusters is as follows : \n", clusters)
        
        centers_old = deepcopy(centers_old, Centeroids)
        #print(centers_old)
        
        for i in range(k):
            Centeroids[i] = np.mean(ND[clusters == i], axis=0)
            #print(Centeroids[i])
        error = np.linalg.norm(Centeroids - centers_old)
        print("\nError is computed as ", error)
    np.savetxt(Outfile2, Centeroids, fmt='%.4f', delimiter=',', header=" P1, Q1, P2, Q2, P3, Q3, P4, Q4, P5, Q5, P6, Q6, P7, Q7, P8, Q8, P9, Q9, P10, Q10, P11, Q11,  P12, Q12,  P13, Q13, P14, Q14,P15, Q15, P16, Q16, P17, Q17, P18, Q18, P19, Q19, P20, Q20, P21, Q21, P22,  Q22, P23, Q23, P24, Q24, P25, Q25, P26, Q26, P27, Q27, P28, Q28, P29, Q29, P30, Q30, P31, Q31, P32, Q32, P33, Q33")

    # Syntax to save it as a mat file    
    adict = {}
    adict['clusters'] = clusters
    sio.savemat(Outfile3, adict)
    
#----- Main Program strats here -----#
if __name__ == '__main__':
    main()