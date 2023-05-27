#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:06:13 2022
@author: msa
"""
import numpy as np
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import pandas as pd
def kmeans(x,k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    centroids = x[idx, :] 
    distances = cdist(x, centroids ,'euclidean') 
    points = np.array([np.argmin(i) for i in distances])
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            current_cent = x[points==idx].mean(axis=0) 
            centroids.append(current_cent)
        centroids = np.vstack(centroids)
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
    return points 
df = pd.read_excel('file.xlsx')
df = np.asarray(df[['Age','Income($)']])
label = kmeans(df,3,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()