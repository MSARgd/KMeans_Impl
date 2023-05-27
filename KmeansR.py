#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:06:13 2022
@author: msa
"""
import sys
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
costs = []
def kmeans(x,k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    centroids = x[idx, :] 
    distances = cdist(x, centroids ,'euclidean') 
    points = np.array([np.argmin(i) for i in distances])
    cost = 0
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            current_cent = x[points==idx].mean(axis=0) 
            centroids.append(current_cent)
        centroids = np.vstack(centroids)
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        cost = cost + np.sum(np.min(distances, axis=1))
        costs.append(cost)
    return points ,cost

data = load_digits().data
pca = PCA(2)
df = pca.fit_transform(data)
label,cost = kmeans(df,3,1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()