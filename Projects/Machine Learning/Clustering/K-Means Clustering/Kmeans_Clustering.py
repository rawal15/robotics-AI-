# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:42:34 2017

@author: shahik
"""

#Importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the mall dataset with pandas
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Using the elbow method to find the exact number of clusters
from sklearn.cluster import KMeans
wcss=[] #list
for i in range(1,11): #loop for 10 clusters
      kmeans=KMeans(n_clusters=i,init= 'k-means++',max_iter=300,n_init= 10,random_state=0)
      kmeans.fit(X)
      wcss.append(kmeans.inertia_) # clusters sum of squares computing and adding in list

plt.plot(range(1,11),wcss)  # X-axis and y-axis values
plt.title('The ELbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the dataset after seeing from elbow
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X) #returns clusters and customer belongings

#Visualising the clusters in 2D
#Clusters number are from 0-4 and giving x & y coordinates of clusters(0 and 1 for columns from X-data), s=size
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Anual income (k$)')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()


