#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the iris dataset with pandas
dataset = pd.read_csv("E:\MARVELLOUS INFOSYSTEM\ML & DataScience\iris.csv") 

x = dataset.iloc[:, [0,1,2,3]].values

#Finding the optimum number of clusters for k-means classification 
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    Kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The Elbow'

plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS') # within cluster sum of square
plt.show()

#Applying kmeans to the dataset / creating the kmens classifier
Kmeans = KMeans(n_clusters = 3, init='k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = Kmeans.fit_predict(x)

# visualising the cluster
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1], s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Iris-Versicolor')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,1], s = 100, c = 'green', label = 'Iris-Virginica')

plt.scatter(Kmeans.cluster_centers_[:,0], Kmeans.cluster_centers_[:,1], s = 100, c ='yellow', label = 'Centroids')

plt.legend()

plt.show()
