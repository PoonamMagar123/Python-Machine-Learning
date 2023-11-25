import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

def MarvellousKMean():
    
    print("______________________________")
    # set three centers , the model should predict similar results
    center_1 = np.array([1,1])
    print(center_1)
    
    print("______________________________")
    center_2 = np.array([5,5])
    print(center_2)
    
    print("______________________________")
    center_3 = np.array([8,1])
    print(center_3)
    
    print("______________________________")
    
    # Generate random data and center it to the three centers
    data_1 = np.random.randn(7,2) + center_1
    print("Elements of first cluster with size "+str(len(data_1)))
    print(data_1)
    
    print("______________________________")
    data_2 = np.random.randn(7,2) + center_2
    print("Elements of second cluster with size "+str(len(data_1)))
    print(data_2)
    
    print("______________________________")
    data_3 = np.random.randn(7,2) + center_3
    print("Elements of first cluster with size "+str(len(data_1)))
    print(data_3)
    
    
    print("______________________________")
    data = np.concatenate((data_1, data_2, data_3), axis=0)
    print("Size of complte data set"+str(len(data)))
    print(data)
    
    print("______________________________")
    plt.scatter(data[:,0], data[:,1], s=7)
    plt.title("Marvellous Infosytem : Input Dataset")
    plt.show()
    
    print("______________________________")
    # Number of clusters
    k = 3
    
    # number of training data
    
    n = data.shape[0]
    print("Total Number of elements are",n)
    print("______________________________")
    
    #Number of feature in the data
    c = data.shape[1]
    print("Total Number of elements are",c)
    print("______________________________")
    
    # Generate random centers , here we use sigma and mean to ensure it represent the whole data
    
    mean  =  np.mean(data, axis = 0)
    print("Value of mean ", mean)
    print("______________________________")
    
    # Calculate standard deviation
    
    std = np.std(data, axis=0)
    print("Value of std", std)
    
    print("______________________________")
    centers = np.random.randn(k,c)*std + mean
    print("Random points are", centers)
    print("______________________________")
    #Plot the data and the centers generated as random
    
    plt.scatter(data[:,0], data[:,1],c='r', s=7)
    plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
    plt.title("Marvellous Infosystem : input dataset with random centroid *")
    plt.show()
    print("______________________________")
    
    center_old = np.zeros(centers.shape) # to store old centers
    center_new = deepcopy(centers)       # Store new centers
    
    print("Values of old centroids")
    print(center_old)
    print("______________________________")
    
    print("Values of new centroids")
    print(center_new)
    print("______________________________")
    
    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n,k))
    
    print("Initial distances are")
    print(distances)
    print("______________________________")
    
    error = np.linalg.norm(center_new - center_old)
    print("Value of error is ", error)
    
    #When , after an update , the estimate of that center stays the same, exit loop
    
    while error != 0:
        print("Value of error is ", error)
        # Measure the distance to every center
        print("Measure the distance to every center")
        for i in range(k):
            print("Iteration Number ",i)
            distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
            
            # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)
            
        centers_old = deepcopy(center_new)
            
            # Calculate mean for every cluster and update the center
            
        for i in range(k):
            center_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(center_new - center_old)
    # end of while
    center_new
        
    # plot the data and the centers generated as random
    plt.scatter(data[:,0], data[:,1] , s=7)
    plt.scatter(center_new[:,0], center_new[:,1], marker='*' ,c='g', s=150 )
    plt.title("Marvellous Infosystem : Final data with Centriod")
    plt.show()
        
def main():
    print("UnSupervised Machine Learning")
    
    print("Clustering using K mean Algorithm")
    
    MarvellousKMean()
if __name__ == "__main__":
    main()
