import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def checkAccuracy(data_path):
    
    #step 1 : Load data
    data = pd.read_csv(data_path)
    
    print("Size of actual dataset", len(data))
    
    # #step2 : Clean, preapare and manipulate data
    feature_names = data[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                     'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                     'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']]
    
    labels = data['Class']  # Assuming your target variable is 'Class'
    
    data_train, data_test, target_train, target_test = train_test_split(feature_names, labels, test_size=0.3, random_state=66)
    
    classifier = KNeighborsClassifier(n_neighbors=3)
    
    classifier.fit(data_train,target_train)
    
    predictions = classifier.predict(data_test)
    
    Accuracy = accuracy_score(target_test,predictions)
    
    accuracy_scores = []
    for k in range(1, 11):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(feature_names, labels)
        predictions = knn_model.predict(feature_names)
        accuracy = accuracy_score(labels, predictions)
        accuracy_scores.append(accuracy*100)
    
    return Accuracy, accuracy_scores
    
def main():
    print("Machine Learning Application")
    
    print("Wine Predictor Application")
    
    Accuracy, accuracy_scores = checkAccuracy("E:\MARVELLOUS INFOSYSTEM\ML & DataScience\WinePredictor (2).csv")
    
    print("Accuracy of classification algorithm with K Neighbor 3 is {:.2f}%".format(Accuracy * 100))
    
    print("Accuracy of  classification algorithms with K Neighbor first 10 is", ", ".join("{:.2f}%".format(score) for score in accuracy_scores))

if __name__ == "__main__":
    main()
        
    