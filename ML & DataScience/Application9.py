import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def loadData(data_path):
    # Step 1: Get Data
    data = pd.read_csv(data_path)
    return data

def prepareData(data):
    # Step 2: Clean, Prepare and Manipulate Data
    # Assuming your features are in columns 'Alcohol' to 'Proline'
    features = data[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                     'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                     'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']]
    
    labels = data['Class']  # Assuming your target variable is 'Class'
    
    return features, labels

def trainData(features, labels):
    # Step 3: Train Data
    # Split the data into 70% training and 30% testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Select the Machine Learning algorithm - K Nearest Neighbors
    model = KNeighborsClassifier(n_neighbors=3)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def testData(model, X_test):
    # Step 4: Test Data
    # Test the trained data with some sample values
    # For example, you can use X_test to make predictions
    predictions = model.predict(X_test)
    
    return predictions

def checkAccuracy(model, features, labels):
    # Step 5: Calculate Accuracy
    # Calculate accuracy for different values of K (neighbors)
    accuracy_scores = []
    for k in range(1, 11):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(features, labels)
        predictions = knn_model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        accuracy_scores.append(accuracy)
    
    return accuracy_scores 

def main():
    print("Wine Predictor Application")
    
    # Load Data
    data_path = "ML & DataScience\WinePredictor.csv"  # Specify the correct path to your data file
    data = loadData(data_path)
    
    # Prepare Data
    features, labels = prepareData(data)
    
    # Train Data
    model, X_test, y_test = trainData(features, labels)
    
    # Test Data
    predictions = testData(model, X_test)
    
    # Calculate Accuracy
    accuracy_scores = checkAccuracy(model, features, labels)
    
    print("Predictions:", predictions)
    print("Accuracy Scores for different values of K:", accuracy_scores, "%")

if __name__ == "__main__":
    main()
