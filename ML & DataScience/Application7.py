import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):
    
    #step 1 : Load data
    data = pd.read_csv(data_path, index_col=0)
    
    print("Size of actual dataset : ", len(data))
    
    #step2 : Clean, preapare and manipulate data
    feature_names  = ['Whether','Temperature']
    
    print("Names of features : ",feature_names)
    
    weather = data.Whether
    Temprature = data.Temperature
    Play = data.Play
    
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    
    #Converting string label into numbers
    weather_encoded = le.fit_transform(weather)
    print(weather_encoded)
    
    #Converting string label into numbers
    temp_encoded = le.fit_transform(Temprature)
    label = le.fit_transform(Play)
    print(temp_encoded)
    
    #combining weather and temp into single list of tuple
    features = list(zip(weather_encoded,temp_encoded))
    
    #step 3 : Train Data
    model = KNeighborsClassifier(n_neighbors=3)
    
    #train the model using the training sets
    model.fit(features,label)
    
    #step 4 : Test Data
    predicted = model.predict([[0,2]]) # 0: overcast, 2 : Mild
    print(predicted)

def main():
    print("Machine Learning Application")
    
    print("Play Predictor Application using K Nearest Knighour algorithm")
    
    MarvellousPlayPredictor("ML & DataScience\PlayPredictor.csv")
    
if __name__ == "__main__":
    main()
        
    