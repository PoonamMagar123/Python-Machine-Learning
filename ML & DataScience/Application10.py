import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def MArvellousTitanicLogistic():
    # Step 1 : Load data
    titanic_data = pd.read_csv('.\ML & DataScience\MarvellousTitanicDataset.csv')
    
    print("First 5 from loaded dataset")
    print(titanic_data.head())
    
    print("Number of passengers are "+str(len(titanic_data)))
    sex = titanic_data.Sex
    pclass = titanic_data.Pclass
    # Play = titanic_data.Play


    # step 2 : Analyze data
    print("Visualization : Survived and non survied passangers")
    figure()
    target = "Survived"
    
    countplot(data=titanic_data, x=target).set_title("Marvellous Infosystem : Survived and non survied passangers")
    show()
    
    print("Visualization : Survived and non survied passangers based on Gender")
    figure()
    target = "Survived"
    
    countplot(data=titanic_data, x=target, hue="Sex").set_title("Marvellous Infosystem : Survived and non survied passangers based on Gender")
    show()
     
    print("Visualization : Survived and non survied passangers based on class")
    figure()
    target = "Survived"
    
    countplot(data=titanic_data, x=target, hue="Pclass").set_title("Visualization : Survived and non survied passangers based on the Passanger class")
    show()
    
    print("Visualization : Survived and non survied passangers based on age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystem : Survived and non survied passangers based on age")
    show()
    
    print("Visualization : Survived and non survied passangers based on Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Visualization : Survived and non survied passangers based on Fare")
    show()
    
    #step 3 : Data Cleaning
    titanic_data.drop("zero", axis=1, inplace=True)
    
    print("First 5 entries from loaded dataset after removing zero culumn")
    print(titanic_data.head(5))
    
   
    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    #Converting string label into numbers
    weather_encoded = le.fit_transform(sex)
    print(weather_encoded)
    
    #Converting string label into numbers
    temp_encoded = le.fit_transform(pclass)
    # label = le.fit_transform(Play)
    print(temp_encoded)
    
    print("Values of sex culumn")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("values of Sex culumn after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    print(Sex.head(5))
    
    print("Values of Pclass culumn after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    print(Pclass.head(5))
    
    print("Values of data set after concatenating new culumns")
    titanic_data = pd.concat([titanic_data, Sex,Pclass],axis=1)
    print(Pclass.head(5))
    
    print("Values of data set after removing irrelevent culumns")
    titanic_data.drop(["Sex", "sibsp", "Parch", "Embarked"], axis=1)
    print(titanic_data.head(5))
    
    x = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]
    
    #step 4 : Data training
    xtrain , xtest, ytrain, ytest = train_test_split(x,y,test_size=0.5)
   # Convert column names to strings
    xtrain.columns = xtrain.columns.astype(str)

    logmodel = LogisticRegression()
    logmodel.fit(xtrain, ytrain)

    #step 4 : Data testing
    prediction = logmodel.predict(xtest)
    
    #step 5 : Calculate Accuracy
    print("Classification report of logistic Regression is  : ")
    print(classification_report(ytest, prediction))
    
    print("Confusion Matrix of logistic Regression is : ")
    print(confusion_matrix(ytest,prediction))
    
    print("Accuracy of Logistic Regression is : ")
    print(accuracy_score(ytest,prediction))
    
def main():
    print("Supervised Machine Learning")

    print("Logistic Regression on Titanic data set")
    
    MArvellousTitanicLogistic()
    
if __name__ == "__main__":
    main()
