###################################
# Required python Packages
###################################
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

###################################
# File Path
###################################

INPUT_PATH = "ML & DataScience\breast-cancer-wisconsin.data"
OUTPUT_PATH = ".\ML & DataScience\breast-cancer-wisconsin.csv"

###################################
# Headers
###################################

HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses", "CancerType"]

###################################
# Function Name : read_data
# Description : Read the data into pandas dataframe
# Input : path of csv file
# Output : Gives the data
# Author : Poonam Dhanaji Magar
# Date : 29/10/2013
###################################
def read_data(path):
    data = pd.read_csv(path)
    return data
###################################
# Function Name : get_headers
# Description : Dataset headers
# Input : dataset
# Output : Returns the header
# Author : Poonam Dhanaji Magar
# Date : 29/10/2013
###################################
def get_headers(dataset):
    return dataset.columns.values

###################################
# Function Name : add_headers
# Description : Add the headers to the Dataset 
# Input : dataset
# Output : Update dataset
# Author : Poonam Dhanaji Magar
# Date : 29/10/2013
###################################

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

###################################
# Function Name : data_file_to_csv
# Input : Nothing
# Output : Write the data to csv
# Author : Poonam Dhanaji Magar
# Date : 29/10/2013
###################################

def data_file_to_csv():
    # Headers
    
    Headers = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses", "CancerType"]
    # load the dataset into Pandas data frame
    dataset = read_data(INPUT_PATH)
    # Add the headers to the loaded dataset
    #Save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("File saved ...!")
    
    ###################################
    # Function Name : split_dataset
    # Description : Split the dataset with train_percentage
    # Input : Dataset with related information
    # Output : Dataset after splitting
    # Author : Poonam Dhanaji Magar
    # Date : 29/10/2013
    ###################################
     
    def split_dataset(dataset, train_percentage, feature_headers, target_header):
        # Split dataset into train and test datset
        train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
        return train_x, test_x, train_y, test_y 
        
        
    ###################################
    # Function Name : handle_missing_values
    # Description : Filter missing values from the dataset
    # Input : dataset the missing values
    # Output : Dateset by removing missing values
    # Author : Poonam Dhanaji Magar
    # Date : 29/10/2013
    ###################################
    def handle_missing_values(dataset, miissing_values_header, missing_label):
        return dataset[dataset[miissing_values_header] != missing_label]
    
    
    ###################################
    # Function Name : random_forest_classifier
    # Description : To the train the random forest classifier with features and target data
    # Author : Poonam Dhanaji Magar
    # Date : 29/10/2013
    ###################################
    def random_forest_classifier(features, target):
        clf = RandomForestClassifier()
        clf.fit(feature, target)
        return clf
    
    ###################################
    # Function Name : main
    # Description : Main Function from where execution starts
    # Author : Poonam Dhanaji Magar
    # Date : 29/10/2013
    ###################################
    
def main():
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    # Get basic statistic of the loaded dataset
    dataset_statistic(dataset)
        
        # Filter miising values
    dataset = handle_missing_values(dataset, HEADERS[6], '?')
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1],HEADERS[-1])
        
    # train and Test dataset size details
    print("Train_x Shape :: ",train_x.shape)
    print("Train_y Shape :: ",train_y.shape)
    print("Test_x Shape :: ",test_x.shape)
    print("Test_y Shape :: ",test_y.shape)
        
        # trained random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained Model :: ", trained_model)
    prediction = trained_model.predict(test_x)
        
    for i in range(0, 205):
        print("Actual Outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], prediction))
            
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y, prediction))
    print("Confusion Matrics :: ", confusion_matrix(test_y, prediction))
        
    ###################################
    # Application starter
    ###################################
if __name__ == "__main__":
    main()
        