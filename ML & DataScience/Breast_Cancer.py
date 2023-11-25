from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def MarvellousSVM():
    #Load Dataset
    cancer = datasets.load_breast_cancer()

    # print the names of the 13 features
    print("Features of the cancer datset : ",cancer.feature_names)

    # print the label type of cancer
    print("Labels of the cancer dataset", cancer.target_names)

    # print data(feature)shape
    print("Shape of datset is : ", cancer.data.shape)
    
    # print the cancer Labels (0:malignant, 1:benign)
    print("First 5 records are : ")
    print(cancer.data[0:5])

    #print the cancer labels (0:malignant, 1:benign)
    print("Target of dataset : ", cancer.target)

    # Split dataset into traning set and test set
    X_train, X_test, Y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.3, random_state = 109) # 70% training and 30% test

    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, Y_train)

    #predict the respnce for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy : how often is the classifier correct?
    print("Accuracy of the model is :",metrics.accuracy_score(y_test, y_pred)*100)
def main():

    print("Machine Learning Application")
    
    print("breast cancer Application")
    
    MarvellousSVM()
if __name__ == "__main__":
    main()
        