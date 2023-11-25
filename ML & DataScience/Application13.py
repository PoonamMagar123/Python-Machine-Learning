import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from seaborn import countplot
from matplotlib.pyplot import figure, show

def SalesPrediction():
    #lode the data
    df = pd.read_csv("E:\MARVELLOUS INFOSYSTEM\ML & DataScience\Advertising.csv")
    
    print(df.shape) 
    
    # step 2 : Analyze data
    print("Visualization : Advertizement add show in TV")
    figure()
    target = "TV"
    
    countplot(data=df, x=target).set_title("TV Shows")
    show()
    
    print("Visualization : Advertizement add show in radio")
    figure()
    target = "radio"
    
    countplot(data=df, x=target).set_title("radio Shows")
    show()
    
    print("Visualization : Advertizement add show in newspaper")
    figure()
    target = "newspaper"
    
    countplot(data=df, x=target).set_title("newspaper Shows")
    show()
    
    #
    print("Visualization : Survived and non survied passangers based on age")
    figure()
    df["TV"].plot.hist().set_title("Marvellous Infosystem : Survived and non survied passangers based on age")
    show()
    
    print("Visualization : Survived and non survied passangers based on age")
    figure()
    df["radio"].plot.hist().set_title("Marvellous Infosystem : Survived and non survied passangers based on age")
    show()
    
    print("Visualization : TV")
    figure()
    df["TV"].plot.hist().set_title("Marvellous Infosystem : TV")
    show()
    
    print("Visualization : radio")
    figure()
    df["radio"].plot.hist().set_title("Marvellous Infosystem : radio")
    show()
    
    print("Visualization : newspaper")
    figure()
    df["newspaper"].plot.hist().set_title("Marvellous Infosystem : newspaper")
    show()
    
def main():
    print("Supervised Machine Learning")
    
    print("Linear Regression Algorithm")
    
    SalesPrediction()
if __name__ == "__main__":
    main()


