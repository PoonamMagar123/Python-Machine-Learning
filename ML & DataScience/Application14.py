from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

def SalesPrediction():
    
    # Load data

    data = pd.read_csv(".\ML & DataScience\Advertising.csv")

    # Print the first few rows of the dataset to understand its structure
    print(data.head())

    # Features (TV, Radio, Newspaper)
    X = data[['TV', 'radio', 'newspaper']]

    # Target variable (Sales)
    y = data['sales']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate R squared value
    r_squared = r2_score(y_test, predictions)

    # Print R squared value
    print("R squared:", r_squared)
    
    # Plotting the actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, color='blue', label='Actual values vs. Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction line')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs. Predicted Sales')
    plt.legend()
    plt.show()

def main():
    print("Supervised Machine Learning")
    
    print("Linear Regression Algorithm")
    
    SalesPrediction()
if __name__ == "__main__":
    main()