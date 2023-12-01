import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def housing_price_dataset():
    # Load data
    data = pd.read_csv(r"E:\MARVELLOUS INFOSYSTEM\ML & DataScience\housing_price_dataset.csv")

    # Drop the "Neighborhood" column
    data.drop("Neighborhood", axis=1, inplace=True)

    # Print the first few rows of the dataset to understand its structure
    print(data.head())

    # Features (SquareFeet, Bedrooms, Bathrooms, YearBuilt)
    X = data[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt']]

    # Target variable (Price)
    y = data['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Calculate R squared value and MAE
    r_squared = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Print evaluation metrics
    print("R squared:", r_squared)
    print("Mean Absolute Error:", mae)

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
    housing_price_dataset()

if __name__ == "__main__":
    main()
