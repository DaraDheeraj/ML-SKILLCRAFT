import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Manually create the data (without using a file)
data = {
    'square_footage': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'bathrooms': [2, 3, 2, 3, 3],
    'price': [400000, 500000, 600000, 700000, 750000]
}

# Create a DataFrame from the manual data
df = pd.DataFrame(data)

# Select the features (X) and target variable (y)
X = df[['square_footage', 'bedrooms', 'bathrooms']]  # Features
y = df['price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Show the predicted prices
print(f'Predicted Prices: {y_pred}')
