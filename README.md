# Predicting-Exam-Score-using-Regression-Model-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Prepare the dataset (Hours of Study vs. Marks Scored)
data = {
    'Hours': [1.1, 2.5, 3.2, 4.5, 5.1, 6.0, 7.4, 8.5, 9.2, 10.3],
    'Marks': [35, 45, 50, 55, 60, 70, 75, 85, 90, 95]
}
df = pd.DataFrame(data)

# Step 2: Split data into features (X) and target (y)
X = df[['Hours']]  # Independent variable
y = df['Marks']    # Dependent variable

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 7: Predict marks for a specific number of study hours
hours = 7.0
# Use a DataFrame with the same column name as the training data
predicted_marks = model.predict(pd.DataFrame([[hours]], columns=['Hours']))
print(f"Predicted marks for {hours} hours of study: {predicted_marks[0]:.2f}")

# Step 8: Visualize the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours of Study')
plt.ylabel('Marks Scored')
plt.title('Linear Regression: Hours vs. Marks')
plt.legend()
plt.show()

Output:
Mean Squared Error: 1.8829663606899607
R-squared: 0.9962805602751803
Predicted marks for 7.0 hours of study: 73.77
<Figure size 640x480 with 1 Axes>
