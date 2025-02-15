import pandas as pd
import numpy as np

# Load the CSV file without headers
data = pd.read_csv('rls_data.csv', header=None)  # No headers in the file

# Assign column names
data.columns = ['x', 'y', 'z']

# Extract x, y, z as NumPy arrays
x = data['x'].values
y = data['y'].values
z = data['z'].values

# Print to verify data
print("First few rows of the dataset:")
print(data.head())

# Step 1: Create the design matrix X
m = len(x)  # Number of data points
X = np.column_stack([x**2, y**2, x * y, x, y, np.ones(m)])  # Design matrix

# Step 2: Solve for coefficients using the normal equation
XtX = np.dot(X.T, X)  # X^T * X
XtZ = np.dot(X.T, z)  # X^T * Z
a = np.linalg.solve(XtX, XtZ)  # Solve for a = (X^T * X)^(-1) * X^T * Z

# Output the coefficients
print("\nCoefficients (a1, a2, a3, a4, a5, a6):")
print(a)

# Step 3: Use a built-in library to compare results (Statsmodels)
import statsmodels.api as sm

# Fit the model using statsmodels
model = sm.OLS(z, X).fit()
print("\nStatsmodels Coefficients:")
print(model.params)

# Step 4: Compare the results
print("\nComparison of Coefficients:")
print(f"Custom Implementation: {a}")
print(f"Statsmodels:           {model.params}")

# Optional: Calculate residuals and R^2 for evaluation
z_pred = np.dot(X, a)  # Predictions from custom implementation
residuals = z - z_pred
r_squared = 1 - (np.sum(residuals**2) / np.sum((z - np.mean(z))**2))

print(f"\nR-squared (Custom Implementation): {r_squared:.4f}")

