# The aim of this capstone project is to provide a foundation
# for understanding and applying linear algebra concepts
# in data science and machine learning.

# Using the Python programming language and relevant libraries,
# the project will enable you to develop
# a data science workflow that incorporates concepts
# such as linear algebra, basic statistics, dimensionality reduction, and linear regression.

# Necessary Libraries:
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Dataset
df_temp = load_diabetes()
X = pd.DataFrame(df_temp.data, columns=df_temp.feature_names)
Y = pd.Series(df_temp.target, name = "target")

# 2. Data Exploration and Cleaning
if X.isnull().sum().sum() == 0:
    print("There are no missing values in the dataset.")
else:
    print("\nMissing values detected in the dataset. They have been filled with the mean of the respective feature.")
    X.fillna(X.mean(), inplace=True)

# 3. Dimension Reducing with PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

X_dimension_reduced = pd.DataFrame(X_pca, columns=[f"Component {i+1}" for i in range(5)])

# 4. Building a Linear Regression Model
X_train, X_test, Y_train, Y_test = train_test_split(X_dimension_reduced, Y, test_size=0.2, random_state=42)
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, Y_train)

Y_pred = linear_reg_model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R^2 Score: {r2}")

# MSE: 2879.5924195103134
# MAE: 43.289479831575356
# R^2 Score: 0.45649113951548226

coefficients = linear_reg_model.coef_
intercept = linear_reg_model.intercept_

for i, coef in enumerate(coefficients):
    print(f"Bileşen {i + 1}: {coef}")

# Bileşen 1: 445.5647494929188
# Bileşen 2: -287.03569005999185
# Bileşen 3: 264.72916295779544
# Bileşen 4: -576.2674899829929
# Bileşen 5: 38.16443426984986

print("\nIntercept:", intercept)
# Intercept: 151.18203998537038