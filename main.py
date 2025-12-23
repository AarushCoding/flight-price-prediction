#Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Load dataset
data = pd.read_csv('Clean_Dataset.csv')

#Create a Linear Regression model
model = LinearRegression()

#Split dataset into features and target variable
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

#Apply OneHotEncoding to categorical features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3,4,5,6,7])], remainder='passthrough')
X = ct.fit_transform(X)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Train the model

model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
score = r2_score(y_test, y_pred)
print(f'r2 Score: {score}')

