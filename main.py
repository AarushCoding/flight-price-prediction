import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('Clean_Dataset.csv')
model = LinearRegression()

X = data.iloc[:, 1:11] 
y = data.iloc[:, -1]


ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0,1,2,3,4,5,6,7])],remainder='passthrough')
X = np.array(ct.fit_transform(X).toarray())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model.fit(X_train, y_train)
