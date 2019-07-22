#Random Forest Regression

#importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing Data set
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values #selecting one col for x, but we dont need a vector of x, We need a Matrix.
Y = dataset.iloc[:,2].values



#NO Splitting into Train & Test due to insufficient AMOUNT of data (only 10 observations in this case)
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0 )"""

"""Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)"""
len(dataset)

#FITTING Regression Model to Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,Y)

#Predicting new result with Regression 
y_pred = regressor.predict([[6.5]])

#Vizualizing Polynomial Regression results 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'Red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'Blue')
plt.title('Random Forest Regression results')
plt.xlabel('Levels')
plt.ylabel('Salaries')
plt.show()