from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import StackingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#doc du lieu
salary = pd.read_csv()
# kiem tra rong
print(salary.isnull().sum())
#ktra trung lap 
d = salary.duplicated()
print("Du lieu loa bo", d.sum())

X = salary[['ex']].values
y= salary[['salary']].values

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

scaler_X= StandardScaler()
scaler_y = StandardScaler()
X_train_scaler= scaler_X.fit_transform(X_train)
X_test_scaler = scaler_X.transform(X_test)
y_train_scaler  = scaler_y.fit_transform(y_train)
#hoi quy tuyen tinh
linear_model = LinearRegression()
linear_model.fit(X_train,y_train.ravel())
y_pred_linear = linear_model.predict(X_test)
mae_linear= mean_absolute_error(y_test,y_pred_linear)
mse_linear = mean_squared_error(y_test,y_pred_linear)
r2_linear = r2_score(y_test,y_pred_linear)

#lasso
lasso_model = Lasso()
lasso_model.fit(X_train,y_train.ravel())
y_pread_lasso = lasso_model.predict(X_test)
mae_lasso = mean_absolute_error(y_test,y_pread_lasso)
mse_lasso = mean_squared_error(y_test,y_pread_lasso)
r2_lasso = r2_score(y_test,y_pread_lasso)


#Neural 
scaler_X= StandardScaler()
scaler_y = StandardScaler()
X_train_scaler= scaler_X.fit_transform(X_train)
X_test_scaler = scaler_X.transform(X_test)
y_train_scaler  = scaler_y.fit_transform(y_train)
mlp_model = MLPRegressor()
mlp_model.fit(X_train_scaler,y_train_scaler.ravel())
y_pred_mlp_scaler = mlp_model.predict(X_train_scaler)
y_pred_mlp = scaler_y.inverse_transform(-1,1)
mae_mlp = mean_absolute_error(y_test,y_pred_mlp)
mse_mlp = mean_squared_error(y_test,y_pred_mlp)
r2_mlp = r2_score(y_test,y_pred_mlp)

