from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def prep_data(csv):
    x = csv[['number_rooms', 'living_area',
       'terrace', 'terrace_area', 'garden',
       'garden_area', 'surface_land', 'number_facades',
       'property_type', 'building_state', 'kitchen', 'region', 'digit']]

    x = pd.get_dummies(data=x, drop_first=True)
    X = x.to_numpy()
    y = csv['price'].to_numpy() 

    # Split the data into test and training sets and scale it
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, y

def train_LinearRegression(X_train, y_train): 
    """Initializes the model"""
    return LinearRegression().fit(X_train, y_train)

def train_DecisionTreeRegressor(X_train, y_train): 
    """Initializes the model"""
    return DecisionTreeRegressor(criterion='squared_error', max_depth=12, min_weight_fraction_leaf=0.0045).fit(X_train, y_train)

def train_XGBRegressor(X_train, y_train): 
    """Initializes the model"""
    return XGBRegressor(objective ='reg:squarederror', n_estimators = 50, seed = 123).fit(X_train, y_train)

def train_SGDRegressor(X_train, y_train): 
    """Initializes the model"""
    return SGDRegressor(max_iter=1000, tol=1e-3).fit(X_train, y_train)

def score(regressor, X_train, X_test, y_train, y_test, y):
    score_train = regressor.score(X_train, y_train)
    score_test = regressor.score(X_test, y_test)

    # Get the root mean squared error
    y_pred = regressor.predict(X_test)
    #rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
    rmse= np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))

    u = ((y_test - y_pred)**2).sum()
    v = ((y_test - y.mean())**2).sum()
    coef_determination = 1 - u/v

    return score_train, score_test, rmse, coef_determination