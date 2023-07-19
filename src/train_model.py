from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def prep_data_relational(csv):
    """Takes a Pandas DataFrame as input, will return X and y values"""
    X = csv[['number_rooms', 'living_area',
       'furnished', 'fireplace', 'terrace', 'terrace_area', 'garden',
       'garden_area', 'surface_land', 'number_facades', 'swimming_pool']].to_numpy()
    y = csv['price'].to_numpy().reshape(-1, 1)

    return X, y

def prep_data_categorical(csv):
    x = csv[['region', 'zip_code', 'property_type', 'property_subtype', 'number_rooms', 'living_area',
       'furnished', 'fireplace', 'terrace', 'terrace_area', 'garden',
       'garden_area', 'surface_land', 'number_facades', 'swimming_pool', 'building_state']]

    x = pd.get_dummies(data=x, drop_first=True)
    X = x.to_numpy()
    y = csv['price'].to_numpy()

    return X, y

def train(X_train, y_train): 
    """Initializes and trains the model"""
    regressor = LinearRegression().fit(X_train, y_train)

    return regressor

def coef_determination(y, pred):
    """Calculate the coefficient determination; the model is better the close to 1 your result is"""
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v