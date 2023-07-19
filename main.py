import src.cleanup as cleanup
import src.train_model as train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import numpy as np

# Get path variables
src_path, out_path = cleanup.build_path()
# Parse csv
csv = cleanup.get_csv(src_path)
# Clean up the csv
csv = cleanup.clean_csv(csv)
# Save the csv
cleanup.save_csv(csv, out_path)
#csv = cleanup.get_csv(out_path)

# Prepare data, filter out columns
X, y = train_model.prep_data_categorical(csv)

# Split the data into test and training sets and scale it
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
regressor = train_model.train_LR(X_train, y_train)

# Get the model score
"""score_train = cross_val_score(regressor, X_train, y_train, cv=10)
score_test = cross_val_score(regressor, X_test, y_test, cv=10)"""
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)


# Get the root mean squared error
y_pred = regressor.predict(X_test)
rmse= np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))

# Calculate the coefficient determination (how well did our model do?)
coef_determination = train_model.coef_determination(y_test, y_pred)

print(f"score train:{score_train}")
print(f"score test:{score_test}")
print(f"rmse:{rmse}")
print(f"coef_determination:{coef_determination}")


