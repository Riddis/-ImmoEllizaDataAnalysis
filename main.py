import src.cleanup as cleanup
import src.train_model as train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Get path variables
src_path, out_path = cleanup.build_path()
# Parse csv
csv = cleanup.get_csv(src_path)
# Clean up the csv
"""csv = cleanup.clean_csv(csv)
# Save the csv
cleanup.save_csv(csv, out_path)"""
csv = cleanup.get_csv(out_path)

# Prepare data, filter out columns
X, y = train_model.prep_data_categorical(csv)

# Split the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y)


#Initialize and fit the model
regressor = train_model.train(X_train, y_train)

# Get the model score
score = regressor.score(X_test, y_test)

# Get the root mean squared error
y_pred = regressor.predict(X_test)
rmse = mean_squared_error(y_true=y, y_pred=y_pred, squared=False)
RMSE= np.sqrt(mean_squared_error(y_pred,y_test))

# Calculate the coefficient determination (how well did our model do?)
coef_determination = train_model.coef_determination(y, y_pred)

print(f"score:{score}")
print(f"rmse:{rmse}")
print(f"coef_determination:{coef_determination}")