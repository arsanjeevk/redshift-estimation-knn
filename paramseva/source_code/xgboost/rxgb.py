import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import randomsplit, training_set_1, training_set_2, training_set_3
from redshift_fn import binDataFunc
from results import eta_015, regression_results, plot_regression_scatter, classification_results, plot_classification_scatter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier




# ----------------------- RXGB1-TEST----------------------------

# Load data
file_path="/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"
X_train,X_test,y_train,y_test,test_indices,y=training_set_1(file_path)

# XGBoost regression model
model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# Hyperparameter grid

param_grid = {
    "n_estimators": [200, 400, 600, 800, 1000, 1200],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2]
}

# Grid search with 10-fold CV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Remove invalid predictions if any
valid = np.isfinite(y_test) & np.isfinite(y_pred)
y_test_valid = y_test[valid]
y_pred_valid = y_pred[valid]

# GridSearchCV results
print("\n<--------RXGB1-TEST-------->")
print("GridSearchCV Results")
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))
print("Best parameters:", grid_search.best_params_)

# Model information
print("\nBest XGBoost Parameters")
print("Number of estimators:", model.n_estimators)
print("Maximum depth:", model.max_depth)
print("Learning rate:", model.learning_rate)

# regression results
results = regression_results(y_test_valid, y_pred_valid, test_name="RXGB-1" )
print(results)

# Plotting the results
plot_regression_scatter(y_test_valid, y_pred_valid, title="RXGB-1")



# ----------------------- RXGB2-TEST----------------------------


# Load data
file_path="/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"
X_train, X_test, y_train, y_test = training_set_2(file_path)

# XGBoost regression model
model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [200, 400, 600, 800, 1000, 1200],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2]
}

# Grid search with 10-fold CV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Remove invalid predictions if any
valid = np.isfinite(y_test) & np.isfinite(y_pred)
y_test_valid = y_test[valid]
y_pred_valid = y_pred[valid]

# GridSearchCV results
print("\n<--------RXGB2-TEST-------->")
print("GridSearchCV Results")
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))
print("Best parameters:", grid_search.best_params_)

# Model information
print("\nBest XGBoost Parameters")
print("Number of estimators:", model.n_estimators)
print("Maximum depth:", model.max_depth)
print("Learning rate:", model.learning_rate)

# regression results
results = regression_results(y_test_valid, y_pred_valid, test_name="RXGB-2" )
print(results)

# Plotting the results
plot_regression_scatter(y_test_valid, y_pred_valid, title="RXGB-2")




# ----------------------- RXGB3-TEST----------------------------

# Load data
file_path="/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"
X_train, X_test, y_train, y_test = training_set_3(file_path)

# XGBoost regression model
model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [200, 400, 600, 800, 1000, 1200],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2]
}

# Grid search with 10-fold CV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Remove invalid predictions if any
valid = np.isfinite(y_test) & np.isfinite(y_pred)
y_test_valid = y_test[valid]
y_pred_valid = y_pred[valid]

# GridSearchCV results
print("\n<--------RXGB3-TEST-------->")
print("GridSearchCV Results")
print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))
print("Best parameters:", grid_search.best_params_)

# Model information
print("\nBest XGBoost Parameters")
print("Number of estimators:", model.n_estimators)
print("Maximum depth:", model.max_depth)
print("Learning rate:", model.learning_rate)

# regression results
results = regression_results(y_test_valid, y_pred_valid, test_name="RXGB-3" )
print(results)

# Plotting the results
plot_regression_scatter(y_test_valid, y_pred_valid, title="RXGB-3")