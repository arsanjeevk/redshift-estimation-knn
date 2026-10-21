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




# ----------------------- CXGB1-TEST----------------------------

file_path="/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"

# Load data and split
X_train,X_test,y_train,y_test,test_indices,y=training_set_1(file_path)

# Create 15 bins
y_all=np.concatenate([y_train,y_test])
y_all_binned,bin_edges,bin_values=binDataFunc(y_all.copy(),15)

n_train=len(y_train)
y_train_binned=y_all_binned[:n_train]
y_test_binned=y_all_binned[n_train:]

# Convert binned redshifts to integer class labels
y_train_class=np.searchsorted(bin_values,y_train_binned).astype(int)
y_test_class=np.searchsorted(bin_values,y_test_binned).astype(int)

print("Number of classes:",len(bin_values))
print("Training classes:",np.unique(y_train_class))
print("Testing classes:",np.unique(y_test_class))

# XGBoost classifier
xgb=XGBClassifier(
    objective="multi:softmax",
    num_class=len(bin_values),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [800, 1000, 1200, 1500, 1800, 2000],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.05, 0.08, 0.1, 0.15, 0.2]
}

# GridSearchCV
grid_search=GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    verbose=1
)

# Fit
grid_search.fit(X_train,y_train_class)

best_model=grid_search.best_estimator_

print("\n<--------CXGB1-TEST-------->")
print("GridSearchCV Results")
print("Best CV accuracy:",grid_search.best_score_)
print("Best parameters:",grid_search.best_params_)

# Predict test set
y_pred_class=best_model.predict(X_test).astype(int)

# Convert predicted classes back to redshift
final_prediction=np.asarray(bin_values)[y_pred_class].astype(np.float32)

print("\nBest XGBoost parameters:")
print("Number of estimators:",best_model.n_estimators)
print("Learning rate:",best_model.learning_rate)
print("Maximum depth:",best_model.max_depth)
print("10-fold CV accuracy:",grid_search.best_score_)

# Classification results
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Accuracy: {accuracy:.4f}")

results=classification_results(
    y_test_binned,
    final_prediction,
    test_name="CXGB1"
)

# Scatter plot using original continuous redshifts
y_test_spec=y[test_indices]

assert len(y_test_spec)==len(final_prediction)

plot_classification_scatter(
    y_test_spec,
    final_prediction,
    title="CXGB1"
)


# ----------------------- CXGB2-TEST--------------------------

file_path="/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"

# Load data and split
X_train,X_test,y_train,y_test=training_set_2(file_path)

# Create 15 bins
y_all=np.concatenate([y_train,y_test])
y_all_binned,bin_edges,bin_values=binDataFunc(y_all.copy(),15)

n_train=len(y_train)
y_train_binned=y_all_binned[:n_train]
y_test_binned=y_all_binned[n_train:]

# Convert binned redshifts to integer class labels
y_train_class=np.searchsorted(bin_values,y_train_binned).astype(int)
y_test_class=np.searchsorted(bin_values,y_test_binned).astype(int)

# print("Number of classes:",len(bin_values))
# print("Training classes:",np.unique(y_train_class))
# print("Testing classes:",np.unique(y_test_class))

# XGBoost classifier
xgb=XGBClassifier(
    objective="multi:softmax",
    num_class=len(bin_values),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [800, 1000, 1200, 1500, 1800, 2000],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.05, 0.08, 0.1, 0.15, 0.2]
}

# GridSearchCV
grid_search=GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    verbose=1
)

# Fit
grid_search.fit(X_train,y_train_class)

best_model=grid_search.best_estimator_

print("\n<--------CXGB2-TEST-------->")
print("GridSearchCV Results")
print("Best CV accuracy:",grid_search.best_score_)
print("Best parameters:",grid_search.best_params_)

# Predict test set
y_pred_class=best_model.predict(X_test).astype(int)

# Convert predicted classes back to redshift
final_prediction=np.asarray(bin_values)[y_pred_class].astype(np.float32)

print("\nBest XGBoost parameters:")
print("Number of estimators:",best_model.n_estimators)
print("Learning rate:",best_model.learning_rate)
print("Maximum depth:",best_model.max_depth)
print("10-fold CV accuracy:",grid_search.best_score_)

# Classification results
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Accuracy: {accuracy:.4f}")

results=classification_results(
    y_test_binned,
    final_prediction,
    test_name="CXGB2"
)

# Scatter plot
plot_classification_scatter(
    y_test,
    final_prediction,
    title="CXGB2"
)


# --------------------------- CXGB3-TEST -------------------------

file_path="/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"

# Load data and split
X_train,X_test,y_train,y_test=training_set_3(file_path)

# Create 15 bins
y_all=np.concatenate([y_train,y_test])
y_all_binned,bin_edges,bin_values=binDataFunc(y_all.copy(),15)

n_train=len(y_train)
y_train_binned=y_all_binned[:n_train]
y_test_binned=y_all_binned[n_train:]

# Convert binned redshifts to integer class labels
y_train_class=np.searchsorted(bin_values,y_train_binned).astype(int)
y_test_class=np.searchsorted(bin_values,y_test_binned).astype(int)

# print("Number of classes:",len(bin_values))
# print("Training classes:",np.unique(y_train_class))
# print("Testing classes:",np.unique(y_test_class))

# XGBoost classifier
xgb=XGBClassifier(
    objective="multi:softmax",
    num_class=len(bin_values),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Hyperparameter grid
param_grid = {
    "n_estimators": [800, 1000, 1200, 1500, 1800, 2000],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.05, 0.08, 0.1, 0.15, 0.2]
}

# GridSearchCV
grid_search=GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    verbose=1
)

# Fit
grid_search.fit(X_train,y_train_class)

best_model=grid_search.best_estimator_

print("\n<--------CXGB3-TEST-------->")
print("GridSearchCV Results")
print("Best CV accuracy:",grid_search.best_score_)
print("Best parameters:",grid_search.best_params_)

# Predict test set
y_pred_class=best_model.predict(X_test).astype(int)

# Convert predicted classes back to redshift
final_prediction=np.asarray(bin_values)[y_pred_class].astype(np.float32)

print("\nBest XGBoost parameters:")
print("Number of estimators:",best_model.n_estimators)
print("Learning rate:",best_model.learning_rate)
print("Maximum depth:",best_model.max_depth)
print("10-fold CV accuracy:",grid_search.best_score_)

# Classification results
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Accuracy: {accuracy:.4f}")

results=classification_results(
    y_test_binned,
    final_prediction,
    test_name="CXGB3"
)

# Scatter plot
plot_classification_scatter(
    y_test,
    final_prediction,
    title="CXGB3"
)