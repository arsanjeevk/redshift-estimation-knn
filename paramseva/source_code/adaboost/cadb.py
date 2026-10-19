import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import randomsplit, training_set_1, training_set_2, training_set_3
from redshift_fn import binDataFunc
from results import eta_015, regression_results, plot_regression_scatter, classification_results, plot_classification_scatter

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# ------------------------- CAdB-1 TEST ----------------------------

file_path = "/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"

# Load data and split
X_train, X_test, y_train, y_test, test_indices, y = training_set_1(file_path)

# Create 15 bins
y_all = np.concatenate([y_train, y_test])
y_all_binned, bin_edges, bin_values = binDataFunc(y_all.copy(), 15)

n_train = len(y_train)
y_train_binned = y_all_binned[:n_train]
y_test_binned = y_all_binned[n_train:]

# Convert binned redshifts to integer class labels
y_train_class = np.searchsorted(bin_values, y_train_binned).astype(int)
y_test_class = np.searchsorted(bin_values, y_test_binned).astype(int)

print("Number of classes:", len(bin_values))
print("Training classes:", np.unique(y_train_class))
print("Testing classes:", np.unique(y_test_class))

# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    random_state=42
)

param_grid = {
    "n_estimators": [1000, 1500, 2000, 2500, 3000],
    "learning_rate": [0.002, 0.005, 0.01, 0.02, 0.05],
    "estimator__max_depth": [2, 3, 4]
}


# Stratified 10-fold cross-validation
kfold = StratifiedKFold(
    n_splits=10,
    random_state=10,
    shuffle=True
)

# GridSearchCV based on classification accuracy
grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=kfold,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train_class)

# Best model based on CV accuracy
best_model = grid_search.best_estimator_

print("\n<-------- CAdB-1 TEST -------->")
print("GridSearchCV Results")
print("Best CV accuracy:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)

# Predict test classes
predicted_class = best_model.predict(X_test)

# Convert predicted classes to representative redshift values
bin_values = np.asarray(bin_values)
y_pred = bin_values[predicted_class]

# Convert to arrays
y_test = np.asarray(y_test, dtype=np.float32)
y_pred = np.asarray(y_pred, dtype=np.float32)

# Classification accuracy
accuracy = np.mean(y_test_class == predicted_class)

print(f"\nCAdB1 Results")
print(f"Accuracy: {accuracy:.4f}")

# Calculate eta_0.15, eta_2sigma, sigma and sigma_NMAD
classification_results(
    y_test,
    y_pred,
    test_name="CAdB1"
)

# Plot classification scatter graph
plot_classification_scatter(
    y_test,
    y_pred,
    title="CAdB1"
)



# ------------------------ CAdB-2 TEST -------------------------

file_path = "/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"

# Load data and split
X_train, X_test, y_train, y_test = training_set_2(file_path)

# Create 15 bins
y_all = np.concatenate([y_train, y_test])
y_all_binned, bin_edges, bin_values = binDataFunc(y_all.copy(), 15)

n_train = len(y_train)
y_train_binned = y_all_binned[:n_train]
y_test_binned = y_all_binned[n_train:]

# Convert binned redshifts to integer class labels
y_train_class = np.searchsorted(bin_values, y_train_binned).astype(int)
y_test_class = np.searchsorted(bin_values, y_test_binned).astype(int)



# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    random_state=42
)

param_grid = {
    "n_estimators": [1000, 1500, 2000, 2500, 3000],
    "learning_rate": [0.002, 0.005, 0.01, 0.02, 0.05],
    "estimator__max_depth": [2, 3, 4]
}

# Eta_0.15 scorer
eta015_scorer = make_scorer(eta_015, greater_is_better=False)

# Stratified 10-fold cross-validation
kfold = StratifiedKFold(
    n_splits=10,
    random_state=10,
    shuffle=True
)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=kfold,
    scoring={
        "eta015": eta015_scorer,
        "accuracy": "accuracy"
    },
    n_jobs=-1,
    refit="eta015",
    verbose=1
)

grid_search.fit(X_train, y_train_class)

# Best model
best_model = grid_search.best_estimator_

# Best CV results
best_index = grid_search.best_index_

cv_accuracy = grid_search.cv_results_["mean_test_accuracy"][best_index]
cv_eta015 = -grid_search.cv_results_["mean_test_eta015"][best_index]

print("\n<-------- CAdB-2 TEST -------->")
print("GridSearchCV Results")
print("CV accuracy:", cv_accuracy)
print("CV eta_0.15:", cv_eta015)
print("Best parameters:", grid_search.best_params_)

# Predict test classes
predicted_class = best_model.predict(X_test)

# Convert predicted classes to representative redshift values
bin_values = np.asarray(bin_values)
y_pred = bin_values[predicted_class]

# Convert to arrays
y_test = np.asarray(y_test, dtype=np.float32)
y_pred = np.asarray(y_pred, dtype=np.float32)

# Classification accuracy
accuracy = np.mean(y_test_class == predicted_class)

# Final results
print(f"Accuracy: {accuracy:.4f}")
classification_results(y_test, y_pred, test_name="CAdB2")

# plot the classification scatter graphs
plot_classification_scatter(y_test, y_pred, title="CAdB2")



# ------------------------- CAdB-3 TEST -------------------------

file_path = "/home/ph25mscst11026.phy.iith/redshift/data/ATLAS.csv"

# Load data and split
X_train, X_test, y_train, y_test = training_set_3(file_path)

# Create 15 bins
y_all = np.concatenate([y_train, y_test])
y_all_binned, bin_edges, bin_values = binDataFunc(y_all.copy(), 15)

n_train = len(y_train)
y_train_binned = y_all_binned[:n_train]
y_test_binned = y_all_binned[n_train:]

# Convert binned redshifts to integer class labels
y_train_class = np.searchsorted(bin_values, y_train_binned).astype(int)
y_test_class = np.searchsorted(bin_values, y_test_binned).astype(int)


# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    random_state=42
)

param_grid = {
    "n_estimators": [1000, 1500, 2000, 2500, 3000],
    "learning_rate": [0.002, 0.005, 0.01, 0.02, 0.05],
    "estimator__max_depth": [2, 3, 4]
}

# Eta_0.15 scorer
eta015_scorer = make_scorer(eta_015, greater_is_better=False)

# Stratified 10-fold cross-validation
kfold = StratifiedKFold(
    n_splits=10,
    random_state=10,
    shuffle=True
)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=kfold,
    scoring={
        "eta015": eta015_scorer,
        "accuracy": "accuracy"
    },
    n_jobs=-1,
    refit="eta015",
    verbose=1
)

grid_search.fit(X_train, y_train_class)

# Best model
best_model = grid_search.best_estimator_

# Best CV results
best_index = grid_search.best_index_

cv_accuracy = grid_search.cv_results_["mean_test_accuracy"][best_index]
cv_eta015 = -grid_search.cv_results_["mean_test_eta015"][best_index]

print("\n<-------- CAdB-3 TEST -------->")
print("GridSearchCV Results")
print("CV accuracy:", cv_accuracy)
print("CV eta_0.15:", cv_eta015)
print("Best parameters:", grid_search.best_params_)

# Predict test classes
predicted_class = best_model.predict(X_test)

# Convert predicted classes to representative redshift values
bin_values = np.asarray(bin_values)
y_pred = bin_values[predicted_class]

# Convert to arrays
y_test = np.asarray(y_test, dtype=np.float32)
y_pred = np.asarray(y_pred, dtype=np.float32)

# Classification accuracy
accuracy = np.mean(y_test_class == predicted_class)

# Final results
print(f"Accuracy: {accuracy:.4f}")
classification_results(y_test, y_pred, test_name="CAdB3")

# plot the classification scatter graphs
plot_classification_scatter(y_test, y_pred, title="CAdB3")