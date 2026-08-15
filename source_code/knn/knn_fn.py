import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

def knn_eu(train_df, test_df):

    features = ["Sp2", "flux_ap2_36", "flux_ap2_45", "flux_ap2_58", "flux_ap2_80",
                "MAG_APER_4_G", "MAG_APER_4_R", "MAG_APER_4_I", "MAG_APER_4_Z"]
    target = "z"

    X_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df[target].to_numpy(dtype=np.float32)
    X_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df[target].to_numpy(dtype=np.float32)

    print("DATA PARTITION")
    print("Training sources:", len(X_train))
    print("Testing sources :", len(X_test))
    print("Number of features:", X_train.shape[1])

    def eta_015(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true) > 0.15 * (1 + y_true))

    eta_scorer = make_scorer(eta_015, greater_is_better=False)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(metric="euclidean"))
    ])

    param_grid = {"knn__n_neighbors": range(2, 101)}

    cv = KFold(n_splits=10, shuffle=True, random_state=10)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=eta_scorer,
        cv=cv,
        refit=True,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_k = grid_search.best_params_["knn__n_neighbors"]
    best_cv_eta = -grid_search.best_score_

    print("\nBEST MODEL")
    print("Best k:", best_k)
    print("Best CV eta_0.15:", best_cv_eta)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results["eta_0.15"] = -cv_results["mean_test_score"]
    cv_results = cv_results[["param_knn__n_neighbors", "eta_0.15"]]

    print("\nCROSS-VALIDATION RESULTS")
    print(cv_results.to_string(index=False))

    y_pred = best_model.predict(X_test)

    delta_z = (y_test - y_pred) / (1 + y_test)

    eta_015_value = eta_015(y_test, y_pred) * 100

    sigma = np.std(delta_z)
    eta_2sigma = np.mean(np.abs(delta_z) > 2 * sigma) * 100

    median_delta_z = np.median(delta_z)
    mad = np.median(np.abs(delta_z - median_delta_z))
    sigma_nmad = 1.4826 * mad

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nFINAL RESULTS")
    print("Training sources:", len(y_train))
    print("Testing sources :", len(y_test))
    print("Number of features:", len(features))
    print("Best k:", best_k)
    print(f"eta_0.15: {eta_015_value:.2f}%")
    print(f"eta_2sigma: {eta_2sigma:.2f}%")
    print(f"sigma: {sigma:.4f}")
    print(f"sigma_NMAD: {sigma_nmad:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return best_model