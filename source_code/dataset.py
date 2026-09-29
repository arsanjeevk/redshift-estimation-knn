import numpy as np
import pandas as pd


# def randomsplit(X, y): # X is the feature matrix and y is the target vector

def training_set_1(file_path): # file_path of ALTAS.csv
    # Read the datasets
    df = pd.read_csv(file_path)

    # Define features and target
    features = ["Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"]
    target = "z"

    # Convert data to NumPy arrays
    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)

    # Split the data into training and testing sets(70% training, 30% testing)
    np.random.seed(42)

    test_indices = np.random.choice(len(X),round(len(X) * 0.3),replace=False )

    train_indices = np.array(
        list(set(range(len(X))) - set(test_indices))
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def training_set_2(file_path): # file_path of ALTAS.csv
    # Read the datasets
    df = pd.read_csv(file_path)

    # Remove leading/trailing whitespace from field names
    df["field"] = df["field"].str.strip()

    # Split the datasets
    df_cdfs = df[df["field"] == "CDFS"].copy()
    df_elais = df[df["field"] == "ELAIS-S1"].copy()
    
    # Define features and target
    features = ["Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"]
    target = "z"

    # Extract features and redshifts
    X_train = df_elais[features].to_numpy(dtype=np.float32)
    X_test = df_cdfs[features].to_numpy(dtype=np.float32)
    y_train = df_elais[target].to_numpy(dtype=np.float32)
    y_test = df_cdfs[target].to_numpy(dtype=np.float32)

    return X_train, X_test, y_train, y_test

def training_set_3(file_path): # file_path of ALTAS.csv
    # Read the datasets
    df = pd.read_csv(file_path)

    # Remove leading/trailing whitespace from field names
    df["field"] = df["field"].str.strip()

    # Split the datasets
    df_cdfs = df[df["field"] == "CDFS"].copy()
    df_elais = df[df["field"] == "ELAIS-S1"].copy()
    
    # Define features and target
    features = ["Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"]
    target = "z"

    # Extract features and redshifts
    X_train = df_cdfs[features].to_numpy(dtype=np.float32)
    X_test = df_elais[features].to_numpy(dtype=np.float32)
    y_train = df_cdfs[target].to_numpy(dtype=np.float32)
    y_test = df_elais[target].to_numpy(dtype=np.float32)

    return X_train, X_test, y_train, y_test