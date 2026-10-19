import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# randomly splitting
def randomsplit(X, y, return_indices=False):
    np.random.seed(42)

    test_indices = np.random.choice(
        len(X),
        round(len(X) * 0.3),
        replace=False
    )

    train_indices = np.array(
        list(set(range(len(X))) - set(test_indices))
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    if return_indices:
        return X_train, X_test, y_train, y_test, train_indices, test_indices

    return X_train, X_test, y_train, y_test

# type-1 test: random 70/30 train-test split
def training_set_1(file_path):
    df = pd.read_csv(file_path)

    features = [
        "Sp2", "flux_ap2_36", "flux_ap2_45", "flux_ap2_58",
        "flux_ap2_80", "MAG_APER_4_G", "MAG_APER_4_R",
        "MAG_APER_4_I", "MAG_APER_4_Z"
    ]
    target = "z"

    X = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test, train_indices, test_indices = randomsplit(
        X, y, return_indices=True
    )

    print("Original data shape:", df.shape)
    print("\nDATA PARTITION")
    print("Total sources:", len(X_train) + len(X_test))
    print("Training sources:", len(X_train))
    print("Testing sources:", len(X_test))

    return X_train, X_test, y_train, y_test, test_indices, y


# type-2 test: ELAIS-S1 → CDFS
def training_set_2(file_path):
    df = pd.read_csv(file_path)
    df["field"] = df["field"].str.strip()

    df_cdfs = df[df["field"] == "CDFS"].copy()
    df_elais = df[df["field"] == "ELAIS-S1"].copy()

    features = [
        "Sp2", "flux_ap2_36", "flux_ap2_45", "flux_ap2_58",
        "flux_ap2_80", "MAG_APER_4_G", "MAG_APER_4_R",
        "MAG_APER_4_I", "MAG_APER_4_Z"
    ]
    target = "z"

    X_train = df_elais[features].to_numpy(dtype=np.float32)
    X_test = df_cdfs[features].to_numpy(dtype=np.float32)
    y_train = df_elais[target].to_numpy(dtype=np.float32)
    y_test = df_cdfs[target].to_numpy(dtype=np.float32)

    print("Original data shape:", df.shape)
    print("\nDATA PARTITION")
    print("Total sources:", len(X_train) + len(X_test))
    print("Training sources(ELAIS-S1):", len(X_train))
    print("Testing sources(CDFS):", len(X_test))

    return X_train, X_test, y_train, y_test


# type-3 test: CDFS → ELAIS-S1
def training_set_3(file_path):
    df = pd.read_csv(file_path)
    df["field"] = df["field"].str.strip()

    df_cdfs = df[df["field"] == "CDFS"].copy()
    df_elais = df[df["field"] == "ELAIS-S1"].copy()

    features = [
        "Sp2", "flux_ap2_36", "flux_ap2_45", "flux_ap2_58",
        "flux_ap2_80", "MAG_APER_4_G", "MAG_APER_4_R",
        "MAG_APER_4_I", "MAG_APER_4_Z"
    ]
    target = "z"

    X_train = df_cdfs[features].to_numpy(dtype=np.float32)
    X_test = df_elais[features].to_numpy(dtype=np.float32)
    y_train = df_cdfs[target].to_numpy(dtype=np.float32)
    y_test = df_elais[target].to_numpy(dtype=np.float32)

    print("Original data shape:", df.shape)
    print("\nDATA PARTITION")
    print("Total sources:", len(X_train) + len(X_test))
    print("Training sources(CDFS):", len(X_train))
    print("Testing sources(ELAIS-S1):", len(X_test))

    return X_train, X_test, y_train, y_test