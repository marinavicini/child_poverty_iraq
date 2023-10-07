import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import scipy
from scipy import stats
from sklearn.model_selection import (
    train_test_split,
    cross_val_predict,
    KFold,
    cross_val_score,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from shapely import wkt


import child_poverty_iraq.data.load_data as ld
import child_poverty_iraq.utils.constants as c


def transform_boxcox(data, target="deprived_mod", plot=False):
    boxcox_transformed, lmbda = stats.boxcox(data[target])
    print(f"Lambda: {lmbda}")

    data[f"{target}_bc"] = boxcox_transformed

    # Plot the transformed data
    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))  # , dpi=200)
        sns.histplot(boxcox_transformed, kde=True)
        plt.title("Moderate Prevalence after Boxcox transformation")
        plt.xlabel("Transformed data")
    return data, lmbda


def reverse_boxcox(values, lmbda):
    retransformed = scipy.special.inv_boxcox(values, lmbda)
    return retransformed


def define_sample_weights(data, countrycode="IRQ", weight=1):
    sample_weights = np.where(data["countrycode"] == "IRQ", weight, 1)
    return sample_weights


def get_standard_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def get_pca(data, n_components=400):
    X = data[c.mosaiks_features]
    scaler = get_standard_scaler(X)

    X_scaled = scaler.transform(X)
    # Create a PCA instance and specify the number of components or variance to retain
    pca = PCA(n_components=n_components)

    # Fit the PCA model to your standardized data
    pca.fit(X_scaled)

    return pca, scaler


def transform_pca(X, pca, scaler):
    X_scaled = scaler.transform(X)
    return pca.transform(X_scaled)


def split_train_test(data, target, test_size=0.2):
    merged_train, merged_test = train_test_split(
        data, test_size=test_size, random_state=60
    )

    return merged_train, merged_test


def clean_target(data, target):
    # print("cleaning data")
    # print(data.shape)
    # print(data[target].isna().sum())
    data_clean = data[data[target].isna() == False].copy()
    # print(data_clean.shape)
    return data_clean


def define_model_cv(X_train, y_train, model=Ridge(alpha=0.1), sample_weights=None):
    if sample_weights is None:
        sample_weights = np.ones(X_train.shape[0])

    # Initialize the ridge regression model
    # model = Ridge(alpha=0.1)

    # Fit the model using K-fold cross-validation on the training data
    kf = KFold(n_splits=5, shuffle=True, random_state=60)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="r2",
        fit_params={"sample_weight": sample_weights},
    )
    # y_train_pred = cross_val_predict(model, X_train, y_train, cv=kf)

    r2_cv = scores.mean()
    r2_cv_sd = scores.std()
    # print("%0.2f R2 with a standard deviation of %0.2f" % (r2_cv, r2_cv_sd))

    # Fit the model to the entire training data
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_train_pred = model.predict(X_train)

    # Calculate mean squared error and R-squared on the training data
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    # print(f"Training Mean Squared Error: {mse_train:.2f}")
    # print(f"Training R-squared: {r2_train:.2f}")
    # print()

    results = {"r2_cv": r2_cv, "r2_cv_sd": r2_cv_sd, "r2_train": r2_train}

    return model, y_train_pred, results


def evaluate_test(model, X_test, y_test):
    # Make predictions on the test data
    y_test_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    # print(f"Mean Squared Error: {mse:.2f}")
    # print(f"R-squared: {r2:.2f}")

    return y_test_pred, {"r2_test": r2}


def invert_pred_bc(y_train_pred, y_train, y_test, y_test_pred, lmbda):
    y_train_pred = reverse_boxcox(y_train_pred, lmbda)
    y_train = reverse_boxcox(y_train, lmbda)

    y_test = reverse_boxcox(y_test, lmbda)
    y_test_pred = reverse_boxcox(y_test_pred, lmbda)
    return y_train_pred, y_train, y_test, y_test_pred


def train_model(
    data,
    target="deprived_mod",
    model=Ridge(alpha=0.1),
    boxcox=False,
    pca_components=None,
    weight=1,
    plot=False,
):
    # Clean target
    data = clean_target(data, target)

    # Boxcox
    if boxcox:
        data, lmbda = transform_boxcox(data, target, plot=plot)
        target_t = f"{target}_bc"
    else:
        target_t = target
        lmbda = None

    # Split train test
    merged_train, merged_test = split_train_test(data, target_t)
    merged_train["is_training"] = 1
    merged_test["is_training"] = 0
    merged = pd.concat([merged_train, merged_test])

    X_train, y_train = merged_train[c.mosaiks_features], merged_train[target_t]
    X_test, y_test = merged_test[c.mosaiks_features], merged_test[target_t]

    # Define weights (if weight = 1, there is no weighting)
    sample_weights = define_sample_weights(
        merged_train, countrycode="IRQ", weight=weight
    )

    # PCA
    if pca_components is not None:
        pca, scaler = get_pca(X_train, n_components=pca_components)
        X_train = transform_pca(X_train, pca, scaler)
        X_test = transform_pca(X_test, pca, scaler)
    else:
        # Scale the features
        scaler = get_standard_scaler(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Model
    model, y_train_pred, results_train = define_model_cv(
        X_train, y_train, model=model, sample_weights=sample_weights
    )
    y_test_pred, results_test = evaluate_test(model, X_test, y_test)
    results_train.update(results_test)

    # Retransform
    if boxcox:
        y_train_pred, y_train, y_test, y_test_pred = invert_pred_bc(
            y_train_pred, y_train, y_test, y_test_pred, lmbda
        )

    # Compute predictions and errors
    merged["predictions"] = list(y_train_pred) + list(y_test_pred)
    merged["target_error"] = merged[target] - merged["predictions"]

    # Save as a geodataframe
    merged["geometry"] = merged["geometry"].apply(lambda x: wkt.loads(x))
    merged = gpd.GeoDataFrame(merged, geometry="geometry")

    model_details = {
        "target": target,
        "model": model,
        "boxcox": boxcox,
        "lmbda": lmbda,
        "pca_components": pca_components,
        "pca": pca,
        "scaler": scaler,
        "weight": weight,
    }

    return (merged, model, model_details, results_train)


# data = pd.read_csv("../../../data/processed/20230918_adm1_mosaiks_pov_merged.csv")
# merged, model, model_details, results = train_model(
#     data,
#     target="deprived_mod",
#     model=Ridge(0.5),
#     boxcox=True,
#     pca_components=300,
#     weight=1,
#     plot=True,
# )

# print(results)
