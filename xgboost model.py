import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def engineer_residual_features(
    model_registry: dict,
    target_station: str
) -> tuple:

    target_res = model_registry[target_station]["residuals"]
    p_lags = model_registry[target_station]["optimal_lag_p"]

    df_features = pd.DataFrame({
        "target_res": target_res
    })

    for lag in range(1, p_lags + 1):

        df_features[f"target_res_lag_{lag}"] = (
            df_features["target_res"].shift(lag)
        )

        df_features[f"target_res_abs_lag_{lag}"] = np.abs(
            df_features[f"target_res_lag_{lag}"]
        )

    df_features["rolling_vol_3d"] = (
        df_features["target_res"]
        .shift(1)
        .rolling(window=3)
        .std()
    )

    df_features["rolling_vol_7d"] = (
        df_features["target_res"]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    for other_station, data in model_registry.items():

        if other_station != target_station:

            other_res = data["residuals"][-len(target_res):]

            df_features[
                f"spatial_res_{other_station}_lag_1"
            ] = pd.Series(other_res).shift(1).values

    df_features.dropna(inplace=True)

    X = df_features.drop(columns=["target_res"])
    y = df_features["target_res"].values

    return X, y


def train_xgb_residual_model(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5
) -> tuple:
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    fold_rmse = []
    fold_residuals = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )

        preds = xgb_model.predict(X_val)

        fold_residuals.extend(y_val - preds)

        rmse = np.sqrt(
            np.mean(
                (preds - y_val) ** 2
            )
        )

        fold_rmse.append(rmse)

        logging.info(
            f"Fold {fold + 1}/{n_splits} - "
            f"Out-of-Sample Validation RMSE: {rmse:.4f}"
        )

    logging.info(
        f"Mean Time-Series CV RMSE: {np.mean(fold_rmse):.4f}"
    )

    xgb_residual_std = np.std(fold_residuals)

    xgb_model.fit(
        X,
        y,
        verbose=False
    )

    return xgb_model, xgb_residual_std
