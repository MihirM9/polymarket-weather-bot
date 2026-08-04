import numpy as np
import pandas as pd
from collections import deque


def simulate_hybrid_paths(
    station_id: str,
    model_registry: dict,
    xgb_model,
    feature_names: list,
    recent_anomalies: np.ndarray,
    recent_residuals: np.ndarray,
    xgb_residual_std: float,
    forecast_horizon: int = 14,
    n_paths: int = 10000
) -> np.ndarray:
    """
    Hybrid Monte Carlo simulator.

    Forecasts future temperature anomalies using

        • AR(p) linear dynamics
        • XGBoost residual correction
        • Gaussian innovation noise

    Returns
    -------
    simulated_anomalies : np.ndarray
        Shape = (n_paths, forecast_horizon)
    """

    # -------------------------------------------------
    # Load station-specific model parameters
    # -------------------------------------------------
    ar_coefs = np.asarray(
        model_registry[station_id]["ar_coefs"],
        dtype=np.float32
    )

    p_lags = model_registry[station_id]["optimal_lag_p"]

    simulated_anomalies = np.zeros(
        (n_paths, forecast_horizon),
        dtype=np.float32
    )

    for path in range(n_paths):

        # Fixed-length rolling buffers
        anomaly_buffer = deque(
            recent_anomalies[-p_lags:],
            maxlen=p_lags
        )

        residual_buffer = deque(
            recent_residuals[-7:],
            maxlen=7
        )

        for day in range(forecast_horizon):

            # -------------------------------------------------
            # 1. Linear AR prediction
            # -------------------------------------------------
            ar_prediction = np.dot(
                ar_coefs,
                np.array(anomaly_buffer, dtype=np.float32)[::-1]
            )

            # -------------------------------------------------
            # 2. Reconstruct XGBoost feature vector
            # -------------------------------------------------
            feature_dict = {}

            # Dynamic residual lag features
            for lag in range(1, p_lags + 1):

                feature_dict[f"target_res_lag_{lag}"] = (
                    residual_buffer[-lag]
                )

                feature_dict[f"target_res_abs_lag_{lag}"] = abs(
                    residual_buffer[-lag]
                )

            # Rolling volatility features
            feature_dict["rolling_vol_3d"] = np.std(
                list(residual_buffer)[-3:]
            )

            feature_dict["rolling_vol_7d"] = np.std(
                list(residual_buffer)[-7:]
            )

            # Preserve training feature schema
            for feature in feature_names:
                feature_dict.setdefault(feature, 0.0)

            X_step = pd.DataFrame(
                [feature_dict],
                columns=feature_names
            )

            # -------------------------------------------------
            # 3. Predict nonlinear residual
            # -------------------------------------------------
            predicted_residual = xgb_model.predict(X_step)[0]

            # -------------------------------------------------
            # 4. Sample remaining innovation
            # -------------------------------------------------
            innovation = np.random.normal(
                loc=0.0,
                scale=xgb_residual_std
            )

            total_residual = (
                predicted_residual +
                innovation
            )

            # -------------------------------------------------
            # 5. Hybrid anomaly forecast
            # -------------------------------------------------
            next_anomaly = (
                ar_prediction +
                total_residual
            )

            simulated_anomalies[path, day] = next_anomaly

            # -------------------------------------------------
            # 6. Roll buffers forward
            # -------------------------------------------------
            anomaly_buffer.append(next_anomaly)
            residual_buffer.append(total_residual)

    return simulated_anomalies