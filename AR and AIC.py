import numpy as np
import pandas as pd

def build_fourier_design_matrix(N: int, K: int) -> np.ndarray:
    t = np.arange(1, N + 1)
    columns = [np.ones(N)]
    for k in range(1, K + 1):
        freq = 2 * np.pi * k * t / 365.25
        columns.append(np.sin(freq))
        columns.append(np.cos(freq))
    return np.column_stack(columns)

def fit_ar_ols(anomalies: np.ndarray, p: int) -> tuple:
    N = len(anomalies)
    z = anomalies[p:]
    
    # construct lagged matrix A
    A = extinction_lag_matrix = np.column_stack([anomalies[p-i:-i] for i in range(1, p + 1)])
    
    # solve standard normal equations: A^T * A * phi = A^T * z
    phi = np.linalg.solve(A.T @ A, A.T @ z)
    
    # dalculate residuals (epsilon)
    residuals = z - (A @ phi)
    residual_variance = np.var(residuals)
    
    return phi, residuals, residual_variance

def evaluate_aic(anomalies: np.ndarray, max_p: int = 14) -> tuple:
    """Finds the optimal lag order p minimizing the Akaike Information Criterion."""
    best_aic = float('inf')
    best_p = 1
    best_phi = None
    best_residuals = None
    
    for p in range(1, max_p + 1):
        phi, residuals, res_var = fit_ar_ols(anomalies, p)
        N_eff = len(residuals)
        
        # Calculate AIC using the Maximum Likelihood formulation
        aic = N_eff * np.log(res_var) + 2 * p
        
        if aic < best_aic:
            best_aic = aic
            best_p = p
            best_phi = phi
            best_residuals = residuals
            
    return best_p, best_phi, best_residuals

def process_linear_engine(parquet_path: str, K: int = 2) -> dict:
    
    df = pd.read_parquet(parquet_path)
    # currently working with only TMAX
    tmax_cols = [col for col in df.columns if col.startswith('TMAX_')]
    model_registry = {}
    N = len(df)
    # fourier matrx
    M_design = build_fourier_design_matrix(N, K)
    
    for col in tmax_cols:
        station_id = col.split('_')[1]
        y = df[col].values
        theta = np.linalg.solve(M_design.T @ M_design, M_design.T @ y)
        seasonal_baseline = M_design @ theta
        anomalies = y - seasonal_baseline
        
        optimal_p, phi, residuals = evaluate_aic(anomalies, max_p=14)
        residual_series = pd.Series(
        residuals,
        index=df.index[optimal_p:]
        )
        model_registry[station_id] = {
            'fourier_coefs': theta,
            'optimal_lag_p': optimal_p,
            'ar_coefs': phi,
            'residuals': residuals
        }
        print(f"Station {station_id} -> Cleaned. Optimal Lag p={optimal_p} chosen via AIC minimization.")
        
    return model_registry
