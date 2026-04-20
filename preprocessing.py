import numpy as np
import pandas as pd

class KalmanFilter:
    """
    Standard Recursive Kalman Filter for sensor denoising.
    Optimized for high-frequency IMU jitter.
    """
    def __init__(self, R=0.1, Q=0.1, A=1.0, B=0.0, C=1.0):
        self.R = R  # Process Noise
        self.Q = Q  # Measurement Noise
        self.A = A
        self.B = B
        self.C = C
        self.cov = np.nan
        self.x = np.nan

    def filter(self, z):
        if np.isnan(self.x):
            self.x = z
            self.cov = self.Q
        else:
            # Prediction
            pred_x = self.x
            pred_cov = self.cov + self.R
            
            # Update
            K = pred_cov / (pred_cov + self.Q)
            self.x = pred_x + K * (z - pred_x)
            self.cov = (1 - K) * pred_cov
        return self.x

def apply_sliding_window(df, window_size=30, step_size=5, feature_cols=None):
    """
    Transforms time-series data into flattened feature vectors.
    Returns: X (matrix of windows), y (labels)
    """
    if feature_cols is None:
        feature_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "speed_kmh"]
    
    # Group by event continuity
    df["group"] = (df["mapped_label"] != df["mapped_label"].shift()).cumsum()
    feats, labs = [], []
    
    for _, g in df.groupby("group"):
        if len(g) < window_size:
            continue
        label = g["mapped_label"].iloc[0]
        for start in range(0, len(g) - window_size, step_size):
            w = g.iloc[start : start + window_size]
            raw_vector = w[feature_cols].values.flatten()
            feats.append(raw_vector)
            labs.append(label)
            
    cols = [f"{s}_t{t}" for t in range(window_size) for s in ["ax", "ay", "az", "gx", "gy", "gz", "speed"]]
    return pd.DataFrame(feats, columns=cols).fillna(0), np.array(labs)
