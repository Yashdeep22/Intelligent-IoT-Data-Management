import time
import pandas as pd
from adtk.detector import QuantileAD
from adtk.data import validate_series


class QuantileADDetector:
    """
    Quantile-based anomaly detector using ADTK.

    Flags values outside specified quantile range and provides
    a numeric anomaly score based on distance from thresholds.
    """

    def __init__(self, high=0.95, low=0.05):
        self.high = high
        self.low = low
        self.model = QuantileAD(high=high, low=low)
        self.name = "QuantileADDetector"

    def detect(self, df: pd.DataFrame) -> dict:
        start_time = time.time()

        # Input validation
        if df is None or df.shape[1] == 0:
            raise ValueError("Input DataFrame must contain at least one column.")

        # Use first column (simple version)
        series = validate_series(df.iloc[:, 0])

        # Detect anomalies
        anomalies = self.model.fit_detect(series)

        # Align index properly
        anomalies = anomalies.reindex(df.index)
        anomaly_flag = anomalies.fillna(False).astype(bool)

   
        # Create NUMERIC anomaly score
        lower = series.quantile(self.low)
        upper = series.quantile(self.high)

        # Distance-based score
        scores = ((series - upper).clip(lower=0) +
                  (lower - series).clip(lower=0))

        scores = scores.reindex(df.index).fillna(0)

        runtime = time.time() - start_time

        return {
            "model_name": self.name,
            "timestamp": df.index,
            "anomaly_flag": anomaly_flag,
            "score": scores,   
            "runtime": runtime
        }
