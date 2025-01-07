import numpy as np
import pandas as pd

class DecomposeResult:
    def __init__(self, observed, trend, seasonal, resid):
        self.observed = observed
        self.trend = trend
        self.seasonal = seasonal
        self.resid = resid

    def plot(self):
        """Plots the decomposition."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        
        # Plot observed
        plt.subplot(411)
        plt.plot(self.observed, label='Observed')
        plt.title('Observed')
        
        # Plot trend
        plt.subplot(412)
        plt.plot(self.trend, label='Trend')
        plt.title('Trend')
        
        # Plot seasonal
        plt.subplot(413)
        plt.plot(self.seasonal, label='Seasonal')
        plt.title('Seasonal')
        
        # Plot residual
        plt.subplot(414)
        plt.scatter(self.resid.index, self.resid, label='Residual')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title('Residual')
        
        plt.tight_layout()
        plt.show()

def seasonal_decompose(series, model='additive', freq=None):
    """Decomposes a time series into seasonal, trend, and residual components."""
    
    # Ensure the series is a pandas Series with a DatetimeIndex
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("The series index must be a DatetimeIndex")

    # Fill frequency if none provided (could be monthly, weekly, etc.)
    if freq is None:
        freq = pd.infer_freq(series.index)
        if freq is None:
            raise ValueError("Frequency of the series could not be inferred. Please provide `freq`.")
    
    # Apply rolling mean for trend extraction
    trend = series.rolling(window=freq, center=True, min_periods=1).mean()

    # Subtract the trend from the original series to get detrended data
    if model == 'additive':
        detrended = series - trend
    elif model == 'multiplicative':
        detrended = series / trend

    # Compute seasonal component (here we'll just use a simple mean for illustration)
    seasonal = detrended.groupby(series.index.month).transform('mean')

    # Compute the residual component
    if model == 'additive':
        resid = series - trend - seasonal
    elif model == 'multiplicative':
        resid = series / (trend * seasonal)

    return DecomposeResult(observed=series, trend=trend, seasonal=seasonal, resid=resid)
