from sklearn.datasets import load_boston
import numpy as np
import xgboost as xgb

X, y = load_boston(return_X_y=True)

STATS = '#, median, mean, std_dev, min_time, max_time, quantile_10, quantile_90'


def get_test_data(size: int = 1):
    """Generates a test dataset of the specified size""" 
    num_rows = len(X)
    test_df = X.copy()

    while num_rows < size:
        test_df = np.append(test_df, test_df, axis=0)
        num_rows = len(test_df)

    return test_df[:size]


def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    time_array = np.array(time_list)

    median = np.median(time_array)
    mean = np.mean(time_array)
    std_dev = np.std(time_array)
    max_time = np.amax(time_array)
    min_time = np.amin(time_array)
    quantile_10 = np.quantile(time_array, 0.1)
    quantile_90 = np.quantile(time_array, 0.9)

    return (median, mean, std_dev, min_time, max_time, quantile_10, quantile_90)