from timeit import default_timer as timer

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import daal4py as d4p
import numpy as np
import pandas as pd

import common
d4p.daalinit()
NUM_LOOPS = 100

print("Computing for Kmeans with Daal")
init_alg = d4p.kmeans_init(nClusters = 5, fptype = "float",
                           method = "randomDense")
centroids = init_alg.compute(common.X_dfc).centroids
alg = d4p.kmeans(nClusters = 5, maxIterations = 100,
                 fptype = "float", accuracyThreshold = 0,
                 assignFlag = False)
result = alg.compute((common.X_dfc), centroids)

def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = common.get_test_data_df(X=common.X_dfc,size = num_observations)
    num_rows = len(test_df)
    ######################
    print("_______________________________________")
    print("Total Number of Rows", num_rows)
    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):
        
        start_time = timer()
        # predict_algo = d4p.decision_forest_regression_prediction(fptype='float') ###Change with the Kmeans Prediction methods
        # predict_result = alg.compute(test_df, centroids) ###Change with the Kmeans Prediction methods
        #predictor.compute(data, MODEL)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    return_elem = common.calculate_stats(inference_times)
    print(num_observations, ", ", return_elem)
    return return_elem