from timeit import default_timer as timer

import daal4py as d4p
import numpy as np
import pandas as pd

import common

NUM_LOOPS = 100
PARAMS = { 
    'nIterations': 10,
    'method': 'defaultDense',
    'fptype': 'double'
}

gbt = d4p.gbt_regression_training(maxIterations=200)
MODEL = gbt.compute(
            pd.DataFrame(common.X, dtype=np.float32), 
            pd.DataFrame(common.y, dtype=np.float32)).model

def run_inference(num_observations:int = 1000):
    """Run xgboost for specified number of observations"""
    # Load data
    test_df = common.get_test_data(num_observations)
    data = pd.DataFrame(test_df, dtype=np.float32)
    predictor = d4p.gbt_regression_prediction(**PARAMS)
    num_rows = len(test_df)

    run_times = []
    inference_times = []
    for _ in range(NUM_LOOPS):
        
        start_time = timer()
        predictor.compute(data, MODEL)
        end_time = timer()

        total_time = end_time - start_time
        run_times.append(total_time*10e3)

        inference_time = total_time*(10e6)/num_rows
        inference_times.append(inference_time)

    print(num_observations, ", ", common.calculate_stats(inference_times))