import argparse
import logging
import pandas as pd
from pathlib import Path

from timeit import default_timer as timer
import common


# Setup logging
# logger = logging.getLogger('__name__')
# logging.getLogger().setLevel(logging.INFO)


# Initialize Command Line arguments
parser = argparse.ArgumentParser(description='Runs inference on models.')
parser.add_argument(
    '-m', '--model', type=str, default="xgboost",
    help="Type of model to run")
parser.add_argument(
    '-o', '--observations', type=int, default=1e6,
    help="Max batch size")
args = parser.parse_args()


# main module
if __name__=='__main__':

    # Load data
    model = args.model
    logging.info(f"Running benchmark for {model}...")

    # Run Inferencing
    if model == 'xgboost':
        import xg
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            xg.run_inference(batch_size)
            batch_size *= 10
            
    elif model == 'daal':
        import daal
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)  
    elif model == 'daal_lm':
        import daal_lm
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_lm.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'daal_rf':
        import daal_rf
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_rf.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'lm':
        import lm
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = lm.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'rf':
        import rf
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = rf.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'daal_logit':
        import daal_logit
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = daal_logit.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
            
    elif model == 'logit':
        import logit
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = logit.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
        
    elif model == 'lm_patch':
        import lm_patch
        temp_df = pd.DataFrame()
        batch_size = 1  # Start with a single observation
        # logging.info(common.STATS)
        while batch_size <= args.observations:
            temp = lm_patch.run_inference(batch_size)
            temp["No_of_Observation"] = batch_size
            temp_df = temp_df.append(temp)
            batch_size *= 10
        print("__________________Summary_______________________")
        print(temp_df)
    else:
        print(f"Could not find benchmark for {model}")
        print(f"Available choices are: xgboost, daal")


    

