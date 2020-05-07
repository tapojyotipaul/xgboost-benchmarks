import argparse
import logging

from pathlib import Path
from timeit import default_timer as timer
import common


# Setup logging
logger = logging.getLogger('__name__')
logging.getLogger().setLevel(logging.INFO)


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
        logging.info(common.STATS)
        while batch_size <= args.observations:
            xg.run_inference(batch_size)
            batch_size *= 10
    else:
        print(f"Could not find benchmark for {model}")


    

