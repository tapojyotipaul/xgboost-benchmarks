from timeit import default_timer as timer
import xgboost as xgb

NUM_LOOPS = 1000

def calculate_stats(time_list):
    """Calculate mean and standard deviation of a list"""
    mean = sum(time_list) / len(time_list) 
    variance = sum([((x - mean) ** 2) for x in time_list]) / len(time_list) 
    std_dev = variance ** 0.5
    return (mean, std_dev)

data = xgb.DMatrix('data/data.xgb')
num_rows = data.num_row()

model = xgb.Booster()
model.load_model('data/model.xgb')

run_times = []
inference_times = []
for i in range(NUM_LOOPS):

    start_time = timer()
    res = model.predict(data)
    end_time = timer()

    total_time = end_time - start_time
    run_times.append(total_time*10e3)
    inference_time = total_time*(10e6)/num_rows
    inference_times.append(inference_time)

    print(f'Loop #{i}', f'{total_time} s', f'\t{inference_time} ms')

mean, std_dev = calculate_stats(run_times)
print(f"Loops: {len(run_times)} | Time per loop {mean:.2f}ms | Standard Deviation {std_dev:.2f}ms")

mean, std_dev = calculate_stats(inference_times)
print(f"Time per inference {mean:.2f}\u03bcs | Standard Deviation {std_dev:.2f}\u03bcs")