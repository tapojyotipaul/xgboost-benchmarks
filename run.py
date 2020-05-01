from timeit import default_timer as timer
import xgboost as xgb

NUM_LOOPS = 10

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
    run_times.append(total_time)
    inference_time = total_time*1000/num_rows
    inference_times.append(inference_time)

    print(f'{total_time} s', f'\t{inference_time} ms')