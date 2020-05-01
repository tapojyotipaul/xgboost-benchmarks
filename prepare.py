from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb

#DATA_URL = """https://www.kaggle.com/wendykan/lending-club-loan-data/download/wzLfTo5dSwyLukcTTrNF%2Fversions%2FZcHnSJZfzhbtP2qKQhbY%2Ffiles%2Floan.csv?datasetVersionNumber=2"""
PARAMS = dict(
    objective ='reg:squarederror', 
    colsample_bytree = 0.3, 
    learning_rate = 0.1,
    max_depth = 5, 
    alpha = 10, 
)

# For testing we will use a smaller dataset
boston = load_boston()
data = pd.DataFrame(boston.data)

X, y = data.iloc[:,:-1], data.iloc[:,-1]
data_xgb = xgb.DMatrix(data=X, label=y)
xgb_model = xgb.train(params=PARAMS, dtrain=data_xgb)

data_xgb.save_binary('data/data.xgb')
xgb_model.save_model('data/model.xgb')