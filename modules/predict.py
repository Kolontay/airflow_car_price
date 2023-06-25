import os
import pandas as pd
import glob
import dill as dill
import json
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '..')

test_dir = f'{path}/data/test'
file_list = os.listdir(test_dir)
json_files = [file for file in file_list if os.path.splitext(file)[1] == '.json']




def predict():
    model_dir = f'{path}/data/models'
    model_files = glob.glob(model_dir + '/cars_pipe_*')
    latest_file = max(model_files, key=os.path.getctime)

    with open(latest_file, 'rb') as file:
        model = dill.load(file)

    predicts = pd.DataFrame()
    for json_file in json_files:
        path_to_file = os.path.join(test_dir, json_file)
        with open(path_to_file) as f:
            df = pd.DataFrame([json.load(f)])
        df['pred'] = model.predict(df)
        predicts = pd.concat([predicts, df.loc[:, ['id', 'pred']]])

    predicts.to_csv(f'{path}/data/predictions/predicts_{datetime.now().strftime("%Y%m%d%H%M")}.csv')





if __name__ == '__main__':
    predict()
