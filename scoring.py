
import os
import numpy as np
import json
import  pandas as pd
import joblib

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)['data']
        #data = np.array(json.loads(data))
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        return result.tolist()
    except Exception as err:
        return str(err)
