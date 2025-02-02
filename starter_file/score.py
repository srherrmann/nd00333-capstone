import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    # Load the model from the registered model in the workspace
    model_path = Model.get_model_path('heart_failure_prediction_best_hyperdrive_model')  # Replace with your model name
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Parse the input data
        #data = np.array(json.loads(raw_data)['data'])

        x = pd.DataFrame(json.loads(raw_data)['data'])
        # Make predictions
        predictions = model.predict(x)
        #predictions = model.predict(data)
        # Return the predictions as JSON
        return json.dumps(predictions.tolist())
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})