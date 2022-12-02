import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# App initialization
app = Flask(__name__)

# Load Models

with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

from tensorflow.keras.models import load_model
model_ann = load_model('churn_model.h5')

# rooting
@app.route('/') # Homepage
def home():
    return '<h1> It works baby! </h1>'

@app.route('/predict', methods=['POST'])
def titanic_predict():
    args= request.json
    print(args,dir(args))

    data_inf = {
        'gender': args.get('gender'),
        'SeniorCitizen': args.get('SeniorCitizen'),
        'Partner': args.get('Partner'),
        'Dependents': args.get('Dependents'),
        'MultipleLines': args.get('MultipleLines'),
        'InternetService': args.get('InternetService'),
        'OnlineSecurity': args.get('OnlineSecurity'),
        'OnlineBackup':args.get('OnlineBackup'),
        'DeviceProtection': args.get('DeviceProtection'),
        'TechSupport': args.get('TechSupport'),
        'Contract': args.get('Contract'),
        'PaperlessBilling':args.get('PaperlessBilling'),
        'PaymentMethod': args.get('PaymentMethod'), 
        'tenure' : args.get('tenure'),
        'MonthlyCharges': args.get('MonthlyCharges'),
        'TotalCharges' : args.get('TotalCharges')
    }

    print('[DEBUG] Data Inference:', data_inf)
    
    data_inf= pd.DataFrame([data_inf])
    data_inf_transform = model_pipeline.transform(data_inf)
    y_pred_inf = model_ann.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5,1,0)

    if y_pred_inf == 0:
        label = 'Not Churn'
    else:
        label = 'Churn'

    print('[DEBUG] result :', y_pred_inf, label)
    print('')

    response=jsonify(
        result = str(y_pred_inf),
        label_names=label
    )

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')