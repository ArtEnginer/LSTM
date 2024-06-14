import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from threading import Thread
import time
import datetime

app = Flask(__name__)

# Load the pre-trained model
model = load_model('lstm_model.h5')

# Load and preprocess the data
def load_and_preprocess_data():
    data = pd.read_excel('dataset/dataset.xlsx')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace=True)

    features = ['PQM08 Real Power Mean (kW)', 
                'RmTmp - DH01_ROW01_LA_TTHT01', 
                'RmTmp - DH01_ROW01_LA_TTHT02', 
                'RmTmp - DH01_ROW01_LA_TTHT03', 
                'RmTmp - DH01_ROW01_LA_TTHT04', 
                'RmRhTL - DH01_ROW01_LA_TTHT01', 
                'RmRhTL - DH01_ROW01_LA_TTHT02']

    data = data[features]
    data = data.dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler, features, data.index

scaled_data, scaler, features, data_index = load_and_preprocess_data()

def create_dataset(dataset, time_step=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), :]
        data_X.append(a)
        data_Y.append(dataset[i + time_step, 1])
    return np.array(data_X), np.array(data_Y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], len(features))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    timestamp_str = request.form['timestamp']
    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')

    # Find the closest data point in the dataset based on the selected timestamp
    closest_index = data_index.get_loc(timestamp, method='nearest')
    if closest_index < time_step:
        closest_index = time_step

    # Create the input sequence for the model
    input_data = scaled_data[closest_index-time_step:closest_index]
    X_new = input_data.reshape(1, time_step, len(features))

    # Perform prediction
    predictions = model.predict(X_new)
    temp_predictions = predictions.reshape(-1, 1)
    full_predictions = np.zeros((len(temp_predictions), len(features)))
    full_predictions[:, 1] = temp_predictions[:, 0]
    inverse_transformed_predictions = scaler.inverse_transform(full_predictions)

    last_temp = inverse_transformed_predictions[-1][1]

    SLA_temp = 25.0  # Example SLA threshold for temperature
    warning_message = ""
    if last_temp > SLA_temp:
        warning_message = f"Warning: Temperature has exceeded the SLA threshold! Current temperature: {last_temp:.2f} °C"

    return render_template('index.html', warning_message=warning_message, last_temp=last_temp)

if __name__ == '__main__':
    app.run(debug=True)
