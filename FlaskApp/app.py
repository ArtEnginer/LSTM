from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import load_model

import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load and preprocess data
def load_data(file_path):
    data = pd.read_excel(file_path, parse_dates=['Timestamp'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace=True)
    return data

def preprocess_data(data):
    data = data.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])
    return np.array(dataX), np.array(dataY)

# Build and train the model
def build_and_train_model(X_train, y_train, input_shape, optimizer, loss, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/modelling', methods=['GET', 'POST'])
def modelling():            
    return render_template('modelling.html')

@app.route('/modelling_proses', methods=['POST'])
def modelling_proses():
    file = request.files['file']
    optimizer = request.form['optimizer']
    loss = request.form['loss']
    epochs = int(request.form['epochs'])
    batch_size = int(request.form['batch_size'])
    train_size = float(request.form['train_size'])
    train_status = 0;
    
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        data = load_data(file_path)
        scaled_data, scaler = preprocess_data(data[['RmTmp - DH01_ROW01_LA_TTHT01', 
                                                    'RmTmp - DH01_ROW01_LA_TTHT02', 
                                                    'RmTmp - DH01_ROW01_LA_TTHT03', 
                                                    'RmTmp - DH01_ROW01_LA_TTHT04',
                                                    'RmRhTL - DH01_ROW01_LA_TTHT01', 
                                                    'RmRhTL - DH01_ROW01_LA_TTHT02']])
        time_step = 15
        X, y = create_dataset(scaled_data, time_step)
        train_size = int(len(X) * train_size)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Select optimizer
        if optimizer == 'adam':
            optimizer = Adam()
        elif optimizer == 'sgd':
            optimizer = SGD()
        elif optimizer == 'rmsprop':
            optimizer = RMSprop()

        model = build_and_train_model(X_train, y_train, (time_step, X.shape[2]), optimizer, loss, epochs, batch_size)
        model.save('model/model.h5')
        # change train status to 1
        train_status = 1
        if train_status == 1:
            flash('Model trained successfully', 'success')
            
            # Mendapatkan nilai loss atau metrik lainnya setelah pelatihan
            loss_value = model.evaluate(X_test, y_test)
            # Contoh untuk metrik lainnya
            # other_metric = model.evaluate(X_test, y_test, metric='accuracy')

            return jsonify({
                'status': 'success',
                'input_details': {
                    # 'optimizer': optimizer,
                    # 'loss': loss,
                    # 'epochs': epochs,
                    # 'batch_size': batch_size,
                    # 'train_size': train_size
                },
                'output_details': {
                    'loss_value': loss_value,
                    # 'other_metric': other_metric
                }
            })
        else:
            return jsonify({'status': 'error'})

@app.route('/forecasting')
def forecasting():
    return render_template('forecasting.html')
@app.route('/forecast', methods=['POST'])
def forecast():
    input_date_str = request.form['input-date']
    input_date = datetime.datetime.strptime(input_date_str, '%Y-%m-%d %H:%M:%S')
    
    model = load_model('model/model.h5')
    time_step = 15  # Adjust according to how your model was trained
    
    # Example: Prepare input data for forecasting
    # This part needs to be adapted based on your specific data and preprocessing steps
    # Here, we assume you prepare input data for the LSTM model
    
    # Prepare input data for LSTM model
    input_data = prepare_input_data(input_date, time_step)
    
    # Predict using the model
    predicted_values = model.predict(input_data)
    
    # Example thresholds (you should adjust according to your requirements)
    sla_suhu = 24
    sla_humiditi = 50
    
    # Example of extracting predicted values
    predicted_suhu = predicted_values[0][0]  # Adjust based on your model's output structure
    predicted_humiditi = predicted_values[0][1]  # Adjust based on your model's output structure
    
    # Example warning logic
    warning_message = ""
    if predicted_suhu > sla_suhu:
        warning_message += "Suhu melebihi batas yang ditetapkan. "
    if predicted_humiditi > sla_humiditi:
        warning_message += "Kelembaban melebihi batas yang ditetapkan. "
    
    # Return the predicted values and any warnings as JSON response
    return jsonify({
        'status': 'success',
        'predicted_suhu': float(predicted_suhu),  # Ensure float conversion for JSON serialization
        'predicted_humiditi': float(predicted_humiditi),
        'warnings': warning_message
    })

def prepare_input_data(input_date, time_step):
    # Implement your logic to prepare input data for LSTM based on input_date
    # Example: Load recent data up to input_date and create sequences of time steps
    # You may need to fetch recent data and preprocess it similarly to the training data preparation
    
    # Example: Assuming you create a sequence of time steps starting from input_date
    # Adjust this based on your actual data and preprocessing steps
    input_data = [...]  # Format input_data as (1, time_step, num_features)
    
    return np.array([input_data])  # Ensure the shape is (1, time_step, num_features)





if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
