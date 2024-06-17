from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta
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
def build_and_train_model(X_train, y_train, X_val, y_val, input_shape, optimizer, loss, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(y_train.shape[1]))
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model, history


@app.route('/')
def dashboard():
    active = 'dashboard'
    model_path = 'model/model.h5'
    if os.path.exists(model_path):
        model_creation_time = os.path.getctime(model_path)
        model_creation_date = pd.to_datetime(model_creation_time, unit='s')
        model_creation_date = model_creation_date.strftime('%Y-%m-%d %H:%M:%S')
        model_modified_time = os.path.getmtime(model_path)
        model_modified_date = pd.to_datetime(model_modified_time, unit='s')
        model_modified_date = model_modified_date.strftime('%Y-%m-%d %H:%M:%S')
        message = f"Model has been created on {model_creation_date} and modified on {model_modified_date}."
    else:
        message = "Please create the model."

    return render_template('dashboard.html', active=active, message=message)

@app.route('/modelling', methods=['GET', 'POST'])
def modelling():         
    active = 'modelling'   
    return render_template('modelling.html', active=active)

@app.route('/modelling_proses', methods=['POST'])
def modelling_proses():
    file = request.files['file']
    optimizer = request.form['optimizer']
    loss = request.form['loss']
    epochs = int(request.form['epochs'])
    batch_size = int(request.form['batch_size'])
    time_step = int(request.form['time_step'])
    train_size = float(request.form['train_size'])
    validation_size = float(request.form['validation_size'])
    train_status = 0
    
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
        time_step = time_step
        X, y = create_dataset(scaled_data, time_step)
        train_size = int(len(X) * train_size)
        validation_size = int(len(X) * validation_size)
        
        X_train, X_val = X[:train_size], X[train_size:train_size + validation_size]
        y_train, y_val = y[:train_size], y[train_size:train_size + validation_size]
        X_test, y_test = X[train_size + validation_size:], y[train_size + validation_size:]
        
        if optimizer == 'adam':
            optimizer = Adam()
        elif optimizer == 'sgd':
            optimizer = SGD()
        elif optimizer == 'rmsprop':
            optimizer = RMSprop()

        model, history = build_and_train_model(X_train, y_train, X_val, y_val, (time_step, X.shape[2]), optimizer, loss, epochs, batch_size)
        model.save('model/model.h5')
        train_status = 1
        if train_status == 1:
            flash('Model trained successfully', 'success')
            
            loss_value = model.evaluate(X_test, y_test)

            history_df = pd.DataFrame(history.history)
            history_df.to_csv('model/history.csv', index=False)

            return jsonify({
                'status': 'success',
                'output_details': {
                    'loss_value': loss_value,
                }
            })
        else:
            return jsonify({'status': 'error'})

@app.route('/model_history')
def model_history():
    if os.path.exists('model/history.csv'):
        history_df = pd.read_csv('model/history.csv')
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['loss'], label='Loss')
        
        if 'val_loss' in history_df:
            plt.plot(history_df['val_loss'], label='Validation Loss')
        
        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return render_template('model_history.html', image_base64=image_base64)
    else:
        flash('No training history found.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/forecasting')
def forecasting():
    active = 'forecasting'
    return render_template('forecasting.html', active=active)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Load model
        model = load_model('model/model.h5')
        
        # Get the last `time_step` data points for prediction
        data = load_data('uploads/dataset.xlsx')
        scaled_data, scaler = preprocess_data(data[['RmTmp - DH01_ROW01_LA_TTHT01', 
                                                    'RmTmp - DH01_ROW01_LA_TTHT02', 
                                                    'RmTmp - DH01_ROW01_LA_TTHT03', 
                                                    'RmTmp - DH01_ROW01_LA_TTHT04',
                                                    'RmRhTL - DH01_ROW01_LA_TTHT01', 
                                                    'RmRhTL - DH01_ROW01_LA_TTHT02']])
        
        # Use the last `time_step` data points
        time_step = 15  # Make sure this matches the time_step used during training
        last_data_points = scaled_data[-time_step:]
        
        # Reshape data to match the model input
        last_data_points = np.reshape(last_data_points, (1, last_data_points.shape[0], last_data_points.shape[1]))
        
        # Predict the next 15-30 minutes
        predictions = []
        for _ in range(2):  # Predicting for 15 and 30 minutes ahead
            pred = model.predict(last_data_points)
            predictions.append(pred)
            # Append the prediction to the input for the next prediction step
            last_data_points = np.append(last_data_points[:, 1:, :], pred.reshape(1, 1, -1), axis=1)
        
        # Inverse transform the predictions
        predictions = np.array(predictions).reshape(-1, scaled_data.shape[1])
        predictions = scaler.inverse_transform(predictions)
        
        # Extract temperature and humidity predictions
        temp_predictions = predictions[:, :4]
        humidity_predictions = predictions[:, 4:]
        
        # Prepare the response
        response = {
            'status': 'success',
            'predictions': {
                '15_minutes': {
                    'temperature': temp_predictions[0].tolist(),
                    'humidity': humidity_predictions[0].tolist()
                },
                '30_minutes': {
                    'temperature': temp_predictions[1].tolist(),
                    'humidity': humidity_predictions[1].tolist()
                }
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/realtime_forecast')
def realtime_forecast():
    active = 'realtime_forecast'
    return render_template('realtime_forecast.html', active=active)
@app.route('/realtime_predict', methods=['POST'])
def realtime_predict():
    try:
        # Load the model
        model = load_model('model/model.h5')
        
        # Get recent data
        recent_data = get_recent_data()

        if recent_data.empty:
            return jsonify({'status': 'error', 'message': 'No recent data available'})

        # Preprocess the recent data
        scaled_data, scaler = preprocess_data(recent_data)

        # Define time step
        time_step = 10  # Sesuaikan berdasarkan waktu step yang digunakan saat pelatihan
        X_recent, _ = create_dataset(scaled_data, time_step)

        # Get the last sequence for prediction
        X_input = X_recent[-1].reshape(1, time_step, X_recent.shape[2])

        # Make prediction for the next 15-30 minutes
        forecast_steps = 15  # 15 minutes prediction, adjust as needed
        predictions = []
        for _ in range(forecast_steps):
            next_prediction = model.predict(X_input)
            next_prediction = scaler.inverse_transform(next_prediction)
            predictions.append(next_prediction[0])
            X_input = np.append(X_input[:, 1:, :], next_prediction.reshape(1, 1, X_input.shape[2]), axis=1)

        # Convert predictions to DataFrame
        forecast_time = [datetime.now() + timedelta(minutes=i) for i in range(1, forecast_steps + 1)]
        forecast_df = pd.DataFrame(predictions, index=forecast_time, columns=recent_data.columns[:-1])

        response = {
            'status': 'success',
            'predictions': forecast_df.to_dict(orient='index')
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def get_recent_data():
    # Misalnya, Anda mengambil data dari database atau API
    # Contoh data dummy untuk ilustrasi
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=30)  # Data 30 menit terakhir
    time_range = pd.date_range(start=start_time, end=end_time, freq='T')
    dummy_data = {
        'Timestamp': time_range,
        'RmTmp - DH01_ROW01_LA_TTHT01': np.random.normal(loc=25, scale=1, size=len(time_range)),
        'RmTmp - DH01_ROW01_LA_TTHT02': np.random.normal(loc=26, scale=1, size=len(time_range)),
        'RmTmp - DH01_ROW01_LA_TTHT03': np.random.normal(loc=24, scale=1, size=len(time_range)),
        'RmTmp - DH01_ROW01_LA_TTHT04': np.random.normal(loc=25, scale=1, size=len(time_range)),
        'RmRhTL - DH01_ROW01_LA_TTHT01': np.random.normal(loc=60, scale=5, size=len(time_range)),
        'RmRhTL - DH01_ROW01_LA_TTHT02': np.random.normal(loc=62, scale=5, size=len(time_range)),
    }
    return pd.DataFrame(dummy_data).set_index('Timestamp')


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
