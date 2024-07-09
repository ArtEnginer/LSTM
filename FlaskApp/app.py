from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, send_file
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
from tensorflow.keras.callbacks import EarlyStopping

import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3) 
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,callbacks=[early_stop])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    if os.path.exists('model/accuracy_plot.png'):
        os.remove('model/accuracy_plot.png')
    plt.savefig('model/accuracy_plot.png')
    plt.close()

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
    try:
        optimizer = request.form['optimizer']
        loss = request.form['loss']
        epochs = int(request.form['epochs'])
        batch_size = int(request.form['batch_size'])
        time_step = int(request.form['time_step'])
        train_size = float(request.form['train_size'])
        train_status = 0
        validation_size = 1 - train_size
                
        file_path = os.path.join('uploads', 'dataset.xlsx')
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
        # Ensure reproducibility by setting the random seed
        np.random.seed(42)

        # Shuffle indices
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        # Split data
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]


        # X_train, X_val = X[:train_size], X[train_size:train_size + validation_size]
        # y_train, y_val = y[:train_size], y[train_size:train_size + validation_size]
        
        
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
            average_accuracy = np.mean(history.history['accuracy'])
            average_val_accuracy = np.mean(history.history['val_accuracy'])
            average_loss = np.mean(history.history['loss'])
            average_val_loss = np.mean(history.history['val_loss'])
            rmse = np.sqrt(np.mean(history.history['loss']))
            mse = np.mean(history.history['loss'])


            history_df = pd.DataFrame(history.history)
            history_df.to_csv('model/history.csv', index=False)
            
            return jsonify({
                'status': 'success',
                'output_details': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'time_step': time_step,
                    'train_size': train_size,
                    'validation_size': validation_size,
                    'optimizer': request.form['optimizer'],
                    'loss': loss,
                    'average_accuracy': average_accuracy,
                    'average_val_accuracy': average_val_accuracy,
                    'average_loss': average_loss,
                    'average_val_loss': average_val_loss,
                    'rmse': rmse,
                    'mse': mse,
                    'visualizations': {
                        'accuracy_plot': 'accuracy_plot.png'
                    }
                }
            })
    except FileNotFoundError:
        return jsonify({ 'status': 'error', 'message': 'File not found' })

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

@app.route('/download_template', methods=['GET', 'POST'])
def download_template():
    # get form templates/ file
    return send_file('templates/dataset.xlsx', as_attachment=True)

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.xlsx'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.xlsx'))
            return redirect(url_for('dataset'))
        else:
            return "Invalid file format", 400
    return render_template('dataset.html', active='dataset')

@app.route('/dataset-source')
def dataset_source():
    try:
        dataset = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.xlsx'))
        total_records = len(dataset)

        # Get pagination parameters from DataTables request
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '')

        # Filtering
        if search_value:
            dataset = dataset[dataset.apply(lambda row: row.astype(str).str.contains(search_value).any(), axis=1)]

        # Sorting
        order_column = int(request.args.get('order[0][column]', 0))
        order_dir = request.args.get('order[0][dir]', 'asc')
        column_name = dataset.columns[order_column]
        dataset = dataset.sort_values(by=column_name, ascending=(order_dir == 'asc'))

        # Pagination
        data_subset = dataset.iloc[start:start + length]

        # Convert to dictionary
        data = data_subset.to_dict(orient='records')

        return jsonify({
            'draw': request.args.get('draw', 1),
            'recordsTotal': total_records,
            'recordsFiltered': len(dataset),
            'data': data
        })
    except FileNotFoundError:
        return jsonify({
            'draw': request.args.get('draw', 1),
            'recordsTotal': 0,
            'recordsFiltered': 0,
            'data': []
        })
    
@app.route('/dataset_preprocessing')
def dataset_preprocessing():
    try:
        data = load_data('uploads/dataset.xlsx')
      
        data = data[['PQM08 Real Power Mean (kW)',
                    'RmTmp - DH01_ROW01_LA_TTHT01', 
                    'RmTmp - DH01_ROW01_LA_TTHT02', 
                    'RmTmp - DH01_ROW01_LA_TTHT03', 
                    'RmTmp - DH01_ROW01_LA_TTHT04',
                    'RmRhTL - DH01_ROW01_LA_TTHT01', 
                    'RmRhTL - DH01_ROW01_LA_TTHT02']]
        data = data.dropna()
        total_records = len(data)
        data = data.reset_index()
        data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        

        ## Get pagination parameters from DataTables request
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '')

        # Filtering
        if search_value:
            data = data[data.apply(lambda row: row.astype(str).str.contains(search_value).any(), axis=1)]

        # Sorting
        order_column = int(request.args.get('order[0][column]', 0))
        order_dir = request.args.get('order[0][dir]', 'asc')
        column_name = data.columns[order_column]
        data = data.sort_values(by=column_name, ascending=(order_dir == 'asc'))

        # Pagination
        data_subset = data.iloc[start:start + length]

        df = data_subset.to_dict(orient='records')
        return jsonify({
            'draw': request.args.get('draw', 1),
            'recordsTotal': total_records,
            'recordsFiltered': len(data),
            'data': df
        })

    except FileNotFoundError:
        return jsonify({
            'draw': request.args.get('draw', 1),
            'recordsTotal': 0,
            'recordsFiltered': 0,
            'data': []
        })

@app.route('/dataset-normalization')
def dataset_normalization():
    try:
        data = load_data('uploads/dataset.xlsx')
    
        data = data[['PQM08 Real Power Mean (kW)',
                    'RmTmp - DH01_ROW01_LA_TTHT01', 
                    'RmTmp - DH01_ROW01_LA_TTHT02', 
                    'RmTmp - DH01_ROW01_LA_TTHT03', 
                    'RmTmp - DH01_ROW01_LA_TTHT04',
                    'RmRhTL - DH01_ROW01_LA_TTHT01', 
                    'RmRhTL - DH01_ROW01_LA_TTHT02']]
        data = data.dropna()
        data = data.reset_index()
        data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        scaled_data, scaler = preprocess_data(data[['RmTmp - DH01_ROW01_LA_TTHT01', 
                                                'RmTmp - DH01_ROW01_LA_TTHT02', 
                                                'RmTmp - DH01_ROW01_LA_TTHT03', 
                                                'RmTmp - DH01_ROW01_LA_TTHT04',
                                                'RmRhTL - DH01_ROW01_LA_TTHT01', 
                                                'RmRhTL - DH01_ROW01_LA_TTHT02']])
        scaled_data = pd.DataFrame(scaled_data, columns=['RmTmp - DH01_ROW01_LA_TTHT01', 
                                                        'RmTmp - DH01_ROW01_LA_TTHT02', 
                                                        'RmTmp - DH01_ROW01_LA_TTHT03', 
                                                        'RmTmp - DH01_ROW01_LA_TTHT04',
                                                        'RmRhTL - DH01_ROW01_LA_TTHT01', 
                                                        'RmRhTL - DH01_ROW01_LA_TTHT02'])
        scaled_data['Timestamp'] = data['Timestamp']
        scaled_data = scaled_data.reset_index(drop=True)
        
        total_records = len(scaled_data)
        # Get pagination parameters from DataTables request
        start = int(request.args.get('start', 0))
        length = int(request.args.get('length', 10))
        search_value = request.args.get('search[value]', '')

        # Filtering
        if search_value:
            scaled_data = scaled_data[scaled_data.apply(lambda row: row.astype(str).str.contains(search_value).any(), axis=1)]

        # Sorting
        order_column = int(request.args.get('order[0][column]', 0))
        order_dir = request.args.get('order[0][dir]', 'asc')
        column_name = scaled_data.columns[order_column]
        scaled_data = scaled_data.sort_values(by=column_name, ascending=(order_dir == 'asc'))

        # Pagination
        data_subset = scaled_data.iloc[start:start + length]

        df = data_subset.to_dict(orient='records')
        return jsonify({
            'draw': request.args.get('draw', 1),
            'recordsTotal': total_records,
            'recordsFiltered': len(scaled_data),
            'data': df
        })

    except FileNotFoundError:
        return jsonify({
            'draw': request.args.get('draw', 1),
            'recordsTotal': 0,
            'recordsFiltered': 0,
            'data': []
        })
    
@app.route('/dataset_visualization')
def dataset_visualization():
    try:
        data = pd.read_excel('uploads/dataset.xlsx')    
        # Converting Timestamp column to datetime
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Plotting temperature data
        plt.figure(figsize=(14, 7))
        plt.plot(data['Timestamp'], data['RmTmp - DH01_ROW01_LA_TTHT01'], label='TTHT01')
        plt.plot(data['Timestamp'], data['RmTmp - DH01_ROW01_LA_TTHT02'], label='TTHT02')
        plt.plot(data['Timestamp'], data['RmTmp - DH01_ROW01_LA_TTHT03'], label='TTHT03')
        plt.plot(data['Timestamp'], data['RmTmp - DH01_ROW01_LA_TTHT04'], label='TTHT04')
        plt.xlabel('Timestamp')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Over Time')
        plt.legend()
        plt.grid(True)
        # if file sudah ada maka timpa dengan yang baru
        if os.path.exists('static/temperature_plot.png'):
            os.remove('static/temperature_plot.png')
        plt.savefig('static/temperature_plot.png')
        plt.close()

        # Plotting humidity data
        plt.figure(figsize=(14, 7))
        plt.plot(data['Timestamp'], data['RmRhTL - DH01_ROW01_LA_TTHT01'], label='TTHT01 Humidity')
        plt.plot(data['Timestamp'], data['RmRhTL - DH01_ROW01_LA_TTHT02'], label='TTHT02 Humidity')
        plt.xlabel('Timestamp')
        plt.ylabel('Humidity (%)')
        plt.title('Humidity Over Time')
        plt.legend()
        plt.grid(True)
        # if file sudah ada maka timpa dengan yang baru
        if os.path.exists('static/humidity_plot.png'):
            os.remove('static/humidity_plot.png')
        plt.savefig('static/humidity_plot.png')
        plt.close()
        return redirect(url_for('dashboard'))
    except FileNotFoundError:
        pass
@app.route('/model/<filename>')
def model(filename):
    return send_file(f'model/{filename}', as_attachment=True)
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
