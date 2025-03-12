from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import Counter
from mail import sendMail
from scipy.signal import find_peaks, butter, filtfilt

app = Flask(__name__)

# Define LSTM model (must match the trained model's structure)
class ActivityLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActivityLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Model parameters (must match training setup)
input_size = 15  
hidden_size = 64
num_classes = 5  

# Load trained model
model = ActivityLSTM(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('activity_recognition_lstm_finetuned.pth'))
model.eval()  

# Label mapping
label_map = {0: 'walking', 1: 'running', 2: 'falling_while_walking', 3: 'falling_while_running', 4: 'gesture'}

def predict_activity(df):
    """Predict activity from sensor data"""
    feature_columns = [col for col in df.columns if col not in ['Time [s]']]
    df = df[feature_columns].dropna()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.values.astype(np.float32))

    time_steps = 50
    X = [df_scaled[i:i+time_steps] for i in range(len(df_scaled) - time_steps)]
    if len(X) == 0:
        return []

    X = torch.tensor(np.array(X), dtype=torch.float32)
    dataset = TensorDataset(X)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch[0])
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())

    return [label_map[p] for p in predictions]

def calculate_steps(df):
    """Detect steps using accelerometer magnitude peaks"""
    step_columns = ["lsm6dsv16x_acc_x [g]", "lsm6dsv16x_acc_y [g]", "lsm6dsv16x_acc_z [g]"]
    if not all(col in df.columns for col in step_columns):
        return "N/A (Missing Data)"

    acc_x, acc_y, acc_z = df[step_columns].values.T
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    acc_magnitude -= np.mean(acc_magnitude)

    def low_pass_filter(data, cutoff_freq, sample_rate, order=4):
        nyquist_freq = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    sample_rate, cutoff_freq = 50, 2.0
    acc_filtered = low_pass_filter(acc_magnitude, cutoff_freq, sample_rate)
    peaks, _ = find_peaks(acc_filtered, height=0.3, distance=25)

    return len(peaks)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        file.seek(0)
        if file.read(1) == '':
            return jsonify({'error': 'Uploaded file is empty'})
        
        file.seek(0)
        df = pd.read_csv(file, encoding='utf-8')

    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Uploaded file is empty or corrupt'})
    except pd.errors.ParserError:
        return jsonify({'error': 'Error parsing CSV file. Check format'})
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'})

    # Make activity predictions
    predictions = predict_activity(df)
    most_common_prediction = Counter(predictions).most_common(1)[0][0] if predictions else "Unknown"

    # Compute step count
    steps = calculate_steps(df) if not df.empty else "N/A"

    # Get temperature data
    temp_column = "stts22h_temp [°C]"
    temp_value = df[temp_column].iloc[0] if temp_column in df.columns and not df.empty else "N/A"

    # Trigger emergency alerts
    if most_common_prediction == 'gesture':
            try:
                sendMail('ujjawalgarg7@gmail.com','GESTURE')
                print("Email sent because gesture was detected")
            except Exception as e:
                print(f"Error sending email: {e}")
    elif most_common_prediction == 'falling_while_walking':
            try:
                sendMail('ujjawalgarg7@gmail.com','FALLEN')
                print("Email sent because gesture was detected")
            except Exception as e:
                print(f"Error sending email: {e}")
    elif most_common_prediction == 'falling_while_running':
            try:
                sendMail('ujjawalgarg7@gmail.com','FALLEN')
                print("Email sent because gesture was detected")
            except Exception as e:
                print(f"Error sending email: {e}")

    return jsonify({
        'activity': most_common_prediction,
        'steps': steps,
        'temp': temp_value
    })

if __name__ == '__main__':
    app.run(debug=True)