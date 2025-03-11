from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import Counter
from mail import sendMail
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
input_size = 15  # Update if different
hidden_size = 64
num_classes = 5  # Update if different

# Load trained model
model = ActivityLSTM(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('activity_recognition_lstm_finetuned.pth'))
model.eval()  # Set to evaluation mode

# Label mapping
label_map = {0: 'walking', 1: 'running', 2: 'falling_while_walking', 3: 'falling_while_running', 4: 'gesture'}

def predict_activity(file):
    df = pd.read_csv(file)

    # Select required columns (exclude time column)
    feature_columns = [col for col in df.columns if col not in ['Time [s]']]
    df = df[feature_columns].dropna()

    # Standardize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.values.astype(np.float32))

    # Prepare input sequences
    time_steps = 50
    X = [df_scaled[i:i+time_steps] for i in range(len(df_scaled) - time_steps)]
    X = torch.tensor(np.array(X), dtype=torch.float32)

    # Create DataLoader
    batch_size = 32
    dataset = TensorDataset(X)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch[0])
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())

    return [label_map[p] for p in predictions]

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

    # Make predictions
    predictions = predict_activity(file)

    # Find the most frequent prediction
    if predictions:
        most_common_prediction = Counter(predictions).most_common(1)[0][0]

        # Call sendMail if the prediction is 'gesture'
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
        elif most_common_prediction == 'falling_while_walking':
            try:
                sendMail('ujjawalgarg7@gmail.com','FALLEN')
                print("Email sent because gesture was detected")
            except Exception as e:
                print(f"Error sending email: {e}")

        return jsonify({'prediction': most_common_prediction})
    else:
        return jsonify({'prediction': "No prediction available"})

if __name__ == '__main__':
    app.run(debug=True)