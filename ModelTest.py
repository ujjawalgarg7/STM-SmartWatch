import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

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

# Model parameters (must match the training setup)
input_size = 15  # Update this if different
hidden_size = 64
num_classes = 5  # Update this if different

# Load trained model
model = ActivityLSTM(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('activity_recognition_lstm.pth'))
model.eval()  # Set to evaluation mode

# Function to preprocess and predict new data in small batches
def predict_activity(file_path, time_steps=50, batch_size=32):
    df = pd.read_csv(file_path)

    # Select required columns (remove time and mp23db01hp_mic [Waveform] column)
    feature_columns = [col for col in df.columns if col not in ['Time [s]']]
    df = df[feature_columns].dropna()

    # Standardize data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.values.astype(np.float32))

    # Prepare input sequences
    X = [df_scaled[i:i+time_steps] for i in range(len(df_scaled) - time_steps)]
    X = torch.tensor(np.array(X), dtype=torch.float32)

    # Create DataLoader for batch processing
    dataset = TensorDataset(X)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make predictions in batches
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch[0])
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())

    return predictions

# Example: Predict activities from a new file
file_path = r"Testing\Gesture\Gesture.csv"
predictions = predict_activity(file_path)

# Label mapping
label_map = {0: 'walking', 1: 'running', 2: 'falling_while_walking', 3: 'falling_while_running',4:'gesture'}

# Print predictions
predicted_labels = [label_map[p] for p in predictions]
print("Predicted activities:", predicted_labels[:20])  # Print first 20 predictions for reference
