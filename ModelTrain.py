import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# File paths
file_paths = {
    'walking': r"Dataset\walking\walking.csv",
    'running': r"Dataset\running\running.csv",
    'falling_while_walking': r"Dataset\FallingWhileWalking\FallingWhileWalking.csv",
    'falling_while_running': r"Dataset\fallingWhileRunning\fallingWhileRunning.csv",
    'gesture': r"Dataset\Gesture\Gesture.csv"
}

# Identify common and missing columns
def get_common_columns(file_paths):
    column_sets = {label: set(pd.read_csv(path, nrows=1).columns) for label, path in file_paths.items()}
    all_columns = set.union(*column_sets.values())
    common_columns = set.intersection(*column_sets.values())
   
    print("Common columns across all files:", common_columns)
    for label, cols in column_sets.items():
        missing = all_columns - cols
        extra = cols - common_columns
        print(f"For {label}: Missing columns: {missing}, Extra columns: {extra}")
   
    return common_columns

# Custom dataset class for memory efficiency
class SensorDataset(Dataset):
    def __init__(self, file_paths, common_columns, time_steps=50):
        self.data = []
        self.labels = []
        self.time_steps = time_steps
       
        label_map = {label: i for i, label in enumerate(file_paths.keys())}
        scaler = StandardScaler()
       
        # Load and preprocess each file
        for label, path in file_paths.items():
            df = pd.read_csv(path, usecols=common_columns).dropna()
            df_scaled = scaler.fit_transform(df.values.astype(np.float32))
           
            for i in range(len(df_scaled) - time_steps):
                self.data.append(df_scaled[i:i+time_steps])
                self.labels.append(label_map[label])
       
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Get common columns
common_columns = get_common_columns(file_paths)
common_columns.discard('Time [s]')  # Remove time column if present

# Create dataset
dataset = SensorDataset(file_paths, common_columns)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM model
class ActivityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActivityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
   
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Model parameters
input_size = len(common_columns)
hidden_size = 64
num_classes = len(file_paths)
model = ActivityLSTM(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.2f}')

# Save model
torch.save(model.state_dict(), 'activity_recognition_lstm.pth')
