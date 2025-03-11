import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import os

# File paths (Training dataset)
train_file_paths = {
    'walking': r"Dataset\walking\walking.csv",
    'running': r"Dataset\running\running.csv",
    'falling_while_walking': r"Dataset\FallingWhileWalking\FallingWhileWalking.csv",
    'falling_while_running': r"Dataset\fallingWhileRunning\fallingWhileRunning.csv",
    'gesture': r"Dataset\Gesture\Gesture.csv"
}

# File paths (Testing/Fine-tuning dataset)
fine_tune_file_paths = {
    'walking': r"Testing\walking\walking.csv",
    'running': r"Testing\running\running.csv",
    'falling_while_walking': r"Testing\FallingWhileWalking\FallingWhileWalking.csv",
    'falling_while_running': r"Testing\fallingWhileRunning\fallingWhileRunning.csv",
    'gesture': r"Testing\Gesture\Gesture.csv"
}

# Identify common columns
def get_common_columns(file_paths):
    column_sets = {label: set(pd.read_csv(path, nrows=1).columns) for label, path in file_paths.items()}
    all_columns = set.union(*column_sets.values())
    common_columns = set.intersection(*column_sets.values())

    print("Common columns across all files:", common_columns)
    return common_columns

# Custom dataset class
class SensorDataset(Dataset):
    def __init__(self, file_paths, common_columns, time_steps=50):
        self.data = []
        self.labels = []
        self.time_steps = time_steps

        label_map = {label: i for i, label in enumerate(file_paths.keys())}
        scaler = StandardScaler()

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

# Get common columns for training
common_columns_train = get_common_columns(train_file_paths)
common_columns_train.discard('Time [s]')

# Create training dataset
train_dataset = SensorDataset(train_file_paths, common_columns_train)
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

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
input_size = len(common_columns_train)
hidden_size = 64
num_classes = len(train_file_paths)

# Initialize model, loss, and optimizer
model = ActivityLSTM(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), 'activity_recognition_lstm.pth')
print("Model training complete. Saved as 'activity_recognition_lstm.pth'.")

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

# ---------------------------------------
# Fine-Tuning on New Dataset
# ---------------------------------------

# Load pre-trained model
model.load_state_dict(torch.load('activity_recognition_lstm.pth'))
model.train()

# Get common columns for fine-tuning dataset
common_columns_fine_tune = get_common_columns(fine_tune_file_paths)
common_columns_fine_tune.discard('Time [s]')

# Create fine-tuning dataset
fine_tune_dataset = SensorDataset(fine_tune_file_paths, common_columns_fine_tune)
fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=32, shuffle=True)

# Fine-tuning parameters
fine_tune_epochs = 3
fine_tune_lr = 0.0005
optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)

# Fine-tuning loop
for epoch in range(fine_tune_epochs):
    total_loss = 0
    for X_batch, y_batch in fine_tune_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Fine-Tune Epoch [{epoch+1}/{fine_tune_epochs}], Loss: {total_loss / len(fine_tune_loader):.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), 'activity_recognition_lstm_finetuned.pth')
print("Fine-tuning complete. Model saved as 'activity_recognition_lstm_finetuned.pth'.")

# Evaluate Fine-Tuned Model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in fine_tune_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

fine_tune_accuracy = correct / total
print(f'Fine-Tuned Test Accuracy: {fine_tune_accuracy:.2f}')