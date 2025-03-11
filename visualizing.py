import pandas as pd
import matplotlib.pyplot as plt

# Define file paths for training and testing datasets
data_files_train = {
    'walking': r"Dataset\walking\walking.csv",
    'running': r"Dataset\running\running.csv",
    'falling_while_walking': r"Dataset\FallingWhileWalking\FallingWhileWalking.csv",
    'falling_while_running': r"Dataset\fallingWhileRunning\fallingWhileRunning.csv",
    'gesture': r"Dataset\Gesture\Gesture.csv"
}

data_files_test = {
    'walking': r"Testing\walking\walking.csv",
    'running': r"Testing\running\running.csv",
    'falling_while_walking': r"Testing\FallingWhileWalking\FallingWhileWalking.csv",
    'falling_while_running': r"Testing\fallingWhileRunning\fallingWhileRunning.csv",
    'gesture': r"Testing\Gesture\Gesture.csv"
}

# Load data into dictionaries
data_train = {activity: pd.read_csv(path) for activity, path in data_files_train.items()}
data_test = {activity: pd.read_csv(path) for activity, path in data_files_test.items()}

# Function to plot data separately for each activity
def plot_sensor_data(sensor, ylabel, title):
    for activity in data_train.keys():
        plt.figure(figsize=(12, 6))
        plt.plot(data_train[activity]['Time [s]'], data_train[activity][sensor], label=f'Training {activity}', linestyle='solid')
        plt.plot(data_test[activity]['Time [s]'], data_test[activity][sensor], label=f'Testing {activity}', linestyle='dashed')
        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)
        plt.title(f'{title} - {activity}')
        plt.legend()
        plt.grid()
        plt.show()

# Plot Accelerometer, Gyroscope, and Magnetometer Data Separately
plot_sensor_data('lsm6dsv16x_acc_x [g]', 'Acceleration [g]', 'Comparison of Accelerometer X Data')
plot_sensor_data('lsm6dsv16x_gyro_x [dps]', 'Gyroscope [dps]', 'Comparison of Gyroscope X Data')
plot_sensor_data('lis2mdl_mag_x [G]', 'Magnetic Field [G]', 'Comparison of Magnetometer X Data')

# Plot Pressure and Temperature Separately for Each Activity
for activity in data_train.keys():
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(data_train[activity]['Time [s]'], data_train[activity]['lps22df_press [hPa]'], label=f'Training {activity} Pressure', linestyle='solid')
    ax1.plot(data_test[activity]['Time [s]'], data_test[activity]['lps22df_press [hPa]'], label=f'Testing {activity} Pressure', linestyle='dashed')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pressure [hPa]', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend()
    
    ax2 = ax1.twinx()
    ax2.plot(data_train[activity]['Time [s]'], data_train[activity]['stts22h_temp [°C]'], label=f'Training {activity} Temp', linestyle='solid')
    ax2.plot(data_test[activity]['Time [s]'], data_test[activity]['stts22h_temp [°C]'], label=f'Testing {activity} Temp', linestyle='dashed')
    ax2.set_ylabel('Temperature [°C]', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend()
    
    fig.suptitle(f'Comparison of Pressure and Temperature Data - {activity}')
    fig.tight_layout()
    plt.show()
