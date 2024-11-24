import numpy as np
import pandas as pd
import os
import json

# Set seed for reproducibility
np.random.seed(0)

# Load and shuffle the data
df = pd.read_csv("mnist.csv")
df = df.values
np.random.shuffle(df)
labels = df[:, 0]
images = df[:, 1:]

num_clients = 5
samples_per_client = 1100  # 1000 train + 100 test
train_samples_per_client = 1000
test_samples_per_client = 100

# Function to save client data to JSON
def save_client_data(directory, client_idx, x_train, y_train, x_test, y_test):
    client_data = {
        'x_train': x_train.tolist(),
        'y_train': y_train.tolist(),
        'x_test': x_test.tolist(),
        'y_test': y_test.tolist()
    }
    print(y_train, y_test, directory, client_idx)
    client_filename = os.path.join(directory, f'client{client_idx}.json')
    with open(client_filename, 'w') as f:
        json.dump(client_data, f)

# Create IID data
iid_dir = 'iid'
os.makedirs(iid_dir, exist_ok=True)

for client_idx in range(num_clients):
    start_idx = client_idx * samples_per_client
    end_idx = start_idx + samples_per_client
    client_images = images[start_idx:end_idx]
    client_labels = labels[start_idx:end_idx]
    x_train = client_images[:train_samples_per_client]
    y_train = client_labels[:train_samples_per_client]
    x_test = client_images[train_samples_per_client:]
    y_test = client_labels[train_samples_per_client:]
    save_client_data(iid_dir, client_idx + 1, x_train, y_train, x_test, y_test)

# Create non-IID data
non_iid_dir = 'non-iid'
os.makedirs(non_iid_dir, exist_ok=True)

label_groups = {
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
    4: [6, 7],
    5: [8, 9]
}

for client_idx in range(1, num_clients + 1):
    client_labels_list = label_groups[client_idx]
    idx = np.isin(labels, client_labels_list)
    client_images = images[idx]
    client_labels = labels[idx]
    # Shuffle client data
    client_data_combined = np.c_[client_labels, client_images]
    np.random.shuffle(client_data_combined)
    client_images = client_data_combined[:, 1:]
    client_labels = client_data_combined[:, 0]
    # Take 1100 samples
    client_images = client_images[:samples_per_client]
    client_labels = client_labels[:samples_per_client]
    x_train = client_images[:train_samples_per_client]
    y_train = client_labels[:train_samples_per_client]
    x_test = client_images[train_samples_per_client:]
    y_test = client_labels[train_samples_per_client:]
    save_client_data(non_iid_dir, client_idx, x_train, y_train, x_test, y_test)