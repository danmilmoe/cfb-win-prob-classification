import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os
import json
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

def getaccuracy(X, y_pred, y_true, threshold):
    total = 0.0
    correct = 0.0
    for i in range(y_true.shape[0]):
        actual_delta = y_true[i] - X[i][-1]
        pred_delta = y_pred[i] - X[i][-1]
        pred_bin, actual_bin = 0
        if actual_delta > threshold:
            actual_bin = 1
        elif actual_delta < -1*threshold:
            actual_bin = -1
        if pred_delta > threshold:
            pred_bin = 1
        elif pred_delta < -1*threshold:
            pred_bin = -1
        total += 1
        if pred_bin == actual_bin:
            correct += 1
    return correct / total
    




features = []  # Input features
targets = []   # Target variable (sign of wp_delta)

for filename in os.listdir('binned_ds_16'):
    with open(os.path.join('binned_ds_16', filename), 'r') as json_file:
        data = json.load(json_file)

    prev_wp = data['starting_win_prob']
    for interval in data['game_datapoints']:
        home_vals = list(interval['home_vals'].values())
        away_vals = list(interval['away_vals'].values())
        neut_vals = list(interval['neut_vals'].values())
        
        feature = np.concatenate((home_vals, away_vals, neut_vals, np.array([prev_wp])), axis=None)
        features.append(feature)  # Append the combined features as one row

        # Store each target
        prev_wp = prev_wp + interval['wp_delta']
        targets.append(prev_wp)


print(len(features))
print(features[0])
print(len(targets))
print(targets[0])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(np.array(features)), np.nan_to_num(np.array(targets)), test_size=0.2, random_state=69420)
print(np.any(np.isnan(X_train)))
# Normalize features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define the DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_train_pred = model.predict(X_train)
train_accuracy = getaccuracy(X_train, y_train_pred, y_train, .05)

y_test_pred = model.predict(X_test)
test_accuracy = getaccuracy(X_test, y_test_pred, y_test, .05)

# Evaluate the model
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


    

