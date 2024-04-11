import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import json
import os

# Assuming 'data' is already loaded with the sentiment and wp_delta data
# and 'win_percent_data' with the win probability changes


# Preprocess the input data: Normalize the sentiment scores and calculate the sign of wp_delta
features = []  # Input features
targets = []   # Target variable (sign of wp_delta)

for filename in os.listdir('pivot_repo/binned_ds_20'):
    with open(os.path.join('pivot_repo/binned_ds_20', filename), 'r') as json_file:
        data = json.load(json_file)

    for interval in data:
        home_vals = list(interval['home_vals'].values())
        away_vals = list(interval['away_vals'].values())
        neut_vals = list(interval['neut_vals'].values())
        
        feature = np.concatenate((home_vals, away_vals, neut_vals), axis=None)
        features.append(feature)  # Append the combined features as one row

        # Store each target
        target = np.sign(interval['wp_delta'])
        targets.append(target)

print(len(features))
print(len(targets))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Normalizing the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshaping input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))  # 'sigmoid' for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Converting targets to be compatible with binary classification
y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)

# Now perform the conversion. This operation is valid for numpy arrays,
# and will correctly map -1 to 0, and 1 to 1.
y_train_binary = (y_train + 1) // 2
y_test_binary = (y_test + 1) // 2

model.fit(X_train, y_train_binary, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
model.evaluate(X_test, y_test_binary)
