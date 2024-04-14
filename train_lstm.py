import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


# Initialize lists to hold the features and target values
features = []
targets = []

# Load and process each file
for filename in os.listdir('binned_ds_spiked_1'):
    with open(os.path.join('binned_ds_spiked_1', filename), 'r') as file:
        data = json.load(file)
        starting_win_prob = data['starting_win_prob']
        final_win_prob = data['ending_win_prob']

        sequences = []
        for point in data['game_datapoints']:
            home_vals = list(point['home_vals'].values())
            away_vals = list(point['away_vals'].values())
            neut_vals = list(point['neut_vals'].values())
            feature = np.concatenate((home_vals, away_vals, neut_vals), axis=None)
            sequences.append(feature)

        features.append(sequences)
        targets.append(final_win_prob)  # Store as a regression target
print(len(features))
print(len(targets))

# Pad sequences to ensure they all have the same length
features_padded = pad_sequences(features, padding='post', dtype='float32')

# Convert to numpy array
features = np.array(features_padded)
targets = np.array(targets, dtype=np.float32)

# Normalize features
scaler = MinMaxScaler()
num_samples, num_timesteps, num_features = features.shape
features = scaler.fit_transform(features.reshape(-1, num_features)).reshape(num_samples, num_timesteps, num_features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=69420)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)  # Output layer for regression
])

# Compile the model with mean squared error loss
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)  # Clipvalue limits the gradient values
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
loss, mse = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MSE: {mse}")

# Predict using the model
predicted_probs = model.predict(X_test)
print(predicted_probs)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], 'o', label='Actual values', markersize=5)  # Plot first 100 actual values
plt.plot(predicted_probs[:100], '.-', label='Predictions', markersize=5)  # Plot first 100 predictions
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Win Probability')
plt.legend()
plt.grid(True)
plt.show()
