import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os
import json

THRESHOLD = 0.07


features = []  # Input features
targets = []   # Target variable (sign of wp_delta)

for filename in os.listdir('pivot_repo/binned_ds_9'):
    with open(os.path.join('pivot_repo/binned_ds_9', filename), 'r') as json_file:
        data = json.load(json_file)

    for interval in data:
        home_vals = list(interval['home_vals'].values())
        away_vals = list(interval['away_vals'].values())
        neut_vals = list(interval['neut_vals'].values())
        
        feature = np.concatenate((home_vals, away_vals, neut_vals), axis=None)
        features.append(feature)  # Append the combined features as one row

        # Store each target
        if abs(interval['wp_delta']) < THRESHOLD:
            target = 0
        elif interval['wp_delta'] > 0:
            target = 1  # Represents increase
        else:
            target = 2
        
        targets.append(target)

print(len(features))
print(len(targets))
print(THRESHOLD)
print(targets.count(0) / len(targets))
print(targets.count(1) / len(targets))
print(targets.count(2) / len(targets))
print(targets.count(3) / len(targets))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=69420)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Define the model for multi-class classification
model = Sequential([
    Dense(124, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(124, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model with categorical_crossentropy loss function
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the mode
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class indices

# Converting true labels from one-hot to indices for comparison
y_true = np.argmax(y_test, axis=1)

# Generating the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Optionally, print the confusion matrix
print("Confusion Matrix:")
print(cm)