import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import json

THRESHOLD = 0.1


features = []  # Input features
targets = []   # Target variable (sign of wp_delta)

for filename in os.listdir('old_bin_ds/binned_ds_9'):
    with open(os.path.join('old_bin_ds/binned_ds_9', filename), 'r') as json_file:
        data = json.load(json_file)

    for interval in data:

        if abs(interval['wp_delta']) < THRESHOLD:
            continue

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
            target = -1
        
        targets.append(target)

print(len(features))
print(len(targets))
print(THRESHOLD)
print(targets.count(0) / len(targets))
print(targets.count(-1) / len(targets))
print(targets.count(1) / len(targets))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=69420)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# Define SVM model
model = SVC(kernel='linear', C=1.0)

# Fit the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Optionally, print the confusion matrix
print("Confusion Matrix:")
print(cm)