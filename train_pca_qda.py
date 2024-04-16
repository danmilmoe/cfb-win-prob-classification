import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Initialize lists to hold the features and target values
features = []
targets = []

# Load and process each file
for filename in os.listdir('binned_ds_spiked_3'):
    with open(os.path.join('binned_ds_spiked_3', filename), 'r') as file:
        data = json.load(file)
        starting_win_prob = data['starting_win_prob']
        final_win_prob = data['ending_win_prob']
        sequences = [starting_win_prob]  # Start sequence with the initial win probability
        for point in data['game_datapoints']:
            home_vals = list(point['home_vals'].values())
            away_vals = list(point['away_vals'].values())
            neut_vals = list(point['neut_vals'].values())
            feature = np.concatenate((home_vals, away_vals, neut_vals), axis=None)
            sequences.extend(feature)
        features.append(sequences)
        targets.append(final_win_prob)

# Convert regression targets into binary classification targets
threshold = 0.5
binary_targets = [1 if x >= threshold else 0 for x in targets]
targets_array = np.array(binary_targets)

max_len = 90  # You might need to adjust this based on your actual data

# Pad sequences
features_padded = pad_sequences(features, padding='post', maxlen=max_len, dtype='float32')

# Convert to numpy array
features_array = np.array(features_padded)
features_array = np.nan_to_num(features_array)  # Handling NaNs if any

# Check the shape of the array
print("Shape of features array:", features_array.shape)

# Now apply StandardScaler
from sklearn.preprocessing import StandardScaler
# Normalize features using StandardScaler
scaler = StandardScaler()
features_array = np.vstack([scaler.fit_transform(features_array[i].reshape(-1, 1)).flatten() for i in range(len(features_array))])

# Define the PCA and QDA pipeline
pca_qda_pipeline = Pipeline([
    ('pca', PCA(n_components=5)),  # Optimal number of components
    ('qda', QuadraticDiscriminantAnalysis(reg_param=1.0))  # Optimal regularization parameter
])

# Set up different seeds for training
num_seeds = 1000
seeds = [random.randint(0, 100000) for _ in range(num_seeds)]
results = []

for seed in seeds:
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features_array, targets_array, test_size=0.2, random_state=seed)
    
    # Fit and predict using the pipeline
    pca_qda_pipeline.fit(X_train, y_train)
    y_pred = pca_qda_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((seed, accuracy))
    print(f"Seed: {seed}, Test Accuracy: {accuracy}")

accuracies = [acc for seed, acc in results]
average_accuracies = np.mean(accuracies)
print(average_accuracies)

# Create a box plot
plt.figure(figsize=(10, 6))
plt.boxplot(accuracies, patch_artist=True)
plt.title('1000 Seeds in a Pipelined PCA + QDA Model')
plt.ylabel('Test Accuracy')
plt.xlabel('Model')
plt.grid(True)
plt.show()