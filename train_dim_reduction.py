import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import json
import numpy as np
import hypertools as hyp
from sklearn.decomposition import PCA

import warnings 

warnings.filterwarnings("ignore")

THRESHOLD = 0.047
RANDOM_STATE = 69
BATCH_SIZE = 64
DROPOUT_RATE = 0.2

features = []
targets = []

for filename in os.listdir('binned_ds_15_buffered_30'):
    with open(os.path.join('binned_ds_15_buffered_30', filename), 'r') as json_file:
        data = json.load(json_file)

    prev_wp = data['starting_win_prob']
    for interval in data["game_datapoints"]:
        home_vals = list(interval['home_vals'].values())
        away_vals = list(interval['away_vals'].values())
        neut_vals = list(interval['neut_vals'].values())
        
        feature = np.concatenate((home_vals, away_vals, neut_vals, np.array([prev_wp])), axis=None)
        features.append(feature)  # Append the combined features as one row

        # Store each target
        if abs(interval['wp_delta']) < THRESHOLD:
            target = 0
        elif interval['wp_delta'] > 0:
            target = 1  # Represents increase
        else:
            target = 2
        
        targets.append(target)

        prev_wp = prev_wp + interval['wp_delta']

f = np.array(features)
print(f.shape)

train_accuracies = []
test_accuracies = []

pca = PCA(.8)

for i in range(50):
    RANDOM_SEED = i*42
    # reduced = hyp.reduce(f, ndims = 16,  reduce={'model' : 'TruncatedSVD', 'params' : {'algorithm': 'randomized', 'random_state': RANDOM_SEED, 'n_iter':8, 'n_components':16}})
    #reduced = hyp.reduce(f, ndims = 16)
    #print('Shape of first reduced array: ', reduced.shape)
    # pca.fit(f)
    # reduced = pca.transform(f)

    #print(len(features))
    #print(len(targets))
    #print(targets.count(0) / len(targets))
    #print(targets.count(1) / len(targets))
    #print(targets.count(2) / len(targets))
    #print(targets.count(3) / len(targets))

    #features = reduced
    # Convert lists to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, targets, test_size=0.2, random_state=RANDOM_SEED)

    # Further splitting the training set into actual training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_SEED)  # Adjust the test_size as needed

    pca.fit(X_train)
    X_train = torch.tensor(pca.transform(X_train), dtype=torch.float32)
    X_val = torch.tensor(pca.transform(X_val), dtype=torch.float32)
    X_test = torch.tensor(pca.transform(X_test), dtype=torch.float32)

    LAYER_SIZES = [X_train.shape[1], 250, 150, 120, 90, 3]  # Include input size and output size

    # Creating PyTorch datasets for training, validation, and testing
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Calculate class weights for the training set
    class_sample_count = torch.tensor([(y_train == t).sum() for t in torch.unique(y_train, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in y_train])

    # Create a WeightedRandomSampler for the training set to handle class imbalance
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    # Print global variables before training
    # Creating data loaders for the training, validation, and testing sets
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # No need for sampling in validation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #print(f"\n-----\nBATCH_SIZE: {BATCH_SIZE}")
    #print(f"DROPOUT_RATE: {DROPOUT_RATE}")
    #print(f"LAYER_SIZES: {LAYER_SIZES}")
    #print(f"THRESHOLD: {THRESHOLD}\n-----\n")

    # Assuming X_train.shape[1] is defined elsewhere and is the input size
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layers = nn.ModuleList()
            for i in range(len(LAYER_SIZES) - 2):
                self.layers.append(nn.Linear(LAYER_SIZES[i], LAYER_SIZES[i+1]))
            self.dropout = nn.Dropout(DROPOUT_RATE)
            self.output = nn.Linear(LAYER_SIZES[-2], LAYER_SIZES[-1])

        def forward(self, x):
            for layer in self.layers:
                x = nn.functional.relu(layer(x))
            x = self.dropout(x)
            x = self.output(x)
            return x

    model = Net()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    patience = 5
    best_loss = float('inf')
    patience_counter = 0
    num_epochs = 60

    for epoch in range(num_epochs):
        model.train()  # Training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training phase
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions
        #print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()  # Evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        #print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # reset patience
        else:
            patience_counter += 1  # decrement patience

        if patience_counter >= patience:
            #print("Early stopping triggered")
            break
    train_accuracies.append(epoch_acc)
    # Evaluating the model
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = correct / total
    test_accuracies.append(accuracy)
    #print(f'Test Accuracy: {accuracy}')

# Confusion Matrix

print("avg train accuracy:")
print(sum(train_accuracies)/len(train_accuracies))
print("avg test accuracy:")
print(sum(test_accuracies)/len(test_accuracies))
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
