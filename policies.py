import os
import json
import ast
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

csv_dir = 'threads'
wp_dir = 'win_probs'
drive_dir = 'drive_win_probs'


THRESHOLD = 0.006

feature_columns = [
    'Segment', 'WC', 'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'BigWords', 'Dic', 'Linguistic', 'function',
    'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'det', 'article', 'number', 'prep', 'auxverb',
    'adverb', 'conj', 'negate', 'verb', 'adj', 'quantity', 'Drives', 'affiliation', 'achieve', 'power', 'Cognition',
    'allnone', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certitude', 'differ', 'memory', 'Affect',
    'tone_pos', 'tone_neg', 'emotion', 'emo_pos', 'emo_neg', 'emo_anx', 'emo_anger', 'emo_sad', 'swear', 'Social',
    'socbehav', 'prosocial', 'polite', 'conflict', 'moral', 'comm', 'socrefs', 'family', 'friend', 'female', 'male',
    'Culture', 'politic', 'ethnicity', 'tech', 'Lifestyle', 'leisure', 'home', 'work', 'money', 'relig', 'Physical',
    'health', 'illness', 'wellness', 'mental', 'substances', 'sexual', 'food', 'death', 'need', 'want', 'acquire',
    'lack', 'fulfill', 'fatigue', 'reward', 'risk', 'curiosity', 'allure', 'Perception', 'attention', 'motion',
    'space', 'visual', 'auditory', 'feeling', 'time', 'focuspast', 'focuspresent', 'focusfuture', 'Conversation',
    'netspeak', 'assent', 'nonflu', 'filler', 'AllPunc', 'Period', 'Comma', 'QMark', 'Exclam', 'Apostro', 'OtherP'
]

'''
BINNING POLICIES

All of these functions (besides the first):

- Accept a filename, e.g. michigan_ohiostate.json


 will return lists of tuples like so:

[(START_UTC, END_UTC, WP_DELTA), ...]
'''

# return the binning function
def get_binning_policy(policy: str):
    if policy == 'spike':
        return bin_by_activity_spike
    elif policy == 'wp_swings':
        return bin_by_wp_swings
    elif policy == 'constant_spacing':
        return bin_by_utc_interval
    elif policy == 'drive':
        return bin_by_drive
    else:
        print(f"INVALID BINNING POLICY: {policy}")


def detect_outliers(y, lag=15, threshold=1.2, influence=0.6):
    # Initialize variables
    signals = np.zeros(len(y))
    filteredY = np.array(y[:lag])
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag-1] = np.mean(y[:lag])
    stdFilter[lag-1] = np.std(y[:lag])

    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1   # Positive signal
            else:
                signals[i] = -1  # Negative signal
            filteredY = np.append(filteredY, influence * y[i] + (1 - influence) * filteredY[i-1])
        else:
            signals[i] = 0     # No signal
            filteredY = np.append(filteredY, y[i])
        
        # Update the filters
        avgFilter[i] = np.mean(filteredY[i-lag+1:i+1])
        stdFilter[i] = np.std(filteredY[i-lag+1:i+1])

    return signals, np.sum(signals != 0)


def bin_by_activity_spike(filename):
    # Step 1: Load the data efficiently
    csv_path = os.path.join(csv_dir, filename[:-5] + '.csv')
    json_path = os.path.join(wp_dir, filename)
    print(f"Binning by spike for {csv_path}")

    try:
        # Only read the 'created_utc' column to save memory
        with open(json_path, 'r') as json_file:
            wp_data = json.load(json_file)

        # Reading play data from JSON
        play_utcs = [item["utc"] for item in wp_data]
        play_wp = [item["home_win_prob"] for item in wp_data]
        print(play_utcs[0], " -> ", play_utcs[-1])


        utc_times = np.array(pd.read_csv(csv_path, usecols=['created_utc'], squeeze=True, dtype={'created_utc': np.int64}))
        utc_times = utc_times[(utc_times > play_utcs[0]) & (utc_times < play_utcs[-1] + 30)]
    except Exception as e:
        print(f"Or its {json_path}")
        print(f"Error loading or processing data from {csv_path}: {e}")
        return

    # Binning the timestamps
    num_bins = 120

    min_time = utc_times.min()
    range_time = utc_times.max() - min_time
    norm_val = range_time / num_bins

    print(f'min={min_time}, max={utc_times.max()}, range={range_time}, norm_val={norm_val}')
    # Map timestamps to bins
    bin_indices = np.floor((utc_times - min_time) / norm_val).astype(int)
    print(bin_indices)
    # Count occurrences in each bin
    binned_timestamps = np.bincount(bin_indices, minlength=num_bins)
    print(binned_timestamps)

    signals, n_peaks = detect_outliers(binned_timestamps)

    print(np.where(signals == 1), "\n", n_peaks, flush=True)
    peak_bins = np.where(signals == 1)[0]  # Get indices of bins that are peaks
    intervals = []

    for bin_index in peak_bins:
        # Find the timestamps that fall into each peak bin
        lower_bound = min_time + bin_index * norm_val
        upper_bound = min_time + (bin_index + 1) * norm_val
        # Get indices of timestamps within the current peak bin range
        indices_in_bin = np.where((utc_times >= lower_bound) & (utc_times < upper_bound))[0]
        # Append the time range and indices
        intervals.append((lower_bound, upper_bound))

    intervals_with_wp = []
    for start, end in intervals:
        idxa = np.argmin(np.abs(play_utcs - start))  # Closest to start
        idxb = np.argmin(np.abs(play_utcs - end))    # Closest to end
        if idxa == idxb:
            wp_delta = 0  # Handle the case where start and end map to the same index
        else:
            wp_delta = play_wp[idxb] - play_wp[idxa]

        intervals_with_wp.append((int(start), int(end), wp_delta))
        print(f"Activity Spike from {int(start)} to {int(end)}, wp_delta = {wp_delta}")

    return intervals_with_wp


def bin_by_wp_swings(filename):
    # Read in the data containing the wp
    with open(os.path.join(wp_dir, filename), 'r') as json_file:    
        data = json.load(json_file) 

    # Get the utc and wp array
    play_utcs = [item["utc"] for item in data]
    play_wp = [item["home_win_prob"] for item in data]

    # Calculate changes in win probability
    wp_changes = [abs(play_wp[i+1] - play_wp[i]) for i in range(len(play_wp)-1)]

    # Identify large changes
    if threshold is None:
        # If no threshold is provided, use the top 5% largest changes
        threshold = sorted(wp_changes)[-max(1, len(wp_changes) // 20)]

    # Find the indexes of the changes that are above the threshold
    significant_changes_idx = [i for i, change in enumerate(wp_changes) if change >= threshold]

    # Get the corresponding UTCs for these significant changes
    intervals = [(play_utcs[i] - UTC_INTERVAL_PRIOR, play_utcs[i] + UTC_INTERVAL_POST) for i in significant_changes_idx]

    return intervals


def bin_by_utc_interval(filename):
    # Read in the data containing the wp
    with open(os.path.join(wp_dir, filename), 'r') as json_file:    
        data = json.load(json_file) 

    # Get the utc and wp array
    play_utcs = [item["utc"] for item in data]

    start_bin = data[0]["utc"]
    end_bin = data[-1]["utc"]


def bin_by_drive(filename):
    json_path = os.path.join(drive_dir, filename)
    print(f"Binning by drive for {filename}")

    with open(json_path, 'r') as json_file:
        drive_data = json.load(json_file)

    intervals = [(item["start_utc"], item["end_utc"], item["delta_win_prob"]) for item in drive_data]

    return intervals


'''
NORM_POLICIES

All of these functions (besides the first):

- Accept a list of game_datapoints
- Return a normalized version of those game_datapoints

'''

def get_norm_policy(policy: str):
    return norm_by_feature


def norm_by_feature(game_datapoints):
        global_maxima = {col: 0 for col in feature_columns + ['comment_count']}

        # Aggregate the maximum values for each sentiment attribute
        for datapoint in game_datapoints:
            for vals_type in ['home_vals', 'away_vals', 'neut_vals']:
                for col in feature_columns + ['comment_count']:
                    current_val = datapoint[vals_type].get(col, 0)
                    global_maxima[col] = max(global_maxima[col], current_val)

        # Normalize the values across all datapoints
        for datapoint in game_datapoints:
            for vals_type in ['home_vals', 'away_vals', 'neut_vals']:
                for col in feature_columns + ['comment_count']:
                    if global_maxima[col] > 0:  # Ensure no division by zero
                        datapoint[vals_type][col] = round(datapoint[vals_type].get(col, 0) / (global_maxima[col] + 1e-7), 5)

        return game_datapoints


'''
CLASSIFICATION_POLICIES

All of these functions (besides the first):
- Accept a value
- return a class number

'''
def get_class_info(policy: str):
    if policy == 'ternary':
        return ternary_classifier, 3
    elif policy == 'binary':
        return binary_classifier, 2
    else:
        return ternary_classifier, 3


def ternary_classifier(val):
    # Store each target
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2


def binary_classifier(val):
    # Store each target
    if val < 0:
        return 0
    else:
        return 1