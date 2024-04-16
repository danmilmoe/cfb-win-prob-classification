import os
import json
import ast
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

csv_dir = 'threads'
wp_dir = 'win_probs'
drive_dir = 'drive_win_probs'


'''
BINNING POLICIES

All of these functions (besides the first):

- Accept a filename, e.g. michigan_ohiostate.json


 will return lists of tuples like so:

[(START_UTC, END_UTC, WP_DELTA), ...]
'''

# return the binning function
def get_binning_policy(policy: str):
    if policy == 'activity_spike':
        return bin_by_activity_spike
    elif policy == 'wp_swings':
        return bin_by_wp_swings
    elif policy == 'constant_spacing':
        return bin_by_utc_interval
    elif policy == 'drive':
        return bin_by_drive
    else:
        print(f"INVALID BINNING POLICY: {policy}")


def bin_by_activity_spike(filename):
    # Step 1: Load the data efficiently
    csv_path = os.path.join(csv_dir, filename[:-5] + '.csv')
    json_path = os.path.join(wp_dir, filename)
    print(f"Binning by spike for {csv_path}")

    try:
        # Only read the 'created_utc' column to save memory
        utc_times = pd.read_csv(csv_path, usecols=['created_utc'], squeeze=True, dtype={'created_utc': np.int64})
    except Exception as e:
        print(f"Error loading or processing data from {csv_path}: {e}")
        return

    # Binning the timestamps
    binned_timestamps = np.bincount(utc_times - utc_times.min())

    x = np.arange(len(binned_timestamps))

    # Polynomial fitting
    if len(x) > N_EVENTS + 1:
        try:
            print(f"Fitting a polynomial of degree {N_EVENTS + 1}")
            coeffs = np.polyfit(x, binned_timestamps, N_EVENTS + 1)
            poly = np.poly1d(coeffs)
        except Exception as e:
            print("Error in polynomial fitting:", e)
            return

        if vis:
            plt.plot(x, binned_timestamps, 'o', label='Data points')
            plt.plot(x, poly(x), '-', label=f'Polynomial Fit (Degree {N_EVENTS + 1})')
            plt.legend()
            plt.show()

        # Step 4: Find derivatives and roots to determine peaks
        poly_deriv = np.polyder(poly)
        critical_points = np.roots(poly_deriv)
        real_critical_points = critical_points[np.isreal(critical_points)].real

        # Filter points to those within the range of x
        valid_critical_points = real_critical_points[(real_critical_points >= 0) & (real_critical_points <= max(x))]

        # Find which are peaks by checking second derivative
        poly_second_deriv = np.polyder(poly, 2)
        peaks = [cp for cp in valid_critical_points if poly_second_deriv(cp) < 0]

        print(f"Polynomial coefficients: {coeffs}")
        print(f"Critical points (real, within range): {valid_critical_points}")
        print(f"Peak points: {peaks}")
    else:
        print("Insufficient data points for polynomial fitting")
        return []

    # Reading play data from JSON
    with open(json_path, 'r') as json_file:
        wp_data = json.load(json_file)

    play_utcs = [item["utc"] for item in wp_data]
    play_wp = [item["home_win_prob"] for item in wp_data]

    intervals = []

    t_0 = utc_times.min()
    for peak in peaks:
        idxa = np.argmin(np.abs(np.array(play_utcs) - (t_0 + peak - UTC_INTERVAL_PRIOR)))
        idxb = np.argmin(np.abs(np.array(play_utcs) - (t_0 + peak + UTC_INTERVAL_POST)))

        wp_delta = play_wp[idxa] - play_wp[idxb]

        intervals.append((int(t_0 + peak - UTC_INTERVAL_PRIOR), int(t_0 + peak + UTC_INTERVAL_POST), wp_delta))
        print(f"Activity Spike from {intervals[-1][0]} to {intervals[-1][1]}, wp_delta = {wp_delta}")

    return intervals


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
    return ternary_classifier, 3

def ternary_classifier(val, threshold):
    # Store each target
    if abs(val) < threshold:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

