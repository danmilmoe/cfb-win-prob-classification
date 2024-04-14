import os
import json
import ast
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# How big are the bins?
INTERVAL_STEP_SIZE = 16
UTC_INTERVAL_PRIOR = 60
UTC_INTERVAL_POST = 120
N_EVENTS = 15


# Replace 'your_directory_path' with the path to your directory containing the JSON files
csv_dir = 'pivot_repo/threads'
wp_dir = 'pivot_repo/win_probs'
drive_dir = 'pivot_repo/drive_win_probs'
indic = 0
output_dir = f'pivot_repo/binned_ds_spiked_{indic}'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# These are ALL of the LIWC scores
all_columns = [
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

# These are the LIWC scores included in the dataset
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

def detect_outliers(y, lag=30, threshold=200, influence=0.5):
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


def bin_timestamps(timestamps):
    print("Binning timestamps...")
    if timestamps is None or len(timestamps) == 0:
        print("hmmm it was empty")
        return []

    # Initialize the timeline from min to max timestamp
    start = min(timestamps)
    end = max(timestamps)
    # Create an array of zeros with length equal to the range of timestamps + 1
    time_bins = [0] * (end - start + 1)
    
    # Count each timestamp
    for timestamp in timestamps:
        time_bins[timestamp - start] += 1  # Increment the count at the position offset by the start
    
    return time_bins


# BINNING POLICIES
def bin_by_activity_spike(filename, vis=False):
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


# Helper functions
def get_first_last_wp(filename):
    with open(os.path.join(wp_dir, filename), 'r') as json_file:
        data = json.load(json_file)

    hometeam, awayteam = filename[:-5].split('_')

    return data[0]["home_win_prob"], data[-1]["home_win_prob"]


def parse_labels(label):
    try:
        return ast.literal_eval(label)
    except:
        return []  # Handle errors or return an empty list if the label cannot be parsed

# Averages the sentiments of a list of datapoints
def average_sentiments(dataframe):
    if dataframe.empty:
        initial_values = {col: 0 for col in feature_columns}
        initial_values['comment_count'] = 0
        return initial_values

    average_values = dataframe[feature_columns].mean().to_dict()
    average_values['comment_count'] = len(dataframe)
    return average_values


def separate_by_affiliation(
    file_path, 
    columns_to_keep, 
    start_utc, 
    end_utc,
    hometeam,
    awayteam
    ):
    home_comments = pd.DataFrame()
    away_comments = pd.DataFrame()
    neut_comments = pd.DataFrame()
    print(f"Separating by affiliation between {start_utc} and {end_utc}")

    try:
        # Read the file in chunks
        for chunk in pd.read_csv(file_path, chunksize=10000, usecols=columns_to_keep):
            print(f"Initial chunk shape: {chunk.shape}")

            chunk['created_utc'] = pd.to_numeric(chunk['created_utc'], errors='coerce')
            chunk['labels'] = chunk['labels'].apply(parse_labels)

            # Filter data within the desired time range
            filtered_chunk = chunk[(chunk['created_utc'] >= start_utc) & (chunk['created_utc'] <= end_utc)]
            print(f"Filtered chunk shape: {filtered_chunk.shape}")

            # Separate comments by affiliation
            home_chunk = filtered_chunk[filtered_chunk['labels'].apply(lambda x: hometeam in x)]
            away_chunk = filtered_chunk[filtered_chunk['labels'].apply(lambda x: awayteam in x)]
            neut_chunk = filtered_chunk[~filtered_chunk['labels'].apply(lambda x: hometeam in x or awayteam in x)]

            print(f"Home chunk shape: {home_chunk.shape}")
            print(f"Away chunk shape: {away_chunk.shape}")
            print(f"Neutral chunk shape: {neut_chunk.shape}")

            # Concatenate to final DataFrame
            home_comments = pd.concat([home_comments, home_chunk])
            away_comments = pd.concat([away_comments, away_chunk])
            neut_comments = pd.concat([neut_comments, neut_chunk])

            # Early exit if next chunk is beyond the end_utc
            if chunk['created_utc'].iloc[-1] > end_utc:
                print("Chunk end time is beyond end_utc, stopping iteration.")
                break

        print(f"#home:{len(home_comments)}, #away:{len(away_comments)}, #neut:{len(neut_comments)}")
        return home_comments, away_comments, neut_comments
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def bin_by_drive(filename):
    json_path = os.path.join(drive_dir, filename)
    print(f"Binning by drive for {filename}")

    with open(json_path, 'r') as json_file:
        drive_data = json.load(json_file)

    intervals = [(item["start_utc"], item["end_utc"], item["delta_win_prob"]) for item in drive_data]

    return intervals


# Construct the spiked dataset
def make_spiked_dataset(binning_policy):
    # These are the columns that we care about
    columns_to_keep = ['created_utc', 'labels'] + feature_columns
    game_datapoints = []

    # Loop through each thread
    for filename in os.listdir(wp_dir):
        if not filename.endswith('.json'):
            print(f"Skipping {filename}")
            continue

        file_path = os.path.join(csv_dir, filename[:-5] + ".csv")
        if not os.path.exists(file_path):
            continue

        hometeam, awayteam = filename[:-5].split('_')

        # Get the intervals according to 
        intervals = binning_policy(filename)
        
        # Iterate through the intervals
        print(intervals)
        

        for start_utc, end_utc, wp_delta in intervals:
            # Process CSV in chunks
            try:
                home_comments, away_comments, neut_comments = separate_by_affiliation(file_path, columns_to_keep, start_utc, end_utc, hometeam, awayteam)

                datapoint = {
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                    "home_vals": average_sentiments(home_comments),
                    "away_vals": average_sentiments(away_comments),
                    "neut_vals": average_sentiments(neut_comments),
                    "wp_delta": wp_delta
                }

                game_datapoints.append(datapoint)
            
            except pd.errors.EmptyDataError:
                print(f'Skipping empty or invalid file: {file_path}')
                continue

        # Create output dir
        json_output_dir = os.path.join(output_dir, f'{hometeam}_{awayteam}.json')

        # Fill the output object
        first, last = get_first_last_wp(filename)

        output_dict = {
            "starting_win_prob": first,
            "ending_win_prob": last,
            "game_datapoints": game_datapoints
        }

        with open(json_output_dir, 'w') as outfile:
            json.dump(output_dict, outfile, indent=4)



# Run the script
if __name__ == "__main__":
    make_spiked_dataset(bin_by_drive) 