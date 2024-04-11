import os
import json
import ast
import pandas as pd

# How big are the bins?
INTERVAL_STEP_SIZE = 16

# Replace 'your_directory_path' with the path to your directory containing the JSON files
csv_dir = 'pivot_repo/threads_all_columns'
json_dir = 'pivot_repo/win_probs'
output_path = f'pivot_repo/binned_ds_{INTERVAL_STEP_SIZE}'

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)


sentiment_columns = ['tone_pos', 'tone_neg', 'emotion', 'emo_pos', 'emo_neg', 'emo_anx', 'emo_anger', 'emo_sad']

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
    'netspeak', 'assent', 'nonflu', 'filler', 'AllPunc', 'Period', 'Comma', 'QMark', 'Exclam', 'Apostro', 'OtherP',
    'Emoji'
]

# Function to compute the average sentiment values for a list of DataFrames
def average_sentiments(dataframes):
    # Initialize the dictionary with sentiment columns set to 0
    initial_values = {col: 0 for col in all_columns}
    # Add the 'comment_count' key with a value of 0
    initial_values['comment_count'] = 0

    if not dataframes or dataframes[0].empty:
        return initial_values
    df_concat = pd.concat(dataframes)
    if df_concat.empty:
        return initial_values
    
    # Compute the average for sentiment columns
    average_values = df_concat[all_columns].mean().to_dict()
    # Add the count of comments to the average_values dictionary
    average_values['comment_count'] = len(df_concat)
    return average_values

# Loop through each file in the directory
for filename in os.listdir(json_dir):
    hometeam, awayteam = '', ''
    game_datapoints = []
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        with open(os.path.join(json_dir, filename), 'r') as json_file:
            data = json.load(json_file)

        hometeam, awayteam = filename[:-5].split('_')
        
        play_utcs = [item["utc"] for item in data]
        play_wp = [item["home_win_prob"] for item in data]
        n_comments_in_bin = []

        # Make the intervals
        intervals = []
        for i in range(0, len(play_utcs) - INTERVAL_STEP_SIZE, INTERVAL_STEP_SIZE):
            wp_delta = play_wp[i + INTERVAL_STEP_SIZE - 1] - play_wp[i]
            intervals.append([play_utcs[i], play_utcs[i + INTERVAL_STEP_SIZE - 1], wp_delta])
            n_comments_in_bin.append((0, 0, 0))

        # Corresponding CSV file name (adjust the naming scheme as necessary)
        csv_filename = filename.replace('.json', '.csv')
        file_path = os.path.join(csv_dir, csv_filename)

        columns_to_keep = ['created_utc', 'labels'] + all_columns
        
        # Check if corresponding CSV file exists
        if os.path.exists(file_path):
            for li in intervals:
                start_utc = li[0]
                end_utc = li[1]
                wp_delta = li[2]

                # Initialize comment lists for the current interval to ensure they're not carrying over data from previous intervals
                home_comments = []
                away_comments = []
                neut_comments = []
                
                # Process CSV in chunks
                for chunk in pd.read_csv(file_path, chunksize=10000, usecols=columns_to_keep):
                    chunk['created_utc'] = pd.to_numeric(chunk['created_utc'], errors='coerce')  # Convert to numeric, make errors NaN
                    chunk['labels'] = chunk['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)  # Safely evaluate strings

                    filtered_chunk = chunk[(chunk['created_utc'] >= start_utc) & (chunk['created_utc'] <= end_utc)]
                    
                    home_chunk = filtered_chunk[filtered_chunk['labels'].apply(lambda x: hometeam in x)]
                    away_chunk = filtered_chunk[filtered_chunk['labels'].apply(lambda x: awayteam in x)]
                    # Neutral comments should not contain either home or away team labels
                    neut_chunk = filtered_chunk[~filtered_chunk['labels'].apply(lambda x: hometeam in x or awayteam in x)]
                    
                    if not neut_chunk.empty:
                        neut_comments.append(neut_chunk)
                    if not home_chunk.empty:
                        home_comments.append(home_chunk)
                    if not away_chunk.empty:
                        away_comments.append(away_chunk)
                        
                # Create a datapoint for the current interval and append it to the game_datapoints list
                datapoint = {
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                    "home_vals": average_sentiments(home_comments),
                    "away_vals": average_sentiments(away_comments),
                    "neut_vals": average_sentiments(neut_comments),
                    "wp_delta": wp_delta
                }
                game_datapoints.append(datapoint)

        else:
            print(f'Corresponding CSV file not found for {filename}')
    else:
        print(f'Non-JSON file found: {filename}')
    
    global_maxima = {col: 0 for col in all_columns + ['comment_count']}

    # Aggregate the maximum values for each sentiment attribute
    for datapoint in game_datapoints:
        for vals_type in ['home_vals', 'away_vals', 'neut_vals']:
            for col in all_columns + ['comment_count']:
                current_val = datapoint[vals_type].get(col, 0)
                global_maxima[col] = max(global_maxima[col], current_val)

    # (Normalization Code Here)
    # Normalize the values across all datapoints
    for datapoint in game_datapoints:
        for vals_type in ['home_vals', 'away_vals', 'neut_vals']:
            for col in all_columns + ['comment_count']:
                if global_maxima[col] > 0:  # Ensure no division by zero
                    datapoint[vals_type][col] = datapoint[vals_type].get(col, 0) / global_maxima[col]

    # Save the game's datapoints to a JSON file
    json_output_path = os.path.join(output_path, f'{hometeam}_{awayteam}.json')
    with open(json_output_path, 'w') as outfile:
        json.dump(game_datapoints, outfile, indent=4)
    