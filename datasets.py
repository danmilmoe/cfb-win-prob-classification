import os
import json
import ast
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from policies import *
from helper import *
from dnn import *
from sklearn.decomposition import PCA


csv_dir = 'threads'
wp_dir = 'win_probs'
drive_dir = 'drive_win_probs'

BINNING_POLICY = 'spike'
NORM_POLICY = 'standard'
CLASS_POLICY = 'ternary'


THRESHOLD = 0.05
random_state = 345
BATCH_SIZE = 32
DROPOUT_RATE = 0.2

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

def make_dataset(output_dir):
    binning_policy_func = get_binning_policy(BINNING_POLICY)
    norm_policy_func = get_norm_policy(NORM_POLICY)

    # These are the columns that we care about
    columns_to_keep = ['created_utc', 'labels'] + feature_columns

    # Loop through each thread
    for filename in os.listdir(drive_dir):
        game_datapoints = []

        if not filename.endswith('.json'):
            print(f"Skipping {filename}")
            continue

        file_path = os.path.join(csv_dir, filename[:-5] + ".csv")
        if not os.path.exists(file_path):
            continue

        hometeam, awayteam = filename[:-5].split('_')

        # Get the intervals according to 
        intervals = binning_policy_func(filename)

        # Iterate through the intervals

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        json_output_dir = os.path.join(output_dir, f'{hometeam}_{awayteam}.json')

        # Fill the output object
        first, last = get_first_last_wp(filename)

        game_datapoints = norm_policy_func(game_datapoints)
        
        output_dict = {
            "starting_win_prob": first,
            "ending_win_prob": last,
            "game_datapoints": game_datapoints
        }


        with open(json_output_dir, 'w') as outfile:
            json.dump(output_dict, outfile, indent=4)
