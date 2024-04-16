import os
import json
import ast
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

csv_dir = 'threads'
wp_dir = 'win_probs'
drive_dir = 'drive_win_probs'

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

    try:
        # Read the file in chunks
        for chunk in pd.read_csv(file_path, chunksize=10000, usecols=columns_to_keep):
            chunk['created_utc'] = pd.to_numeric(chunk['created_utc'], errors='coerce')
            chunk['labels'] = chunk['labels'].apply(parse_labels)

            # Filter data within the desired time range
            filtered_chunk = chunk[(chunk['created_utc'] >= start_utc) & (chunk['created_utc'] <= end_utc)]
            filtered_chunk = filtered_chunk[(filtered_chunk[feature_columns] != 0).sum(axis=1) >= 3]
            
            # Separate comments by affiliation
            home_chunk = filtered_chunk[filtered_chunk['labels'].apply(lambda x: hometeam in x)]
            away_chunk = filtered_chunk[filtered_chunk['labels'].apply(lambda x: awayteam in x)]
            neut_chunk = filtered_chunk[~filtered_chunk['labels'].apply(lambda x: hometeam in x or awayteam in x)]

            # Concatenate to final DataFrame
            home_comments = pd.concat([home_comments, home_chunk])
            away_comments = pd.concat([away_comments, away_chunk])
            neut_comments = pd.concat([neut_comments, neut_chunk])

            # Early exit if next chunk is beyond the end_utc
            if chunk['created_utc'].iloc[-1] > end_utc:
                break

        return home_comments, away_comments, neut_comments
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
