import requests
import matplotlib.pyplot as plt
import json

api_key = 'Bearer HcHGxJdzBhWhYXn+kaTMzYz9tpeCy4mIgVru8dZhMtjJphuLJ9PKnYnxM5ZOca2F'

def get_wp_chart(game_id='401520368'):
    url = f'https://api.collegefootballdata.com/metrics/wp?gameId={game_id}'

    # Optional: If the API requires an API key for access, include it in the headers
    headers = {
        'Authorization': api_key, # Replace YOUR_API_KEY with your actual API key
    }

    # Perform the GET request
    response = requests.get(url, headers=headers)

    data = response.json() if response.status_code == 200 else []
    play_numbers = [play['playNumber'] for play in data]
    home_win_probs = [float(play['homeWinProb']) for play in data]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(play_numbers, home_win_probs, marker='o', linestyle='-', color='b')
    plt.title('Home Win Probability vs. Play Number')
    plt.xlabel('Play Number')
    plt.ylabel('Home Win Probability')
    plt.grid(True)
    plt.show()


# Query for the list of plays throughout the game
def get_plays_for_game(season_type = 'regular', year = 2021, week = 1, team='michigan'):
    # The API endpoint URL
    url = f'https://api.collegefootballdata.com/plays?seasonType={season_type}&year={year}&week={week}&team={team}'

    # Optional: Include your API key in the headers if the API requires authentication
    headers = {
        'Authorization': api_key,  # Replace YOUR_API_KEY with your actual API key
    }

    # Perform the GET request
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        # For demonstration, let's just print the first 5 plays' data to keep the output manageable
        print(json.dumps(data, indent=4))
    else:
        print("Failed to retrieve data. Status code:", response.status_code)



get_plays_for_game(season_type='postseason', week=1, year=2023)
