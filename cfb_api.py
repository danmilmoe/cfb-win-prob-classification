import cfbd
import json
import matplotlib.pyplot as plt

configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'j5Z7/ASYjq4l6PMJFJpY7WxwO5BtbY/U2TFBOVdziGKRvjI8WajGUnVRiUBGrIX8'
configuration.api_key_prefix['Authorization'] = 'Bearer'

games_api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
metrics_api_instance = cfbd.Api(cfbd.ApiClient(configuration))


games = games_api_instance.get_games(year=2023, team='michigan')
for game in games:
    print(f'id: {game._id} home: {game._home_team} away: {game._away_team}')
    wp = metrics_api_instance.get_win_probability_data(game_id=game._id)
    # print(wp)
    print(f'initial win percentage: {wp[0]._home_win_prob}')
    play_list = [play._play_id for play in wp]
    wp_list = [play._home_win_prob for play in wp]
    plot = plt.plot(play_list, wp_list)
    break



# print(games)