import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_column_distributions(data, cols, figsize=(12, 8)):
    plt.close("all")
    _, axes = plt.subplots(nrows=4, ncols=int(np.ceil(len(cols) // 4)), figsize=figsize)
    for ax, cname in zip(axes.ravel(), cols):
        sns.histplot(data[cname], kde=True, ax=ax, shrink=1, discrete=True)
    plt.tight_layout()

def plot_goal_difference_distribution(data):
    plt.close("all")
    plt.figure(figsize=(12, 8))
    sns.histplot(data['GD'], kde=True, shrink=1, discrete=True)
    plt.title('Goal Difference Distribution')
    plt.xlabel('Goal Difference')
    plt.ylabel('Frequency')
    plt.show()


# def create_league_table(paths : list[str] = None) -> pd.DataFrame:
#     if paths is None:
#         paths = [os.path.join(os.path.dirname(__file__), f'../../data/{file}') for file in os.listdir(os.path.join(os.path.dirname(__file__), '../../data')) if file.endswith('.csv')]

#     league_tables = {}

#     for path in paths:
#         df = pd.read_csv(path)

#         teams_stats = {}

#         for _, row in df.iterrows():
#             home_team = row['HomeTeam']
#             away_team = row['AwayTeam']
#             home_goals = row['FTHG']
#             away_goals = row['FTAG']
#             result = row['FTR']

#             if home_team not in teams_stats:
#                 teams_stats[home_team] = {'Played': 0, 'Won': 0, 'Drawn': 0, 'Lost': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Points': 0}
#             if away_team not in teams_stats:
#                 teams_stats[away_team] = {'Played': 0, 'Won': 0, 'Drawn': 0, 'Lost': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Points': 0}

#             teams_stats[home_team]['Played'] += 1
#             teams_stats[home_team]['GF'] += home_goals
#             teams_stats[home_team]['GA'] += away_goals
#             teams_stats[home_team]['GD'] = teams_stats[home_team]['GF'] - teams_stats[home_team]['GA']

#             teams_stats[away_team]['Played'] += 1
#             teams_stats[away_team]['GF'] += away_goals
#             teams_stats[away_team]['GA'] += home_goals
#             teams_stats[away_team]['GD'] = teams_stats[away_team]['GF'] - teams_stats[away_team]['GA']

#             if result == 'H':
#                 teams_stats[home_team]['Won'] += 1
#                 teams_stats[home_team]['Points'] += 3
#                 teams_stats[away_team]['Lost'] += 1
#             elif result == 'A':
#                 teams_stats[away_team]['Won'] += 1
#                 teams_stats[away_team]['Points'] += 3
#                 teams_stats[home_team]['Lost'] += 1
#             elif result == 'D':
#                 teams_stats[home_team]['Drawn'] += 1
#                 teams_stats[home_team]['Points'] += 1
#                 teams_stats[away_team]['Drawn'] += 1
#                 teams_stats[away_team]['Points'] += 1

#         league_table = pd.DataFrame(teams_stats).T.reset_index()
#         league_table.columns = ['Team', 'Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points']
        
#         league_table = league_table.sort_values(by=['Points', 'GD'], ascending=[False, False]).reset_index(drop=True)

#         league_tables[path.split('/')[-1].split('.')[0][3:]] = league_table
    
#     return league_tables

