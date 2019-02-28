import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

teams_df = pd.read_csv('./csv_data/Teams.csv')
team_conferences_df = pd.read_csv('./csv_data/TeamConferences.csv')
reg_season_results_df = pd.read_csv('./csv_data/RegularSeasonCompactResults.csv')

all_team_wins_df = pd.merge(teams_df[['TeamID','TeamName']], reg_season_results_df[['WTeamID', 'WScore', 'LTeamID', 'LScore', 'Season']], left_on="TeamID", right_on='WTeamID')

calc_teams = ['Kansas', 'Kansas St', 'Missouri']
calc_colors = ['blue', 'purple', 'yellow']
calc_results = []

for team_name in calc_teams:
    single_team_wins_df = all_team_wins_df[all_team_wins_df['TeamName'] == team_name]
    win_years = list(set(single_team_wins_df['Season']))

    avg_margin_victs = []
    for year in win_years:
        kansas_year_df = single_team_wins_df[single_team_wins_df["Season"] == year]
        margin_victory_year_df = kansas_year_df['WScore'] - kansas_year_df['LScore']
        total_margin_year = margin_victory_year_df.sum()
        avg_margin_year = (total_margin_year / (len(margin_victory_year_df)))
        avg_margin_victs.append(avg_margin_year)

    calc_results.append(avg_margin_victs)


plt.hist(calc_results, label=calc_teams, color=calc_colors)
plt.legend(loc='upper right')
plt.show()