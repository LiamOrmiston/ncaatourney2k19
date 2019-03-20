import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import csv

from helpers import tourney_round

# Clean NCAA Tournament Seed Data
df_seeds = pd.read_csv('./csv_data/NCAATourneySeeds.csv')
df_seeds['seed'] = df_seeds['Seed'].apply(lambda x: int(x[1:3]))
df_seeds = df_seeds[['Season', 'TeamID', 'seed']]

# NCAA Tournament Results
df_tourney = pd.read_csv('./csv_data/NCAATourneyCompactResults.csv')
df_tourney = df_tourney.drop(['WLoc'], axis=1)

# Regular Season Results
df_regular_season = pd.read_csv('./csv_data/RegularSeasonDetailedResults.csv')
# Add to regular season result data
df_regular_season['WFGM2'] = df_regular_season.WFGM - df_regular_season.WFGM3
df_regular_season['WFGA2'] = df_regular_season.WFGA - df_regular_season.WFGA3
df_regular_season['LFGM2'] = df_regular_season.LFGM - df_regular_season.LFGM3
df_regular_season['LFGA2'] = df_regular_season.LFGA - df_regular_season.LFGA3

df_regular_season = df_regular_season.drop(['DayNum', 'WLoc', 'NumOT'], axis=1)


# Team and Conference Information
df_teams = pd.read_csv('./csv_data/Teams.csv')
df_team_conferences = pd.read_csv('./csv_data/TeamConferences.csv')
df_conferences = pd.read_csv('./csv_data/Conferences.csv')

df_conference_names = df_team_conferences.merge(df_conferences, on=['ConfAbbrev'])

# Merge information into Regular Season Results
df_win_team = df_teams.rename(columns={'TeamID':'WTeamID'})[['WTeamID','TeamName']]
df_win_confs = df_conference_names.rename(columns={'TeamID':'WTeamID'})[['Season', 'WTeamID', 'Description']]
df_lose_team = df_teams.rename(columns={'TeamID': 'LTeamID'})[['LTeamID', 'TeamName']]
df_lose_confs = df_conference_names.rename(columns={'TeamID':'LTeamID'})[['Season', 'LTeamID', 'Description']]

df_regular_season = df_regular_season.merge(df_win_team, on='WTeamID').rename(columns={'TeamName':'WTeamName'}) \
    .merge(df_win_confs, on=['Season', 'WTeamID']).rename(columns={'Description': 'WConfName'}) \
    .merge(df_lose_team, on='LTeamID').rename(columns={'TeamName': 'LTeamName'}) \
    .merge(df_lose_confs, on=['Season', 'LTeamID']).rename(columns={'Description': 'LConfName'})

df = df_regular_season

# Winner stats related to offensive efficiency:
df['Wposs'] = df.apply(lambda row: row.WFGA + 0.475 * row.WFTA + row.WTO - row.WOR, axis=1)
df['Wshoot_eff'] = df.apply(lambda row: row.WScore / (row.WFGA + 0.475 * row.WFTA), axis=1)
df['Wscore_op'] = df.apply(lambda row: (row.WFGA + 0.475 * row.WFTA) / row.Wposs, axis=1)
df['Woff_rtg'] = df.apply(lambda row: row.WScore/row.Wposs*100, axis=1)

# Loser stats related to offensive efficiency:
df['Lposs'] = df.apply(lambda row: row.LFGA + 0.475 * row.LFTA + row.LTO - row.LOR, axis=1)
df['Lshoot_eff'] = df.apply(lambda row: row.LScore / (row.LFGA + 0.475 * row.LFTA), axis=1)
df['Lscore_op'] = df.apply(lambda row: (row.LFGA + 0.475 * row.LFTA) / row.Lposs, axis=1)
df['Loff_rtg'] = df.apply(lambda row: row.LScore/row.Lposs*100, axis=1)

# Defensive and net efficiency:
df['Wdef_rtg'] = df.apply(lambda row: row.Loff_rtg, axis=1)
df['Wsos'] = df.apply(lambda row: row.Woff_rtg - row.Loff_rtg, axis=1)
df['Ldef_rtg'] = df.apply(lambda row: row.Woff_rtg, axis=1)
df['Lsos'] = df.apply(lambda row: row.Loff_rtg - row.Woff_rtg, axis=1)

# Impact Estimate - 
# First calculate the teams' overall statistical contribution (the numerator):
Wie = df.apply(lambda row: row.WScore + row.WFGM + row.WFTM - row.WFGA - row.WFTA + row.WDR + (0.5 * row.WOR) + row.WAst + row.WStl + (0.5 * row.WBlk) - row.WPF - row.WTO, axis=1)
Lie = df.apply(lambda row: row.LScore + row.LFGM + row.LFTM - row.LFGA - row.LFTA + row.LDR + (0.5 * row.LOR) + row.LAst + row.LStl + (0.5 * row.LBlk) - row.LPF - row.LTO, axis=1)

# Then divide by the total game statistics (the denominator):
df['Wie'] = Wie / (Wie + Lie) * 100
df['Lie'] = Lie / (Lie + Wie) * 100

# Other winner stats:
df['Wts_pct'] = df.apply(lambda row: row.WScore / (2 * (row.WFGA + 0.475 * row.WFTA)) * 100, axis=1)
df['Wefg_pct'] = df.apply(lambda row: (row.WFGM2 + 1.5 * row.WFGM3) / row.WFGA, axis=1)
df['Worb_pct'] = df.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
df['Wdrb_pct'] = df.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
df['Wreb_pct'] = df.apply(lambda row: (row.Worb_pct + row.Wdrb_pct) / 2, axis=1)
df['Wto_poss'] = df.apply(lambda row: row.WTO / row.Wposs, axis=1)
df['Wft_rate'] = df.apply(lambda row: row.WFTM / row.WFGA, axis=1)
df['Wast_rtio'] = df.apply(lambda row: row.WAst / (row.WFGA + 0.475*row.WFTA + row.WTO + row.WAst) * 100, axis=1)
df['Wblk_pct'] = df.apply(lambda row: row.WBlk / row.LFGA2 * 100, axis=1)
df['Wstl_pct'] = df.apply(lambda row: row.WStl / row.Lposs * 100, axis=1)

# Other loser stats:
df['Lts_pct'] = df.apply(lambda row: row.LScore / (2 * (row.LFGA + 0.475 * row.LFTA)) * 100, axis=1)
df['Lefg_pct'] = df.apply(lambda row: (row.LFGM2 + 1.5 * row.LFGM3) / row.LFGA, axis=1)
df['Lorb_pct'] = df.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)
df['Ldrb_pct'] = df.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)
df['Lreb_pct'] = df.apply(lambda row: (row.Lorb_pct + row.Ldrb_pct) / 2, axis=1)
df['Lto_poss'] = df.apply(lambda row: row.LTO / row.Lposs, axis=1)
df['Lft_rate'] = df.apply(lambda row: row.LFTM / row.LFGA, axis=1)
df['Last_rtio'] = df.apply(lambda row: row.LAst / (row.LFGA + 0.475*row.LFTA + row.LTO + row.LAst) * 100, axis=1)
df['Lblk_pct'] = df.apply(lambda row: row.LBlk / row.WFGA2 * 100, axis=1)
df['Lstl_pct'] = df.apply(lambda row: row.LStl / row.Wposs * 100, axis=1)

df_regular_season = df
print(df_regular_season.head())


# Add Tournament Round Data to the Tourney dataframe 
df_tourney['tourn_round'] = df_tourney.DayNum.apply(tourney_round)

df_tourney = df_tourney.merge(df_seeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']) \
.rename(columns={'seed': 'Wseed'}).drop(['TeamID'], axis=1) \
.merge(df_seeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']) \
.rename(columns={'seed': 'Lseed'}).drop(['TeamID'], axis=1) \
.merge(df_win_team, on='WTeamID').rename(columns={'TeamName': 'WTeamName'}) \
.merge(df_win_confs, on=['Season', 'WTeamID']).rename(columns={'Description': 'WConfName'}) \
.merge(df_lose_team, on='LTeamID').rename(columns={'TeamName': 'LTeamName'}) \
.merge(df_lose_confs, on=['Season', 'LTeamID']).rename(columns={'Description': 'LConfName'})

df_tourney['point_diff'] = df_tourney.WScore - df_tourney.LScore

df_upsets = df_tourney[df_tourney.Wseed > df_tourney.Lseed]

upset_count = df_upsets.groupby(['Season'], as_index=False).Wseed.count().rename(columns={'Wseed': 'upset_count'})

print(upset_count.head())