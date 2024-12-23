from pathlib import Path
from typing import List
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from .data_fetcher import WANTED_SEASONS

# WANTED_DATA_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
ALL_DATA_COLUMNS = ["Div","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HTHG","HTAG","HTR","Referee","HS","AS","HST","AST","HF","AF","HC","AC","HY","AY","HR","AR", "B365H", "B365D", "B365A"]

class DataAggregator():
    RECENCY_PARAMETER = 0.01
    TEAM_FORM_WINDOW = 5

    def __init__(self, DATA_FOLDER_PATH=Path.cwd().joinpath("..", "data")):
        self.DATA_FOLDER_PATH = DATA_FOLDER_PATH

    def read_csv(self, file_path: Path, wanted_features: List[str] = ALL_DATA_COLUMNS) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, sep=",", usecols=wanted_features)
            return df
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def get_data(self, wanted_leagues: List[str], wanted_features: List[str] = ALL_DATA_COLUMNS, wanted_seasons: List[str] = WANTED_SEASONS) -> pd.DataFrame:
        paths = []
        for subdirectory in os.listdir(self.DATA_FOLDER_PATH):
            if subdirectory not in wanted_leagues:
                continue
            subdirectory_path = self.DATA_FOLDER_PATH.joinpath(subdirectory)
            for file in os.listdir(subdirectory_path):
                if file.split('-')[1].split('.')[0] not in wanted_seasons:
                    continue
                paths.append(subdirectory_path.joinpath(file))

        df = pd.concat([self.read_csv(file_path=path, wanted_features=wanted_features) for path in paths], axis=0, ignore_index=True)
        return df
    
    def format_date(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        new_df = df.copy()
        
        def parse_date(date_str):
            try:
                if len(date_str.split('/')[-1]) == 2: 
                    return pd.to_datetime(date_str, format='%d/%m/%y', dayfirst=True)
                else:
                    return pd.to_datetime(date_str, format='%d/%m/%Y', dayfirst=True)
            except ValueError:
                return pd.NaT 
            except AttributeError:
                return pd.NaT

        new_df[date_column] = new_df[date_column].apply(parse_date)

        new_df['Year'] = new_df[date_column].dt.year
        new_df['Month'] = new_df[date_column].dt.month

        new_df["Day"] = new_df[date_column].dt.day
        new_df["DayOfWeek"] = new_df[date_column].dt.dayofweek

        new_df = new_df.sort_values(by=date_column).reset_index(drop=True)
        
        return new_df
    
    def ordinal_encode_teams(self, df: pd.DataFrame, home_team_column: str, away_team_column: str) -> tuple[pd.DataFrame, dict]:
        new_df = df.copy()
        le = LabelEncoder()
        new_df[home_team_column] = le.fit_transform(df[home_team_column])
        new_df[away_team_column] = le.fit_transform(df[away_team_column])

        team_mapping = dict(zip(le.transform(le.classes_), le.classes_))

        return new_df, team_mapping
    
    def one_hot_encode_teams(self, df:pd.DataFrame, home_team_column: str, away_team_column: str) -> pd.DataFrame:
        new_df = df.copy()
        new_df = pd.get_dummies(new_df, columns=[home_team_column, away_team_column], prefix=[home_team_column, away_team_column])
        return new_df
    
    def encode_result(self, df: pd.DataFrame, mapping: dict, result_column: str) -> pd.DataFrame:
        new_df = df.copy()
        
        new_df[result_column] = df[result_column].map(mapping)
        return new_df

    def create_recency_variable(self, df: pd.DataFrame, date_column: str, recency_parameter: int = RECENCY_PARAMETER) -> pd.DataFrame:
        df['Recency'] = df[date_column].apply(lambda x: np.exp(-recency_parameter * (pd.Timestamp('21/10/2024') - x).days / 365))
        return df
    
    def create_form_data(self, df: pd.DataFrame, form_window: int = TEAM_FORM_WINDOW) -> pd.DataFrame:
        weights = np.arange(1, form_window + 1)
        # Map the 'FTR' values to points for the home and away teams
        df['HomePoints'] = df['FTR'].map({1: 3, 0: 1, -1: 0})
        df['AwayPoints'] = df['FTR'].map({1: 0, 0: 1, -1: 3})

        # Shift the points to calculate rolling averages for past games
        df['ShiftedHomePoints'] = df.groupby('HomeTeam')['HomePoints'].shift(1)
        df['ShiftedAwayPoints'] = df.groupby('AwayTeam')['AwayPoints'].shift(1)

        # Calculate rolling points per game
        home_team_ppg = df.groupby('HomeTeam')['ShiftedHomePoints'].rolling(window=form_window).mean().reset_index(0, drop=True)
        away_team_ppg = df.groupby('AwayTeam')['ShiftedAwayPoints'].rolling(window=form_window).mean().reset_index(0, drop=True)

        # home_team_ppg = df.groupby('HomeTeam')['ShiftedHomePoints'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)
        # away_team_ppg = df.groupby('AwayTeam')['ShiftedAwayPoints'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)

        # Add the new PPG columns to the DataFrame
        df = pd.concat([df, home_team_ppg.rename('HomeTeamPPG'), away_team_ppg.rename('AwayTeamPPG')], axis=1)

        # Drop intermediate columns
        df = df.drop(columns=['HomePoints', 'AwayPoints', 'ShiftedHomePoints', 'ShiftedAwayPoints'])

        # Fill missing values with the mean
        df["HomeTeamPPG"] = df["HomeTeamPPG"].fillna(df.groupby("HomeTeam")["HomeTeamPPG"].transform("mean"))
        df["AwayTeamPPG"] = df["AwayTeamPPG"].fillna(df.groupby("AwayTeam")["AwayTeamPPG"].transform("mean"))


        df['ShiftedFTR_Home'] = df.groupby('HomeTeam')['FTR'].shift(1)
        df['ShiftedFTR_Away'] = df.groupby('AwayTeam')['FTR'].shift(1)

        # home_team_win_form = df.groupby('HomeTeam')['ShiftedFTR_Home'].rolling(window=form_window).mean().reset_index(0, drop=True)
        # away_team_win_form = df.groupby('AwayTeam')['ShiftedFTR_Away'].rolling(window=form_window).mean().reset_index(0, drop=True)

        home_team_win_form = df.groupby('HomeTeam')['ShiftedFTR_Home'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)
        away_team_win_form = df.groupby('AwayTeam')['ShiftedFTR_Away'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)

        df = pd.concat([df, home_team_win_form.rename('HomeTeamWinForm'), away_team_win_form.rename('AwayTeamWinForm')], axis=1)

        df["HomeTeamWinForm"] = df["HomeTeamWinForm"].fillna(df.groupby("HomeTeam")["HomeTeamWinForm"].transform("mean"))
        df["AwayTeamWinForm"] = df["AwayTeamWinForm"].fillna(df.groupby("AwayTeam")["AwayTeamWinForm"].transform("mean"))

        df = df.drop(columns=['ShiftedFTR_Home', 'ShiftedFTR_Away'])

        df['ShiftedFTHG_Home'] = df.groupby('HomeTeam')['FTHG'].shift(1)
        df['ShiftedFTAG_Away'] = df.groupby('AwayTeam')['FTAG'].shift(1)

        # home_team_goal_form = df.groupby('HomeTeam')['ShiftedFTHG_Home'].rolling(window=form_window).mean().reset_index(0, drop=True)
        # away_team_goal_form = df.groupby('AwayTeam')['ShiftedFTAG_Away'].rolling(window=form_window).mean().reset_index(0, drop=True)

        home_team_goal_form = df.groupby('HomeTeam')['ShiftedFTHG_Home'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)
        away_team_goal_form = df.groupby('AwayTeam')['ShiftedFTAG_Away'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)

        df = pd.concat([df, home_team_goal_form.rename('HomeTeamGoalForm'), away_team_goal_form.rename('AwayTeamGoalForm')], axis=1)

        df = df.drop(columns=['ShiftedFTHG_Home', 'ShiftedFTAG_Away'])

        df["HomeTeamGoalForm"] = df["HomeTeamGoalForm"].fillna(df.groupby("HomeTeam")["HomeTeamGoalForm"].transform("mean"))
        df["AwayTeamGoalForm"] = df["AwayTeamGoalForm"].fillna(df.groupby("AwayTeam")["AwayTeamGoalForm"].transform("mean"))

        df['ShiftedFTAG_Home'] = df.groupby('HomeTeam')['FTAG'].shift(1)
        df['ShiftedFTHG_Away'] = df.groupby('AwayTeam')['FTHG'].shift(1)

        # home_team_goal_against_form = df.groupby('HomeTeam')['ShiftedFTAG_Home'].rolling(window=form_window).mean().reset_index(0, drop=True)
        # away_team_goal_against_form = df.groupby('AwayTeam')['ShiftedFTHG_Away'].rolling(window=form_window).mean().reset_index(0, drop=True)

        home_team_goal_against_form = df.groupby('HomeTeam')['ShiftedFTAG_Home'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)
        away_team_goal_against_form = df.groupby('AwayTeam')['ShiftedFTHG_Away'].rolling(window=form_window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).reset_index(0, drop=True)

        df = pd.concat([df, home_team_goal_against_form.rename('HomeTeamGoalAgainstForm'), away_team_goal_against_form.rename('AwayTeamGoalAgainstForm')], axis=1)

        df = df.drop(columns=['ShiftedFTAG_Home', 'ShiftedFTHG_Away'])

        df["HomeTeamGoalAgainstForm"] = df["HomeTeamGoalAgainstForm"].fillna(df.groupby("HomeTeam")["HomeTeamGoalAgainstForm"].transform("mean"))
        df["AwayTeamGoalAgainstForm"] = df["AwayTeamGoalAgainstForm"].fillna(df.groupby("AwayTeam")["AwayTeamGoalAgainstForm"].transform("mean"))

        return df
    
    def create_gd_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        new_df["GD"] = new_df["FTHG"] - new_df["FTAG"]
        return new_df
    
    def preprocess_data(self, df: pd.DataFrame, date_column: str, home_team_column: str, away_team_column: str, result_column: str, form_window: int) -> pd.DataFrame:
        df = self.format_date(df, date_column)
        df = self.encode_result(df,
                                mapping={"H": 1, "D": 0, "A": -1}, 
                                result_column=result_column)
        df = self.create_form_data(df, form_window=form_window)
        # df = self.one_hot_encode_teams(df, home_team_column=home_team_column, away_team_column=away_team_column)
        df, team_mapping = self.ordinal_encode_teams(df, home_team_column=home_team_column, away_team_column=away_team_column)

        df.drop(columns=["Div", "FTHG", "FTAG", "HTHG", "HTAG", "HTR", "Referee", "HS", "AS", "HST", "AST", "HF", "AF", "HC","AC","HY","AY","HR","AR"], inplace=True)

        return df
    
    def calculate_accuracy(self, df: pd.DataFrame, result_column: str, prediction_column: str) -> tuple[float, float]:
        correct = 0
        won = 0

        bet_sum = 10

        for _, row in df.iterrows():
            if row[result_column] == row[prediction_column]:
                correct += 1
                if row["FTR"] == "H":
                    won += bet_sum * row["B365H"]
                elif row["FTR"] == "D":
                    won += bet_sum * row["B365D"]
                else:
                    won += bet_sum * row["B365A"]
            won -= bet_sum

        return correct / len(df), won
    
    def save_metrics(self, model_name: str, accuracy: float, profit: float) -> None:
        # append the metrics to the existing csv file
        metrics = pd.DataFrame([[model_name, accuracy, profit]], columns=["Model", "Accuracy", "Profit"])
        metrics.to_csv(Path.cwd().joinpath("metrics.csv"), mode="a", header=None, index=False)

    def read_metrics(self) -> pd.DataFrame:
        return pd.read_csv(Path.cwd().joinpath("metrics.csv"), index_col="Model")

if __name__ == "__main__":
    data_aggregator = DataAggregator(DATA_FOLDER_PATH=Path.cwd().joinpath("data"))
    # print(data_aggregator.read_csv(file_path=Path.cwd().joinpath("data", "E0", "E0-2324.csv")).head())

    print(data_aggregator.get_data(wanted_leagues=["E0"],wanted_features=["HomeTeam"]).head())