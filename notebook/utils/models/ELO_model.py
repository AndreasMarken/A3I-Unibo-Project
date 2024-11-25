from typing import Callable, List, Optional
from scipy.stats import logistic
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
from tqdm import tqdm

class ELOModel:
    def __init__(self, initial_rating: int = 1500, alpha: int = 20, normalization_constand_cdf: int = 400, sharpening_factor: float = 1.0, home_advantage: int = 50, momentum_boost: float = 10, draw_factor: float = 0.2, seed: Optional[int] = 42):
        self.ratings = {}
        self.initial_rating = initial_rating
        self.alpha = alpha
        self.history = {}
        self.normalization_constand_cdf = normalization_constand_cdf
        self.sharpening_factor = sharpening_factor
        self.home_advantage = home_advantage
        self.momentum_boost = momentum_boost
        self.streaks = {}
        self.draw_factor = draw_factor

        np.random.seed(seed)

    def get_rating(self, team: str) -> int:
        return self.ratings.get(team, self.initial_rating)
    
    def update_streak(self, team: str, result: float) -> None:
        if team not in self.streaks:
            self.streaks[team] = 0
    
        if result == 1:
            self.streaks[team] = max(0, self.streaks[team] + 1)
        elif result == 0:
            self.streaks[team] = min(0, self.streaks[team] - 1)
        else:
            self.streaks[team] = 0
    
    def calculate_expected_score(self, home_team: str, away_team: str) -> tuple[float, float]:
        momentum_factor_home_team = self.momentum_boost * self.streaks.get(home_team, 0)
        momentum_factor_away_team = self.momentum_boost * self.streaks.get(away_team, 0)

        rating_home_team = self.get_rating(home_team) + self.home_advantage + momentum_factor_home_team
        rating_away_team = self.get_rating(away_team) + momentum_factor_away_team

        expected_home_team = logistic.cdf((rating_home_team - rating_away_team) / (self.normalization_constand_cdf * self.sharpening_factor))
        expected_away_team = 1 - expected_home_team

        expected_draw = self.draw_factor * (1 - abs(expected_home_team - expected_away_team))

        expected_home_team -= expected_draw / 2
        expected_away_team -= expected_draw / 2

        return expected_home_team, expected_draw, expected_away_team

    def update_ratings(self, home_team: str, away_team: str, result: float, recency_parameter: float = 1.0) -> None:
        expected_home_team, expected_draw, expected_away_team = self.calculate_expected_score(home_team, away_team)
        rating_home_team = self.get_rating(home_team)
        rating_away_team = self.get_rating(away_team)

        # TODO: Create adaptive alpha

        new_rating_home_team = rating_home_team + self.alpha * (result - expected_home_team) * recency_parameter
        new_rating_away_team = rating_away_team + self.alpha * ((1 - result) - expected_away_team) * recency_parameter
        
        self.ratings[home_team] = new_rating_home_team
        self.ratings[away_team] = new_rating_away_team

        self.record_history(home_team, new_rating_home_team)
        self.record_history(away_team, new_rating_away_team)

    def record_history(self, team: str, rating: float) -> None:
        if team not in self.history:
            self.history[team] = []
        self.history[team].append(rating)

    def process_match(self, home_team: str, away_team: str, full_time_result: str, recency_parameter: float = 1.0) -> None:
        if full_time_result == 'H':
            result = 1  
        elif full_time_result == 'A':
            result = 0  
        else:  
            result = 0.5
        
        self.update_ratings(home_team, away_team, result, recency_parameter)

    def get_ratings(self) -> dict:
        return self.ratings

    def predict(self, match_data: pd.DataFrame) -> pd.DataFrame:
        # Ensure all teams in match_data are initialized in self.ratings
        for team in pd.concat([match_data['HomeTeam'], match_data['AwayTeam']]).unique():
            if team not in self.ratings:
                self.ratings[team] = self.initial_rating

        # Generate predictions for each match
        match_data = match_data.copy()  # create a copy
        match_data['Prediction'] = match_data.apply(
            lambda row: self._get_match_outcome(row['HomeTeam'], row['AwayTeam']),
            axis=1
        )

        
        return match_data[['HomeTeam', 'AwayTeam', 'Prediction', 'FTR', 'B365H', 'B365D', 'B365A']]


    def _get_match_outcome(self, home_team: str, away_team: str) -> str:
        home_expected_score, expected_draw, away_expected_score = self.calculate_expected_score(home_team, away_team)
        
        probabilities = [home_expected_score, expected_draw, away_expected_score]
        outcomes = ['H', 'D', 'A']
        return np.random.choice(outcomes, p=probabilities)

    def plot_ratings(self, top_k: int = 5) -> None:
        final_ratings = self.get_ratings()
        sorted_teams = sorted(final_ratings.items(), key=lambda item: item[1], reverse=True)

        top_teams = sorted_teams[:top_k]

        plt.figure(figsize=(12, 12))
        for team, _ in top_teams:
            try:
                plt.plot(self.history[team], label=team)
            except:
                pass

        plt.title('Elo Rating Evolution - Top Teams')
        plt.xlabel('Match Number')
        plt.ylabel('Elo Rating')
        plt.legend()
        plt.grid()
        plt.show()

    def generate_league_table(self, match_data: pd.DataFrame) -> pd.DataFrame:
        # Get predicted results for each match
        predictions = self.predict(match_data)

        # Initialize points dictionary
        points = {}
        for team in pd.concat([predictions['HomeTeam'], predictions['AwayTeam']]).unique():
            points[team] = 0

        # Assign points based on predictions
        for _, row in predictions.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            result = row['Prediction']

            if result == 'H':  # Home team win
                points[home_team] += 3
            elif result == 'A':  # Away team win
                points[away_team] += 3
            else:  # Draw
                points[home_team] += 1
                points[away_team] += 1

        # Create the league table DataFrame
        league_table = pd.DataFrame(list(points.items()), columns=['Team', 'Points'])
        league_table = league_table.sort_values(by='Points', ascending=False).reset_index(drop=True)

        return league_table

    @staticmethod
    def plot_league_table(league_table: pd.DataFrame, file_path: str) -> None:
        top_10 = league_table.head(20)
        
        _, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(top_10['Team'], top_10['Points'], color='skyblue')
        
        for i, (_, points) in enumerate(zip(top_10['Team'], top_10['Points'])):
            ax.text(points + 0.5, i, f'{points}', va='center', fontsize=12)
        
        ax.invert_yaxis()

        ax.set_title(f'Top 10 Teams in the League season', fontsize=16)
        ax.set_xlabel('Points', fontsize=14)
        ax.set_ylabel('Teams', fontsize=14)
        plt.show()

        # plt.savefig(os.path.join(os.path.dirname(__file__), f'../../data/tables/{file_path}'), bbox_inches='tight')
        # plt.close()
    
    def calculate_accuracy(self, match_data: pd.DataFrame) -> float:
        predictions = self.predict(match_data)
        correct_predictions = predictions[predictions['FTR'] == predictions['Prediction']]
        return len(correct_predictions) / len(predictions)

    def fit(self, match_data: pd.DataFrame, recency_parameter: float = 1.0) -> None:
        for _, row in match_data.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            result = row['FTR']
            self.process_match(home_team, away_team, result, recency_parameter)

    def cross_validate(self, match_data: pd.DataFrame, n_folds: int = 5) -> float:
        accuracy = 0
        n = len(match_data)
        fold_size = n // n_folds

        for i in range(n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size

            test_data = match_data.iloc[start:end]
            train_data = pd.concat([match_data.iloc[:start], match_data.iloc[end:]])
            self.fit(train_data)
            accuracy += self.calculate_accuracy(test_data)

        return accuracy / n_folds

    @staticmethod
    def grid_search(match_data: pd.DataFrame, coarse_param_grid: dict, n_folds: int = 5) -> dict:
        best_params = None
        best_accuracy = 0

        print('Coarse grid search...')
        for params in tqdm(ParameterGrid(coarse_param_grid)):
            model = ELOModel(**params)
            accuracy = model.cross_validate(match_data, n_folds)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

        print(f'Finished with coarse grid search. Current best params: {best_params}')
        # Refine grid around best_params
        refined_grid = {
            'initial_rating': [best_params['initial_rating'] - 100, best_params['initial_rating'], best_params['initial_rating'] + 100],
            'alpha': [max(5, best_params['alpha'] - 5), best_params['alpha'], best_params['alpha'] + 5],
            'normalization_constand_cdf': [max(100, best_params['normalization_constand_cdf'] - 100), best_params['normalization_constand_cdf'], best_params['normalization_constand_cdf'] + 100],
            'sharpening_factor': [max(0.5, best_params['sharpening_factor'] - 0.25), best_params['sharpening_factor'], best_params['sharpening_factor'] + 0.25],
            'home_advantage': [max(0, best_params['home_advantage'] - 25), best_params['home_advantage'], best_params['home_advantage'] + 25],
            'momentum_boost': [max(0, best_params['momentum_boost'] - 5), best_params['momentum_boost'], best_params['momentum_boost'] + 5],
            'draw_factor': [max(0, best_params['draw_factor'] - 0.05), best_params['draw_factor'], best_params['draw_factor'] + 0.5]
        }

        print('Refined grid search...')
        for params in tqdm(ParameterGrid(refined_grid)):
            model = ELOModel(**params)
            accuracy = model.cross_validate(match_data, n_folds)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
        
        print(f'Finished with refined grid search. Best params: {best_params}')

        return best_params

    def save_model(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath: str) -> 'ELOModel':
        with open(filepath, 'rb') as f:
            return pickle.load(f)