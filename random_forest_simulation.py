from simulation_module import BaseSimulation
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class RandomForestSimulation(BaseSimulation):

    def initialize_dataset(self):
        """
        Initializes a dataset for random forest regression.
        Uses CSV data if available; otherwise, generates a sinusoidal dataset.
        """
        if self.csv_path and self.feature_columns and self.target_column:
            super().initialize_dataset()
        else:
            self.X = np.linspace(0, 10, self.initial_points).reshape(-1, 1)
            self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a sinusoidal relationship with noise.
        """
        return 5 * np.sin(X.squeeze()) + np.random.randn(X.shape[0]) * 2

    def initialize_models(self):
        """
        Initializes a Random Forest Regressor.
        """
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=self.np_seed)
        }


def run_random_forest_simulation():
    sim = RandomForestSimulation(
        simulation_id=3,
        np_seed=3,
        initial_points=15,
        new_points=20,
        window_size=10,
        anomaly_threshold_multiplier=3.0
    )
    sim.run()


if __name__ == "__main__":
    run_random_forest_simulation()
