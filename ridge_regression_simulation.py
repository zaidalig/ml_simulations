from simulation_module import BaseSimulation
from sklearn.linear_model import Ridge
import numpy as np


class RidgeRegressionSimulation(BaseSimulation):

    def initialize_dataset(self):
        if self.csv_path and self.feature_columns and self.target_column:
            super().initialize_dataset()
        else:
            self.X = np.random.rand(self.initial_points, 1) * 10
            self.y = self.generate_target(self.X)

    def generate_target(self, X):
        return 2.8 * X.squeeze() + np.random.randn(X.shape[0]) * 1.2

    def initialize_models(self):
        self.models = {
            'ridge': Ridge(alpha=1.0)
        }


def run_ridge_regression_simulation():
    sim = RidgeRegressionSimulation(
        simulation_id=8,
        np_seed=8,
        initial_points=14,
        new_points=20,
        window_size=10,
        anomaly_threshold_multiplier=2.0
    )
    sim.run()


if __name__ == "__main__":
    run_ridge_regression_simulation()
