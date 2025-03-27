from simulation_module import BaseSimulation
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


class GradientBoostingSimulation(BaseSimulation):

    def initialize_dataset(self):
        """
        Initializes a dataset with polynomial relationships or loads CSV.
        """
        if self.csv_path and self.feature_columns and self.target_column:
            super().initialize_dataset()
        else:
            X = np.random.rand(self.initial_points, 1) * 10
            X2 = X ** 2
            self.X = np.hstack((X, X2))
            self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Polynomial relationship with noise.
        """
        return 1.5 * X[:, 0] ** 2 + 2 * X[:, 1] + np.random.randn(X.shape[0]) * 3

    def initialize_models(self):
        self.models = {
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=self.np_seed)
        }


def run_gradient_boosting_simulation():
    sim = GradientBoostingSimulation(
        simulation_id=10,
        np_seed=10,
        initial_points=20,
        new_points=40,
        window_size=25,
        anomaly_threshold_multiplier=1.5
    )
    sim.run()


if __name__ == "__main__":
    run_gradient_boosting_simulation()
