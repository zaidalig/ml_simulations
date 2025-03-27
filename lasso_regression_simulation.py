from simulation_module import BaseSimulation
from sklearn.linear_model import Lasso
import numpy as np


class LassoRegressionSimulation(BaseSimulation):

    def initialize_dataset(self):
        if self.csv_path and self.feature_columns and self.target_column:
            super().initialize_dataset()
        else:
            X1 = np.random.rand(self.initial_points, 1) * 10
            X2 = np.random.rand(self.initial_points, 1) * 10
            self.X = np.hstack((X1, X2))
            self.y = self.generate_target(self.X)

    def generate_target(self, X):
        return 4 * X[:, 0] + np.random.randn(X.shape[0]) * 2

    def initialize_models(self):
        self.models = {
            'lasso': Lasso(alpha=0.1)
        }


def run_lasso_regression_simulation():
    sim = LassoRegressionSimulation(
        simulation_id=9,
        np_seed=9,
        initial_points=17,
        new_points=23,
        window_size=15,
        anomaly_threshold_multiplier=2.0
    )
    sim.run()


if __name__ == "__main__":
    run_lasso_regression_simulation()
