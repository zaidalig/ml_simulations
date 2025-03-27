from simulation_module import BaseSimulation
from sklearn.linear_model import LinearRegression
import numpy as np


class LinearRegressionSimulation(BaseSimulation):

    def initialize_dataset(self):
        """
        Initializes a linear dataset either from CSV or generates synthetic data.
        """
        if self.csv_path and self.feature_columns and self.target_column:
            super().initialize_dataset()
        else:
            self.X = np.random.rand(self.initial_points, 1) * 10
            self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a linear relationship with noise.
        """
        return 2 * X.squeeze() + np.random.randn(X.shape[0]) * 1.5

    def initialize_models(self):
        """
        Initializes a Linear Regression model.
        """
        self.models = {
            'linear': LinearRegression()
        }


def run_linear_regression_simulation():
    sim = LinearRegressionSimulation(
        simulation_id=1,
        np_seed=1,
        initial_points=10,
        new_points=30,
        window_size=15,
        anomaly_threshold_multiplier=2.0
    )
    sim.run()


if __name__ == "__main__":
    run_linear_regression_simulation()
