from simulation_module import BaseSimulation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

class PolynomialRegressionSimulation(BaseSimulation):
    def initialize_dataset(self):
        """
        Initializes a polynomial dataset.
        """
        self.X = np.random.rand(self.initial_points, 1) * 10
        self.y = self.generate_target(self.X)

    def generate_target(self, X):
        """
        Generates a polynomial relationship with noise.
        """
        return 2 * (X.squeeze() ** 2) - 3 * X.squeeze() + np.random.randn(X.shape[0]) * 0.5

    def initialize_models(self):
        """
        Initializes a Polynomial Regression model.
        """
        degree = 2  # Set the polynomial degree
        self.models = {
            "polynomial": make_pipeline(PolynomialFeatures(degree), LinearRegression())
        }

def run_polynomial_regression_simulation():
    sim = PolynomialRegressionSimulation(
        simulation_id=11,
        np_seed=1,
        initial_points=10,
        new_points=30,
        window_size=15,
        anomaly_threshold_multiplier=2.0
    )
    sim.run()

if __name__ == "__main__":
    run_polynomial_regression_simulation()
