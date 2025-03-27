from simulation_module import BaseSimulation
from sklearn.linear_model import ElasticNet
import numpy as np


class ElasticNetRegressionSimulation(BaseSimulation):

    def initialize_dataset(self):
        if self.csv_path and self.feature_columns and self.target_column:
            super().initialize_dataset()
        else:
            self.X = np.random.rand(self.initial_points, 1) * 10
            self.y = self.generate_target(self.X)

    def generate_target(self, X):
        return 3 * X.squeeze() + 2 + np.random.randn(X.shape[0]) * 0.5

    def initialize_models(self):
        self.models = {
            "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5)
        }


def run_elastic_net_simulation():
    sim = ElasticNetRegressionSimulation(
        simulation_id=12,
        np_seed=1,
        initial_points=10,
        new_points=30,
        window_size=15,
        anomaly_threshold_multiplier=2.0
    )
    sim.run()


if __name__ == "__main__":
    run_elastic_net_simulation()
