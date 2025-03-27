import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import logging
import json
import os
import pandas as pd  # Added for CSV support


class BaseSimulation:

    def __init__(
        self,
        simulation_id,
        np_seed=42,
        initial_points=10,
        new_points=30,
        window_size=15,
        anomaly_threshold_multiplier=2.0,
        csv_path=None,
        feature_columns=None,
        target_column=None,
    ):
        """
        Initializes the base simulation with common parameters.

        Parameters:
        - simulation_id (int): Unique identifier for the simulation.
        - np_seed (int): Seed for NumPy's random number generator.
        - initial_points (int): Number of initial data points.
        - new_points (int): Number of new data points to add during the simulation.
        - window_size (int): Rolling window size for training.
        - anomaly_threshold_multiplier (float): Multiplier to determine anomaly threshold.
        - csv_path (str, optional): Path to custom CSV file (if using uploaded data).
        - feature_columns (list, optional): List of column names to use as features.
        - target_column (str, optional): Column name to use as target.
        """
        self.simulation_id = simulation_id
        self.np_seed = np_seed
        self.initial_points = initial_points
        self.new_points = new_points
        self.window_size = window_size
        self.anomaly_threshold_multiplier = anomaly_threshold_multiplier

        self.csv_path = csv_path
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Initialize random seed
        np.random.seed(self.np_seed)

        # Initialize dataset
        self.X = None
        self.y = None
        self.initialize_dataset()

        # Initialize models
        self.models = {}
        self.initialize_models()

        # Metrics storage
        self.metrics = {
            'iteration': [],
            'mse': {model: [] for model in self.models},
            'r2': {model: [] for model in self.models},
            'anomalies': []
        }

        # Setup logging
        self.setup_logging()

    def initialize_dataset(self):
        """
        Initializes the dataset from CSV if provided, otherwise requires override by child classes.
        """
        if self.csv_path and self.feature_columns and self.target_column:
            df = pd.read_csv(self.csv_path)
            df = df.dropna()
            self.X = df[self.feature_columns].values
            self.y = df[self.target_column].values
        else:
            raise NotImplementedError("Please override initialize_dataset or provide CSV path with feature/target columns.")

    def generate_target(self, X):
        """
        Generates target variable with some noise. To be implemented by child classes.

        Parameters:
        - X (np.ndarray): Feature data.

        Returns:
        - y (np.ndarray): Target data.
        """
        raise NotImplementedError("Please implement the generate_target method.")

    def initialize_models(self):
        """
        Initializes models. To be implemented by child classes.
        """
        raise NotImplementedError("Please implement the initialize_models method.")

    def setup_logging(self):
        """
        Sets up logging for the simulation.
        """
        log_filename = f'simulation_{self.simulation_id}.log'
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
        self.logger = logging.getLogger()
        self.logger.info(f"Starting Simulation {self.simulation_id}")

    def detect_anomaly(self, new_y, predicted_y):
        """
        Detects if the new data point is an anomaly.

        Parameters:
        - new_y (float): Actual target value of the new data point.
        - predicted_y (float): Predicted target value of the new data point.

        Returns:
        - is_anomaly (bool): True if anomaly detected, else False.
        """
        threshold = self.anomaly_threshold_multiplier * np.std(self.y)
        return abs(new_y - predicted_y) > threshold

    def train_models(self, X_train, y_train):
        """
        Trains all initialized models on the provided training data.

        Parameters:
        - X_train (np.ndarray): Training feature data.
        - y_train (np.ndarray): Training target data.
        """
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict_models(self, X):
        """
        Generates predictions from all models on the provided data.

        Parameters:
        - X (np.ndarray): Feature data for prediction.

        Returns:
        - predictions (dict): Dictionary of predictions per model.
        """
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        return predictions

    def evaluate_models(self, y_true, predictions):
        """
        Evaluates all models based on Mean Squared Error and R-squared metrics.

        Parameters:
        - y_true (np.ndarray): Actual target values.
        - predictions (dict): Dictionary of predictions per model.
        """
        for model_name, y_pred in predictions.items():
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            self.metrics['mse'][model_name].append(mse)
            self.metrics['r2'][model_name].append(r2)

    def log_metrics(self, iteration):
        """
        Logs the evaluation metrics for the current iteration.

        Parameters:
        - iteration (int): Current iteration number.
        """
        self.logger.info(f"Iteration {iteration}")
        for model_name in self.models:
            mse = self.metrics['mse'][model_name][-1]
            r2 = self.metrics['r2'][model_name][-1]
            self.logger.info(f"{model_name.capitalize()} - MSE: {mse:.2f}, R2: {r2:.2f}")

    def run(self):
        """
        Runs the simulation loop.
        """
        self.train_models(self.X, self.y)
        self.logger.info("Models trained on initial dataset.")

        # Skip real-time loop if using full CSV
        if self.new_points == 0:
            predictions = self.predict_models(self.X)
            self.evaluate_models(self.y, predictions)
            self.metrics['iteration'].append(1)
            self.metrics['anomalies'].append(False)
            self.log_metrics(1)
            self.save_metrics()
            self.export_predictions_csv(predictions) 
            self.logger.info("Simulation completed and metrics saved (CSV mode).")
            return

        # Otherwise, run iterative logic (synthetic mode)
        # You can include your full plotting and update logic here as needed
        self.logger.warning("new_points > 0 but CSV dataset used — skipping loop. Implement if needed.")
        # Save predictions as well
        self.export_predictions_csv(predictions)

    def save_metrics(self):
        """
        Saves the metrics dictionary to a JSON file.
        """
        metrics_filename = f'simulation_{self.simulation_id}_metrics.json'
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info("Metrics saved to JSON file.")

    def export_predictions_csv(self, predictions):
        """
        Exports predictions, residuals, and anomaly flags to CSV.
        """
        data = self.X.tolist()
        if isinstance(self.y, np.ndarray):
            y = self.y.tolist()
        else:
            y = list(self.y)

        output = []
        for i in range(len(self.X)):
            row = {
                **{f"feature_{j+1}": float(self.X[i][j]) for j in range(self.X.shape[1])},
                "true_y": float(y[i]),
                "predicted_y": float(predictions[list(self.models.keys())[0]][i]),
                "residual": float(y[i] - predictions[list(self.models.keys())[0]][i]),
                "anomaly": bool(self.metrics['anomalies'][i]) if i < len(self.metrics['anomalies']) else False
            }
            output.append(row)

        df = pd.DataFrame(output)
        export_path = f"simulation_{self.simulation_id}_predictions.csv"
        df.to_csv(export_path, index=False)
        self.logger.info(f"Predictions exported to {export_path}")
