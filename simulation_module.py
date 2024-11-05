# simulation_module.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import logging
import json
import os

class BaseSimulation:
    def __init__(
        self,
        simulation_id,
        np_seed=42,
        initial_points=10,
        new_points=30,
        window_size=15,
        anomaly_threshold_multiplier=2.0,
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
        """
        self.simulation_id = simulation_id
        self.np_seed = np_seed
        self.initial_points = initial_points
        self.new_points = new_points
        self.window_size = window_size
        self.anomaly_threshold_multiplier = anomaly_threshold_multiplier

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
        Initializes the dataset. To be implemented by child classes.
        """
        raise NotImplementedError("Please implement the initialize_dataset method.")

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
        # Train models on the initial dataset before entering the loop
        self.train_models(self.X, self.y)
        self.logger.info("Models trained on initial dataset.")

        # Setup real-time plotting
        plt.ion()
        # Determine the number of subplots
        num_plots = len(self.models) + 3 + (1 if 'tree' in self.models else 0)
        fig, axs = plt.subplots(
            num_plots, 1, figsize=(14, 5 * num_plots)
        )
        plot_idx = 0
        ax_main = axs[plot_idx]
        plot_idx += 1
        ax_residual = axs[plot_idx]
        plot_idx += 1
        ax_mse = axs[plot_idx]
        plot_idx += 1
        ax_window = axs[plot_idx]
        plot_idx += 1
        ax_feature_importance = axs[plot_idx] if 'tree' in self.models else None

        for i in range(self.new_points):
            iteration = i + 1

            # Generate new data point
            new_X = self.generate_new_data_point()
            new_y = self.generate_target(new_X)

            # Ensure new_y is a scalar
            if isinstance(new_y, np.ndarray):
                if new_y.size == 1:
                    new_y = float(new_y)
                else:
                    raise ValueError(f"new_y has multiple values: {new_y}")

            # Predict before adding for anomaly detection
            predictions_before = self.predict_models(new_X)

            # Anomaly detection
            is_anomaly = False
            anomaly_message = ""
            for model_name, pred in predictions_before.items():
                if pred.size == 1 and self.detect_anomaly(new_y, pred[0]):
                    is_anomaly = True
                    anomaly_color = 'orange'
                    # Handle formatting based on feature dimensionality
                    if self.X.shape[1] == 1:
                        x_formatted = f"{new_X.squeeze():.2f}"
                    else:
                        # For multiple features, format each feature value
                        x_values = new_X.squeeze()
                        if isinstance(x_values, np.ndarray):
                            x_formatted = ', '.join([f"{x:.2f}" for x in x_values])
                        else:
                            x_formatted = f"{x_values:.2f}"
                    anomaly_message = f"Anomaly detected at Iteration {iteration}: X=[{x_formatted}], y={new_y:.2f}"
                    self.logger.warning(anomaly_message)
                    print(f"Simulation {self.simulation_id} | Iteration {iteration}: {anomaly_message}")
                    break
            if not is_anomaly:
                anomaly_color = 'green'

            # Add new data point
            self.X = np.vstack([self.X, new_X])
            self.y = np.append(self.y, new_y)

            # Rolling window
            if len(self.X) > self.window_size:
                X_train = self.X[-self.window_size:]
                y_train = self.y[-self.window_size:]
            else:
                X_train = self.X
                y_train = self.y

            # Train models
            self.train_models(X_train, y_train)

            # Predict on all data
            predictions = self.predict_models(self.X)

            # Evaluate models
            self.evaluate_models(self.y, predictions)

            # Log metrics
            self.metrics['iteration'].append(iteration)
            self.metrics['anomalies'].append(is_anomaly)
            self.log_metrics(iteration)

            # Plotting
            self.update_plots(axs, plot_idx, ax_main, ax_residual, ax_mse, ax_window, ax_feature_importance, new_X, predictions, anomaly_color, iteration)

            plt.pause(0.5)  # Adjust as needed

        plt.ioff()
        plt.show()

        # Save metrics to JSON
        self.save_metrics()

        self.logger.info("Simulation completed and metrics saved.")

    def generate_new_data_point(self):
        """
        Generates a new data point. Can be overridden by child classes for different data generation.

        Returns:
        - new_X (np.ndarray): New feature data point.
        """
        return np.random.rand(1, self.X.shape[1]) * 10  # Adjusted to match feature dimensions

    def update_plots(self, axs, plot_idx, ax_main, ax_residual, ax_mse, ax_window, ax_feature_importance, new_X, predictions, anomaly_color, iteration):
        """
        Updates all plots with the latest data and predictions.

        Parameters:
        - axs (array): Array of Axes objects.
        - plot_idx (int): Current plot index.
        - ax_main (Axes): Main plot for data and predictions.
        - ax_residual (Axes): Residual plot.
        - ax_mse (Axes): MSE over iterations plot.
        - ax_window (Axes): Rolling window data points plot.
        - ax_feature_importance (Axes): Feature importance plot (if applicable).
        - new_X (np.ndarray): New feature data point.
        - predictions (dict): Predictions from all models.
        - anomaly_color (str): Color indicating if the new point is an anomaly.
        - iteration (int): Current iteration number.
        """
        # Determine if the dataset has one or multiple features
        if self.X.shape[1] == 1:
            # Single-feature dataset
            ax_main.clear()
            ax_main.scatter(self.X, self.y, color='blue', label='Training Data')
            for model_name, y_pred in predictions.items():
                if model_name == 'linear':
                    ax_main.plot(self.X, y_pred, color='red', label='Linear Regression Line')
                elif model_name == 'tree':
                    ax_main.plot(self.X, y_pred, color='green', label='Decision Tree Prediction')
                elif model_name == 'random_forest':
                    ax_main.plot(self.X, y_pred, color='purple', label='Random Forest Prediction')
                elif model_name == 'svr':
                    ax_main.plot(self.X, y_pred, color='brown', label='SVR Prediction')
                elif model_name == 'knn':
                    ax_main.plot(self.X, y_pred, color='cyan', label='KNN Prediction')
                elif model_name == 'neural_network':
                    ax_main.plot(self.X, y_pred, color='magenta', label='Neural Network Prediction')
                elif model_name == 'bayesian':
                    ax_main.plot(self.X, y_pred, color='orange', label='Bayesian Regression Prediction')
                elif model_name == 'ridge':
                    ax_main.plot(self.X, y_pred, color='grey', label='Ridge Regression Prediction')
                elif model_name == 'lasso':
                    ax_main.plot(self.X, y_pred, color='lime', label='Lasso Regression Prediction')
                elif model_name == 'gradient_boosting':
                    ax_main.plot(self.X, y_pred, color='navy', label='Gradient Boosting Prediction')

            # Highlight the new prediction
            for model_name, pred in predictions.items():
                if pred[-1] is not None:
                    ax_main.scatter(new_X, pred[-1], color=anomaly_color, marker='x', s=100, label='New Data Prediction')

            ax_main.set_xlabel('X')
            ax_main.set_ylabel('y')
            ax_main.set_title(f'Simulation {self.simulation_id}: Real-Time Model Comparison')
            ax_main.legend(loc='upper left')

        elif self.X.shape[1] >= 2:
            # Multi-feature dataset (e.g., 2 features)
            ax_main.clear()
            ax_main.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis', label='Training Data')
            for model_name, y_pred in predictions.items():
                # Plotting predictions against the first two features for visualization
                ax_main.scatter(self.X[:, 0], self.X[:, 1], c=y_pred, cmap='coolwarm', alpha=0.5, label=f'{model_name.capitalize()} Prediction')
            # Highlight the new prediction
            for model_name, pred in predictions.items():
                if pred[-1] is not None:
                    ax_main.scatter(new_X[0, 0], new_X[0, 1], c=anomaly_color, marker='x', s=100, label='New Data Prediction')
            ax_main.set_xlabel('Feature 1 (X1)')
            ax_main.set_ylabel('Feature 2 (X2)')
            ax_main.set_title(f'Simulation {self.simulation_id}: Real-Time Model Comparison')
            ax_main.legend(loc='upper left')

        # Residual Plot for the first model (e.g., linear)
        first_model = list(self.models.keys())[0]
        residuals = self.y - predictions[first_model]
        ax_residual.clear()
        ax_residual.stem(range(len(residuals)), residuals, linefmt='r-', markerfmt='ro', basefmt='b-')
        ax_residual.set_xlabel('Data Point Index')
        ax_residual.set_ylabel('Residuals')
        ax_residual.set_title(f'Residual Plot for {first_model.capitalize()}')
        ax_residual.axhline(0, color='black', linewidth=0.5)

        # MSE Over Time for all models
        ax_mse.clear()
        for model_name in self.models:
            ax_mse.plot(
                self.metrics['iteration'],
                self.metrics['mse'][model_name],
                marker='o',
                label=f'{model_name.capitalize()} MSE'
            )
        ax_mse.set_xlabel('Iteration')
        ax_mse.set_ylabel('Mean Squared Error')
        ax_mse.set_title('MSE Over Time')
        ax_mse.legend()

        # Rolling Window Data Points
        if len(self.X) > self.window_size:
            X_train = self.X[-self.window_size:]
            y_train = self.y[-self.window_size:]
        else:
            X_train = self.X
            y_train = self.y

        ax_window.clear()
        if self.X.shape[1] == 1:
            ax_window.scatter(X_train, y_train, color='blue', label='Rolling Window Data')
            ax_window.set_xlabel('X')
            ax_window.set_ylabel('y')
        elif self.X.shape[1] >= 2:
            ax_window.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Rolling Window Data')
            ax_window.set_xlabel('Feature 1 (X1)')
            ax_window.set_ylabel('Feature 2 (X2)')
        ax_window.set_title('Data Points Used in Rolling Window')
        ax_window.legend()

        # Feature Importance for Tree-Based Models
        if ax_feature_importance and 'tree' in self.models:
            ax_feature_importance.clear()
            feature_importances = self.models['tree'].feature_importances_
            ax_feature_importance.bar(range(len(feature_importances)), feature_importances, color='green')
            ax_feature_importance.set_title('Feature Importance for Decision Tree')
            ax_feature_importance.set_ylabel('Importance')
            ax_feature_importance.set_xlabel('Features')
            ax_feature_importance.set_xticks(range(len(feature_importances)))
            ax_feature_importance.set_xticklabels([f'Feature {i+1}' for i in range(len(feature_importances))])

        plt.tight_layout()
        plt.draw()

    def save_metrics(self):
        """
        Saves the metrics dictionary to a JSON file.
        """
        metrics_filename = f'simulation_{self.simulation_id}_metrics.json'
        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info("Metrics saved to JSON file.")
