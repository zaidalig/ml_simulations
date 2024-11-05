# ML Simulations

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Usage](#usage)
  - [Running Simulations](#running-simulations)
  - [Launching the Dashboard](#launching-the-dashboard)
- [Project Structure](#project-structure)
  - [Description of Key Files](#description-of-key-files)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [Additional Tips](#additional-tips)

## Overview
**ML Simulations** is a collection of machine learning simulation scripts designed to demonstrate various regression models in real-time. Each simulation evaluates model performance, detects anomalies, and visualizes results through dynamic plots. Additionally, a Streamlit dashboard provides an interactive interface to monitor and analyze simulation metrics.

## Features
- **Diverse Regression Models**: Includes Linear Regression, Decision Tree, Random Forest, Support Vector Regressor (SVR), K-Nearest Neighbors (KNN), Neural Networks, Bayesian Ridge, Ridge, Lasso, and Gradient Boosting.
- **Real-Time Visualization**: Dynamic matplotlib plots display model predictions, residuals, Mean Squared Error (MSE) over iterations, and rolling window data points.
- **Anomaly Detection**: Identifies and logs anomalies based on deviations in target values.
- **Comprehensive Dashboard**: Streamlit-based dashboard for monitoring metrics, visualizing performance, and accessing simulation logs.
- **Modular Design**: Easily extendable to include more models or simulation scenarios.

## Technologies Used
- **Programming Language**: Python 3.8+
- **Libraries**:
  - NumPy - Numerical computations
  - Matplotlib - Data visualization
  - Scikit-Learn - Machine learning models
  - Streamlit - Interactive dashboards
  - Pandas - Data manipulation
- **Version Control**: Git & GitHub

## Installation
Follow these steps to set up and run the project on your local machine.

### Prerequisites
- **Python 3.8+**: Ensure Python is installed. [Download here](https://www.python.org/downloads/).
- **Git**: Install Git from [git-scm.com](https://git-scm.com/).
- **GitHub Account**: For pushing to repositories.

### Setting Up the Virtual Environment
It is recommended to use a virtual environment to manage dependencies.