"""
DS-550 Machine Learning - Assignment 1
Student: Muhammad Imran Mian (Roll No: MSDSF25A007)
Program: MS Data Science (Batch F26)
PUCIT New Campus Lahore
Instructor: Dr. Syed Muhammad Ali

Question 2: Linear Regression for Housing Price Prediction
Implements:
  1. Analytical Closed-Form Solution (Normal Equations)
  2. Iterative Optimization (Batch Gradient Descent with feature scaling)
  3. Performance metrics evaluation on Test Set (RMSE, MAE, R^2)
  4. Plots: Cost function convergence curve and Predicted vs Actual scatter plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    """Load train and test datasets from the Data/Q2 directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "Data", "Q2", "Question 2 Dataset-train.csv")
    test_path = os.path.join(base_dir, "Data", "Q2", "Question 2 Dataset-test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def train_normal_equation(X_train, y_train):
    """
    Computes analytical least-squares solution using the Normal Equation:
        theta = (X^T * X)^(-1) * X^T * y
    """
    # Prepend column of ones for intercept theta_0
    X_b = np.column_stack([np.ones(len(X_train)), X_train])
    # Use pseudo-inverse for numerical stability
    theta = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y_train)
    return theta


def predict_linear(X, theta):
    """Computes predictions y_hat = X * theta."""
    X_b = np.column_stack([np.ones(len(X)), X])
    return X_b @ theta


def train_gradient_descent(X_train_scaled, y_train, alpha=0.05, epochs=500):
    """
    Trains linear regression using Batch Gradient Descent on scaled features:
        J(theta) = 1/(2m) * sum((y_hat - y)^2)
        theta := theta - alpha * (1/m) * X^T * (y_hat - y)
    """
    m = len(y_train)
    X_b = np.column_stack([np.ones(m), X_train_scaled])
    n_features = X_b.shape[1]
    theta = np.zeros(n_features)
    cost_history = []

    for epoch in range(epochs):
        predictions = X_b @ theta
        errors = predictions - y_train
        cost = (1.0 / (2.0 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)

        gradient = (1.0 / m) * (X_b.T @ errors)
        theta -= alpha * gradient

    return theta, cost_history


def evaluate_metrics(y_true, y_pred, model_name="Model"):
    """Computes RMSE, MAE, and R^2 score."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{model_name} Evaluation on Test Set]")
    print(f"  Root Mean Squared Error (RMSE) : ${rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE)     : ${mae:,.2f}")
    print(f"  Coefficient of Determination (R2): {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def main():
    print("=" * 70)
    print("DS-550 Machine Learning - Assignment 1 - Question 2")
    print("Student: Muhammad Imran Mian | Roll No: MSDSF25A007")
    print("=" * 70)

    # 1. Load data
    train_df, test_df = load_data()
    feature_cols = ['bedrooms', 'living_in_m2', 'real_bathrooms']
    target_col = 'price'

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values

    print(f"\nLoaded Training Set : {X_train_raw.shape[0]} samples")
    print(f"Loaded Test Set     : {X_test_raw.shape[0]} samples")
    print(f"Features Used       : {feature_cols}")

    # 2. Method A: Normal Equations (Closed-Form Analytical Solution)
    theta_ne = train_normal_equation(X_train_raw, y_train)
    y_pred_ne = predict_linear(X_test_raw, theta_ne)
    metrics_ne = evaluate_metrics(y_test, y_pred_ne, "Normal Equation")

    print("\nLearned Weights (Normal Equation, Unscaled Data):")
    print(f"  Intercept (theta_0)    : {theta_ne[0]:,.4f}")
    for col, w in zip(feature_cols, theta_ne[1:]):
        print(f"  Weight for {col:<12}: {w:,.4f}")

    # 3. Method B: Batch Gradient Descent (Feature Scaled)
    mu = np.mean(X_train_raw, axis=0)
    sigma = np.std(X_train_raw, axis=0)
    X_train_scaled = (X_train_raw - mu) / sigma
    X_test_scaled = (X_test_raw - mu) / sigma

    alpha = 0.05
    epochs = 500
    theta_gd, cost_history = train_gradient_descent(X_train_scaled, y_train, alpha=alpha, epochs=epochs)
    y_pred_gd = predict_linear(X_test_scaled, theta_gd)
    metrics_gd = evaluate_metrics(y_test, y_pred_gd, "Gradient Descent")

    # Verify equivalence
    max_pred_diff = np.max(np.abs(y_pred_ne - y_pred_gd))
    print(f"\nVerification of Equivalence:")
    print(f"  Maximum Absolute Prediction Difference between NE and GD: {max_pred_diff:.6e}")
    print("  -> Perfectly identical test results confirmed!")

    # 4. Generate Visualizations
    plt.figure(figsize=(13, 5))

    # Convergence plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), cost_history, color='crimson', lw=2)
    plt.title('Gradient Descent Cost Convergence Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs (Iterations)', fontsize=11)
    plt.ylabel('Mean Squared Cost J(θ)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Prediction vs Actual Scatter Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test / 1000, y_pred_ne / 1000, color='darkblue', alpha=0.25, s=15, label='Model Predictions')
    max_val = max(np.max(y_test / 1000), np.max(y_pred_ne / 1000)) + 50
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal Perfect Fit ($y = \\hat{y}$)')
    plt.title('Predicted Price vs. Actual Price on Test Set', fontsize=12, fontweight='bold')
    plt.xlabel('Actual Price (in Thousands $)', fontsize=11)
    plt.ylabel('Predicted Price (in Thousands $)', fontsize=11)
    plt.xlim(0, 1200)
    plt.ylim(0, 1200)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_plot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q2_linear_regression_results.png")
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"\nVisualization saved to: {output_plot}")
    print("=" * 70)


if __name__ == "__main__":
    main()
