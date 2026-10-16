"""
DS-550 Machine Learning - Assignment 1
Student: Muhammad Imran Mian (Roll No: MSDSF25A007)
Program: MS Data Science (Batch F26)
PUCIT New Campus Lahore
Instructor: Dr. Syed Muhammad Ali

Question 4: Logistic Regression Classifier
Implements:
  1. Linear Decision Boundary Classifier (Batch Gradient Descent)
  2. Non-linear Decision Boundary Classifier (Polynomial Feature Mapping)
  3. Comprehensive Comparative Performance Analysis:
     - Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
  4. Plots: Decision Boundaries on Train/Test and ROC Curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
)
from sklearn.preprocessing import PolynomialFeatures


def sigmoid(z):
    """Numerically stable sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))


def load_q4_data():
    """Load train and test datasets from Data/Q4 directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "Data", "Q4", "Question 4 Dataset-train.csv")
    test_path = os.path.join(base_dir, "Data", "Q4", "Question 4 Dataset-test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def train_logistic_regression(X, y, alpha=0.5, epochs=10000, lambda_reg=0.0):
    """
    Fits logistic regression parameters using batch gradient descent:
        cost J(theta) = -1/m * sum( y*log(h) + (1-y)*log(1-h) ) + lambda/(2m)*sum(theta[1:]^2)
        grad = 1/m * X^T * (h - y) + (lambda/m) * [0, theta[1:]]
    """
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        h = sigmoid(X @ theta)
        reg_term = (lambda_reg / m) * theta
        reg_term[0] = 0.0  # Do not regularize intercept
        gradient = (1.0 / m) * (X.T @ (h - y)) + reg_term
        theta -= alpha * gradient

    return theta


def evaluate_classifier(y_true, y_prob, threshold=0.5, model_name="Model"):
    """Evaluates classification performance across multiple metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[{model_name} Performance on Test Set]")
    print(f"  Accuracy   : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print("  Confusion Matrix (TN, FP / FN, TP):")
    print(f"    [[TN={cm[0,0]:2d}, FP={cm[0,1]:2d}]")
    print(f"     [FN={cm[1,0]:2d}, TP={cm[1,1]:2d}]]")

    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "ROC_AUC": roc_auc, "CM": cm
    }


def main():
    print("=" * 70)
    print("DS-550 Machine Learning - Assignment 1 - Question 4")
    print("Student: Muhammad Imran Mian | Roll No: MSDSF25A007")
    print("=" * 70)

    # 1. Load Data
    train_df, test_df = load_q4_data()
    X_tr = train_df[['x1', 'x2']].values
    y_tr = train_df['class'].values
    X_te = test_df[['x1', 'x2']].values
    y_te = test_df['class'].values

    print(f"\nTraining set size: {len(train_df)} (Class 0: {sum(y_tr==0)}, Class 1: {sum(y_tr==1)})")
    print(f"Test set size    : {len(test_df)} (Class 0: {sum(y_te==0)}, Class 1: {sum(y_te==1)})")

    # 2. Linear Decision Boundary (Degree 1)
    X_tr_lin = np.column_stack([np.ones(len(X_tr)), X_tr])
    X_te_lin = np.column_stack([np.ones(len(X_te)), X_te])

    theta_lin = train_logistic_regression(X_tr_lin, y_tr, alpha=0.5, epochs=8000, lambda_reg=0.0)
    prob_lin = sigmoid(X_te_lin @ theta_lin)
    metrics_lin = evaluate_classifier(y_te, prob_lin, model_name="Linear Decision Boundary")

    print("\nLinear Model Parameters (theta_0 + theta_1*x1 + theta_2*x2 = 0):")
    print(f"  theta_0 (Intercept): {theta_lin[0]:.4f}")
    print(f"  theta_1 (x1)       : {theta_lin[1]:.4f}")
    print(f"  theta_2 (x2)       : {theta_lin[2]:.4f}")

    # 3. Non-linear Decision Boundary (Quadratic, Degree 2)
    poly2 = PolynomialFeatures(degree=2, include_bias=True)
    X_tr_p2 = poly2.fit_transform(X_tr)
    X_te_p2 = poly2.transform(X_te)

    theta_p2 = train_logistic_regression(X_tr_p2, y_tr, alpha=0.5, epochs=10000, lambda_reg=0.001)
    prob_p2 = sigmoid(X_te_p2 @ theta_p2)
    metrics_p2 = evaluate_classifier(y_te, prob_p2, model_name="Non-Linear Decision Boundary (Quadratic)")

    # 4. Generate Visualizations: Decision Boundaries
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    u = np.linspace(-1.1, 0.5, 300)
    v = np.linspace(-0.9, 0.5, 300)
    U, V = np.meshgrid(u, v)
    grid_pts = np.column_stack([U.ravel(), V.ravel()])

    Z_lin = sigmoid(np.column_stack([np.ones(len(grid_pts)), grid_pts]) @ theta_lin).reshape(U.shape)
    Z_p2 = sigmoid(poly2.transform(grid_pts) @ theta_p2).reshape(U.shape)

    for ax, split, X_data, y_data in zip(axes, ['Train Set', 'Test Set'], [X_tr, X_te], [y_tr, y_te]):
        ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1], color='blue', label='Class 0', alpha=0.8, edgecolors='k')
        ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1], color='red', marker='x', s=60, label='Class 1', alpha=0.8)
        ax.contour(U, V, Z_lin, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        ax.contour(U, V, Z_p2, levels=[0.5], colors='green', linestyles='-', linewidths=2)
        ax.set_title(f'Decision Boundaries on {split}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Feature $x_1$', fontsize=11)
        ax.set_ylabel('Feature $x_2$', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)

    custom_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='--'),
        Line2D([0], [0], color='green', lw=2, linestyle='-'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='x', color='red', lw=0, markersize=8)
    ]
    axes[0].legend(custom_lines, ['Linear Decision Boundary', 'Quadratic Decision Boundary', 'Class 0', 'Class 1'], loc='lower left')
    axes[1].legend(custom_lines, ['Linear Decision Boundary', 'Quadratic Decision Boundary', 'Class 0', 'Class 1'], loc='upper left')

    plt.tight_layout()
    boundary_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q4_decision_boundaries_comparison.png")
    plt.savefig(boundary_plot_path, dpi=200)
    plt.close()

    # 5. Generate Visualizations: ROC Curves
    fpr_lin, tpr_lin, _ = roc_curve(y_te, prob_lin)
    fpr_p2, tpr_p2, _ = roc_curve(y_te, prob_p2)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_lin, tpr_lin, color='blue', lw=2, label=f'Linear Boundary (AUC = {metrics_lin["ROC_AUC"]:.3f})')
    plt.plot(fpr_p2, tpr_p2, color='green', lw=2, linestyle='--', label=f'Non-Linear Boundary (AUC = {metrics_p2["ROC_AUC"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k:', lw=1.5, label='Random Baseline (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) on Test Set', fontsize=12, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    roc_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q4_roc_curves.png")
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()

    print(f"\nSaved plots:\n  {boundary_plot_path}\n  {roc_plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
