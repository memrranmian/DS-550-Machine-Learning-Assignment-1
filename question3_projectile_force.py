"""
DS-550 Machine Learning - Assignment 1
Student: Muhammad Imran Mian (Roll No: MSDSF25A007)
Program: MS Data Science (Batch F26)
PUCIT New Campus Lahore
Instructor: Dr. Syed Muhammad Ali

Question 3: Wind Tunnel Projectile Air Resistance
Implements:
  1. 5th-degree interpolating polynomial using np.linalg.solve (Vandermonde Matrix)
  2. Quadratic polynomial regression using least squares (Normal Equation)
  3. Force estimation at projectile velocity v = 750 ft/sec (x = 7.5)
  4. Visualization and conceptual comparison between interpolation vs regression
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def solve_interpolation(x, y):
    """
    Finds the interpolating polynomial of degree 5 passing through all 6 points:
        p(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
    Solves the linear system:
        V * c = y
    where V is the Vandermonde matrix.
    """
    # Vandermonde matrix with increasing powers: [1, x, x^2, x^3, x^4, x^5]
    V = np.vander(x, N=len(x), increasing=True)
    c_interp = np.linalg.solve(V, y)
    return c_interp, V


def solve_quadratic_regression(x, y):
    """
    Fits a quadratic polynomial using least squares regression:
        p(x) = c0 + c1*x + c2*x^2
    Solves the overdetermined system using Normal Equation:
        c = (X^T * X)^(-1) * X^T * y
    """
    X = np.column_stack([np.ones_like(x), x, x ** 2])
    c_quad = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return c_quad, X


def main():
    print("=" * 70)
    print("DS-550 Machine Learning - Assignment 1 - Question 3")
    print("Student: Muhammad Imran Mian | Roll No: MSDSF25A007")
    print("=" * 70)

    # Given Experimental Data
    # Velocity in units of 100 ft/sec: x
    # Force in units of 100 lb: y
    x = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    y = np.array([0, 2.90, 14.8, 39.6, 74.3, 119], dtype=float)

    print("\nGiven Experimental Wind Tunnel Data:")
    for xi, yi in zip(x, y):
        print(f"  Velocity: {xi:4.1f} (hundred ft/sec) = {xi*100:4.0f} ft/sec  -->  Force: {yi:5.2f} (hundred lb) = {yi*100:6.1f} lb")

    # (i) 5th-Degree Interpolating Polynomial
    c_interp, V = solve_interpolation(x, y)
    print("\n" + "-" * 70)
    print("(i) 5th-Degree Interpolating Polynomial Coefficients:")
    for i, c in enumerate(c_interp):
        print(f"    c_{i} (coefficient of x^{i}): {c:14.8f}")

    poly_str_interp = (
        f"p(x) = {c_interp[0]:.4f} + {c_interp[1]:.4f}x + ({c_interp[2]:.4f})x^2 + "
        f"{c_interp[3]:.4f}x^3 + ({c_interp[4]:.4f})x^4 + {c_interp[5]:.6f}x^5"
    )
    print(f"\nInterpolation Polynomial Formulation:\n  {poly_str_interp}")

    # (ii) Quadratic Polynomial via Least Squares Regression
    c_quad, X = solve_quadratic_regression(x, y)
    print("\n" + "-" * 70)
    print("(ii) Quadratic Polynomial Regression Coefficients:")
    for i, c in enumerate(c_quad):
        print(f"    c_{i} (coefficient of x^{i}): {c:14.8f}")

    poly_str_quad = f"p(x) = {c_quad[0]:.4f} + ({c_quad[1]:.4f})x + {c_quad[2]:.4f}x^2"
    print(f"\nQuadratic Regression Formulation:\n  {poly_str_quad}")

    # Target: Estimation at projectile velocity = 750 ft/sec -> x = 7.5
    x_target = 7.5
    # Interpolation estimate: evaluate p_interp(7.5)
    pred_interp_hundred = np.polyval(c_interp[::-1], x_target)
    pred_interp_lb = pred_interp_hundred * 100.0

    # Quadratic regression estimate: evaluate p_quad(7.5)
    pred_quad_hundred = c_quad[0] + c_quad[1] * x_target + c_quad[2] * (x_target ** 2)
    pred_quad_lb = pred_quad_hundred * 100.0

    print("\n" + "=" * 70)
    print("ESTIMATIONS AT VELOCITY = 750 ft/sec (x = 7.5):")
    print(f"  1. 5th-Degree Interpolation Prediction: {pred_interp_hundred:.4f} (hundred lb) = {pred_interp_lb:,.2f} lb")
    print(f"  2. Quadratic Regression Prediction    : {pred_quad_hundred:.4f} (hundred lb) = {pred_quad_lb:,.2f} lb")
    print(f"  Absolute Difference                   : {abs(pred_interp_lb - pred_quad_lb):.2f} lb ({(abs(pred_interp_lb - pred_quad_lb)/pred_interp_lb)*100:.2f}%)")
    print("=" * 70)

    # Visualization
    x_dense = np.linspace(-0.5, 10.5, 400)
    y_interp_dense = np.polyval(c_interp[::-1], x_dense)
    y_quad_dense = c_quad[0] + c_quad[1] * x_dense + c_quad[2] * (x_dense ** 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, y_interp_dense, 'r-', lw=2.2, label='5th-Degree Interpolating Polynomial $p(x)$')
    plt.plot(x_dense, y_quad_dense, 'b--', lw=2.2, label='Quadratic Least Squares Fit $p(x)$')
    plt.scatter(x, y, color='black', s=80, zorder=5, label='Experimental Observations')

    # Highlight 750 ft/sec estimates
    plt.scatter([x_target], [pred_interp_hundred], color='red', marker='^', s=130, zorder=6,
                label=f'Interp Estimate @ 750 ft/s ({pred_interp_hundred:.2f} hundred lb)')
    plt.scatter([x_target], [pred_quad_hundred], color='blue', marker='s', s=110, zorder=6,
                label=f'Quadratic Estimate @ 750 ft/s ({pred_quad_hundred:.2f} hundred lb)')

    plt.axvline(x=7.5, color='gray', linestyle=':', alpha=0.6)
    plt.title('Air Resistance Force vs. Projectile Velocity (Interpolation vs. Regression)', fontsize=13, fontweight='bold')
    plt.xlabel('Velocity (in 100 ft/sec), $x$', fontsize=11)
    plt.ylabel('Force (in 100 lb), $y$', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q3_interpolation_vs_regression.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nPlot saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
