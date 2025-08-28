import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Generate synthetic dataset
# ---------------------------
np.random.seed(42)
X = 2 * np.random.rand(100, 1)   # 100 data points
y = 4 + 3 * X + np.random.rand(100, 1)  # true relation y = 4 + 3x + noise

# Batch Gradient Descent
# ---------------------------
def batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(y)
    w, b = 0.0, 0.0
    cost_history = []

    for i in range(n_iterations):
        y_pred = w * X + b
        dw = (1/m) * np.sum((y_pred - y) * X)
        db = (1/m) * np.sum(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        cost_history.append(cost)

    return w, b, cost_history

# ---------------------------
# Stochastic Gradient Descent
# ---------------------------
def stochastic_gradient_descent(X, y, learning_rate=0.1, n_epochs=50):
    m = len(y)
    w, b = 0.0, 0.0
    cost_history = []

    for epoch in range(n_epochs):
        for i in range(m):
            rand_i = np.random.randint(m)  # pick random sample
            xi = X[rand_i:rand_i+1]
            yi = y[rand_i:rand_i+1]

            y_pred = w * xi + b
            dw = (y_pred - yi) * xi
            db = (y_pred - yi)

            w -= learning_rate * dw
            b -= learning_rate * db

        # compute cost at end of epoch
        y_pred_all = w * X + b
        cost = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        cost_history.append(cost)

    return w, b, cost_history

# ---------------------------
# Mini-Batch Gradient Descent
# ---------------------------
def mini_batch_gradient_descent(X, y, learning_rate=0.1, n_epochs=50, batch_size=16):
    m = len(y)
    w, b = 0.0, 0.0
    cost_history = []

    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]

            y_pred = w * xi + b
            dw = (1/len(xi)) * np.sum((y_pred - yi) * xi)
            db = (1/len(xi)) * np.sum(y_pred - yi)

            w -= learning_rate * dw
            b -= learning_rate * db

        # compute cost at end of epoch
        y_pred_all = w * X + b
        cost = (1/(2*m)) * np.sum((y_pred_all - y)**2)
        cost_history.append(cost)

    return w, b, cost_history

# ---------------------------
# Compare All Methods
# ---------------------------
w_bgd, b_bgd, cost_bgd = batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=100)
w_sgd, b_sgd, cost_sgd = stochastic_gradient_descent(X, y, learning_rate=0.01, n_epochs=50)
w_mgd, b_mgd, cost_mgd = mini_batch_gradient_descent(X, y, learning_rate=0.05, n_epochs=50, batch_size=16)

plt.figure(figsize=(10,6))
plt.plot(cost_bgd, label="Batch GD")
plt.plot(cost_sgd, label="Stochastic GD")
plt.plot(cost_mgd, label="Mini-Batch GD")
plt.xlabel("Iterations/Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Convergence Comparison: Batch vs SGD vs Mini-Batch")
plt.legend()
plt.show()

# ---------------------------
# Final Models
# ---------------------------
print(f"Batch GD final model: y = {w_bgd:.2f}x + {b_bgd:.2f}")
print(f"SGD final model:     y = {w_sgd:.2f}x + {b_sgd:.2f}")
print(f"Mini-Batch final:    y = {w_mgd:.2f}x + {b_mgd:.2f}")

# Plot regression line from Batch GD (smoothest one)
plt.scatter(X, y, label="Data")
plt.plot(X, w_bgd*X + b_bgd, color="red", label="Batch GD Line")
plt.plot(X, w_sgd*X + b_sgd, color="green", linestyle="--", label="SGD Line")
plt.plot(X, w_mgd*X + b_mgd, color="blue", linestyle=":", label="Mini-Batch Line")
plt.legend()
plt.show()
