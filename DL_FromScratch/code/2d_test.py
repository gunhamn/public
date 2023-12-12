import numpy as np
import matplotlib.pyplot as plt


# Number of points
n_points = 50

# Generate random points
np.random.seed(0)
train_dots = np.random.rand(n_points, 2) * 2 - 1  # points in range [-1, 1]

# Labels for the points
train_dots_labels = np.array([1 if y > x else 0 for x, y in train_dots])


# Visualize the data
plt.figure(figsize=(6,6))
plt.scatter(train_dots[train_dots_labels == 1][:, 0], train_dots[train_dots_labels == 1][:, 1], color='blue', label='Label 1')
plt.scatter(train_dots[train_dots_labels == 0][:, 0], train_dots[train_dots_labels == 0][:, 1], color='red', label='Label 0')
plt.plot([-1, 1], [-1, 1], 'k--')  # line y = x
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title(f'{n_points} Points Linearly Separable by y=x')
plt.show()

# Output the first few elements of train_dots and train_dots_labels
train_dots[:5], train_dots_labels[:5]
