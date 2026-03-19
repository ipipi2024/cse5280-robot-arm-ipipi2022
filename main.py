import numpy as np
import matplotlib.pyplot as plt

# Initial setup
x = np.array([2.0, 3.0])
g = np.array([10.0, 8.0])

alpha = 0.1

# Setup plot
plt.ion()
fig, ax = plt.subplots()

for _ in range(50):
    # Gradient descent step
    direction = g - x
    x = x + alpha * direction

    # Clear and redraw
    ax.clear()
    ax.set_title("Moving particle (Gradient Descent)")
    
    # Plot current position
    ax.scatter(x[0], x[1], color='blue', label='particle')
    ax.scatter(g[0], g[1], color='red', label='goal')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.legend()

    plt.pause(0.1)

plt.ioff()
plt.show()