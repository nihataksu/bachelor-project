import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def gelu(x):
    return x * norm.cdf(x)


# Generate a range of values from -3 to 3, which is typical for viewing activation functions
x_values = np.linspace(-3, 3, 300)
y_values = gelu(x_values)

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(x_values, y_values, label="GELU")
plt.title("GELU Activation Function")
plt.xlabel("Input value (x)")
plt.ylabel("Output value")
plt.grid(True)
plt.legend()
plt.show()
