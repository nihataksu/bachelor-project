import matplotlib.pyplot as plt
import os


def squared(n, a):
    squares = []
    for i in range(n):
        squares.append(a * i**2)
    return squares


def train(n, a, experiment_name):
    working_folder = os.path.join("experiment_results", experiment_name)

    os.makedirs(working_folder, exist_ok=True)
    squared_numbers = squared(n, a)

    # Plot the squared numbers
    plt.plot(squared_numbers, marker="o")
    plt.title("Squared Numbers")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)

    file_path = os.path.join(working_folder, "squared_numbers_plot.png")

    plt.savefig(file_path)

    print(os.getenv("NO_PLT_SHOW"))

    NO_PLT_SHOW = os.getenv("NO_PLT_SHOW") == "True"

    print(NO_PLT_SHOW)

    if not NO_PLT_SHOW:
        plt.show()
