import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve


def HilbertIndices(n):
    """
    Generate Hilbert indices for a grid approximating a square layout of patches.

    Args:
    n (int): Total number of patches.

    Returns:
    np.array: Indices sorted according to Hilbert curve distances.
    """
    # Determine the order of the Hilbert curve based on the number of patches
    order = int(np.ceil(np.log2(np.sqrt(n))))
    print("Order is ", order)

    # Initialize Hilbert curve with 2 dimensions and determined order
    hilbert_curve = HilbertCurve(p=order, n=2)
    points = []

    # Determine the maximum x and y values based on the number of patches
    side_length = int(np.ceil(np.sqrt(n)))

    # Collect all points whose Hilbert distances need to be calculated
    for i in range(n):
        x = i // side_length
        y = i % side_length
        points.append([x, y])

    # Calculate the Hilbert distances for all points
    distances = hilbert_curve.distances_from_points(points)

    # Return the indices sorted by their Hilbert distances
    return np.array(distances).argsort()
