from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np

p = 1
n = 2
hilbert_curve = HilbertCurve(p, n)
distances = list(range(4))
points = hilbert_curve.points_from_distances(distances)

for point, dist in zip(points, distances):
    print(f"point(h={dist}) = {point}")


points = [[0, 0], [0, 1], [1, 1], [1, 0]]
distances = hilbert_curve.distances_from_points(points)
for point, dist in zip(points, distances):
    print(f"distance(x={point}) = {dist}")


# Assume we have the following array:
a = np.array([10, 20, 30, 40, 50])

# And we have hilbert_indices calculated as follows:
indices = np.array([4, 2, 3, 0, 1])

# Applying fancy indexing:
new_a = a[indices]

print(new_a)
# Output will be: [40 10 50 20 30]
