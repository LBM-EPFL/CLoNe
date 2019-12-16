"""
 _________________________________
|  / __ \\/ |        / \\  / |      |
| /./  \\/|.|    __  |..\\ |.| ___  |
| |.|    |.|   /  \\ |...\\|.|/ . \\ |
| \\.\\__/\\|.|__( () )|.|\\.'.|| ._/ |
|__\\____/\\____/\\__/_\\_|_\\__/\\___|_|
|Clustering of local neighborhoods|
|_________________________________|
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import sys

from plot import plot_clusters
from clone import CLoNe



print(__doc__)
print("Chose a dataset below: ")
print("  1. Noisy circles")
print("  2. Noisy moons")
print("  3. Varied")
print("  4. Anisotropic")
print("  5. Blobs")
print("  6. No structure")
print("  7. Flame")
print("  8. Spiral")
print("  9. Aggregation")
print(" 10. S4")

idx = 0
while idx < 1 or idx > 10:
    idx = int(input(">> #: "))
idx -= 1

default = 10
datasets = [
    ("noisy_circles.txt", 6),
    ("noisy_moons.txt",   default),
    ("varied.txt",        default),
    ("aniso.txt",         default),
    ("blobs.txt",         default),
    ("no_structure.txt",  default),
    ("flame.txt",         default),
    ("spiral.txt",        8),
    ("aggregation.txt",   4),
    ("s4.txt",            3),
    ]

name = datasets[idx][0]
pdc = datasets[idx][1]
with open("examples/%s"%name, 'r') as f:
    next(f)
    data = np.array([[float(x) for x in line.split()] for line in f])
    data = StandardScaler().fit_transform(data)

print("> Clustering %s..."%name)
t = time.time()
clone = CLoNe(pdc=pdc, verbose=False)
clone.fit(data)
print("> Done: %.2f sec"%(time.time() - t))

# Get data from clustering
centers = clone.centers
core_card = clone.core_card
labels = clone.labels_
labels_all = clone.labels_all
rho = clone.rho

# Summary
header = "  |  #center  |    Dens    #Core  |  # el  | -outl  |"
subh  =  "  |-----------|-------------------|--------|--------|"
top = "   " + "-" * (len(header) - 4)
print(top + "\n" + header + "\n" + subh + "\n" + top)
for c in range(len(centers)):
    elem = len(labels_all[labels_all == c])
    outl = len(labels[labels == c])
    line =  "  |%2i - %5i | %7.2f  %7i  | %6i | %6i |"%(c+1, centers[c]+1, rho[centers[c]], core_card[centers[c]], elem, outl)
    print(line)
print(top)

# Plot
plot_clusters(clone, data, ".")
