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

if len(sys.argv) == 3:
    name = sys.argv[1]
    pdc = float(sys.argv[2])
    idx = 0

else:
    print("Chose a dataset below: ")
    print(" 1. Noisy circles        8. Spiral")
    print(" 2. Noisy moons          9. Aggregation")
    print(" 3. Varied              10. S4")
    print(" 4. Anisotropic         11. dim64")
    print(" 5. Blobs               12. dim128")  
    print(" 6. No structure        13. dim512")
    print(" 7. Flame               14. dim1024")

    idx = 0
    while idx < 1 or idx > 14:
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
        ("dim64.txt",         4),
        ("dim128.txt",        4),
        ("dim512.txt",        4),
        ("dim1024.txt",       4),
        ]

    name = "examples/%s"%datasets[idx][0]
    pdc = datasets[idx][1]
    
with open(name, 'r') as f:
    if idx < 10:
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
if data.shape[1] > 3:
    print("> WARNING: data has more than 3 dimensions. Not plotting.")
else:
    plot_clusters(clone, data, ".")
