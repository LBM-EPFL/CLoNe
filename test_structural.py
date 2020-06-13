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

import os
import time
import shutil
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clone import CLoNe
from plot import plot_clusters
from structural_utils import load_md_args, show_cluster_info

import mdtraj

print(__doc__)
start = time.time()

# Parse, get parameters
[pdc, n_resize, filt, verbose, traj, topo, at_sel, feat, pca, system_name] = load_md_args(argparse.ArgumentParser())

# Create result folder based on dataset name
if not os.path.exists("results"):
    os.makedirs("results")
out_idx = 1
output_folder = "results/%s_%i/"
while os.path.exists(output_folder%(system_name, out_idx)):
    out_idx += 1
output_folder = output_folder%(system_name, out_idx)
os.makedirs(output_folder)

# Load data from feature file or atom selection
if feat != "None":
    with open(feat, "r") as f:
        headers = next(f).split()
        coords = np.array([[float(x) for x in line.split()] for line in f])
    print("> Loading %i features from %s"%(len(headers), feat))
else:
    while True:
        try:
            selection = struct_ens.topology.select(at_sel)
        except Exception:
            at_sel = input("Invalid atom selection; try again: ")
            continue
        break
    coords = struct_ens.xyz[:, selection]
    coords = coords.reshape(coords.shape[0], coords.shape[1] * coords.shape[2])
    headers = ["C%i"%x for x in range(len(coords[0]))]  # general headers

# Principal component analysis
if pca:
    original_coords = coords.copy()
    pca_obj = PCA(n_components=pca)
    coords = pca_obj.fit_transform(coords)
    eigenvalues = pca_obj.explained_variance_ratio_
    ratio = np.sum(eigenvalues[:pca])
    pca_headers = ["PC%i (%.2f)"%(x + 1, eigenvalues[x]) for x in range(pca)]

    print("> PCA: %i => %i dimension(s) with eigenval.: %s"%(len(headers), pca, str(eigenvalues)))

    with open("%sPCA_coords.txt"%output_folder, "w") as f:
        for x in range(pca):
            f.write("PC%i(%.2f) "%(x + 1, eigenvalues[x]))
        f.write("\n")
        for el in coords:
            for n in el:
                f.write("%f "%n)
            f.write("\n")
else:
    original_coords = coords.copy()  # keep for better interpretability
    coords = StandardScaler().fit_transform(coords)

# Cluster data
print("> Clustering %i points, %i dimensions..."%(coords.shape[0], coords.shape[1]))
clone = CLoNe(pdc=pdc, n_resize=n_resize, filt=filt, verbose=verbose)
clone.fit(coords)
runtime = time.time() - start
print("> Run time: %.2fs"%runtime)

# Display results
# > Statistics on unscaled data for better interpretability
# > If PCA was done, chose data to visualize. Sometimes it makes sense to look at PC,
#   sometimes to look at original coords...
if not pca:
    # Stats
    show_cluster_info(clone, original_coords, output_folder, headers)
    # Plot
    plot_clusters(clone, original_coords, output_folder, headers)
else:
    # Stats
    show_original = -1
    while show_original not in [1,2,3]:
        show_original = int(input("> Show statistics on:\n   1. Original coords (%i dimensions)\n   2. PCA coords (%i dimensions)\n   3. Both\n   > Choice: "%(len(headers), len(pca_headers))))
    if show_original == 1:
        show_cluster_info(clone, original_coords, output_folder, headers)
    elif show_original == 2:
        show_cluster_info(clone, coords, output_folder, pca_headers)
    else:
        show_cluster_info(clone, original_coords, output_folder, headers)
        show_cluster_info(clone, coords, output_folder, pca_headers)
    # Plot
    plot_clusters(clone, coords, output_folder, pca_headers)

print("> Saved output files in: %s"%output_folder)
