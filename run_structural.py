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
from md_utils import load_md_args, show_cluster_info

import mdtraj

print(__doc__)
start = time.time()

# Parse, get parameters
[pdc, n_resize, filt, verbose, traj, topo, at_sel, feat, pca, system_name] = load_md_args(argparse.ArgumentParser())
assert os.path.isfile(traj), "Trajectory %s not found"%traj
assert not traj.endswith('.pdb') and os.path.isfile(topo), "Topology %s not found but required"%topo

# Create result folder based on dataset name
if not os.path.exists("results"):
    os.makedirs("results")
out_idx = 1
output_folder = "results/%s_%i/"
while os.path.exists(output_folder%(system_name, out_idx)):
    out_idx += 1
output_folder = output_folder%(system_name, out_idx)
os.makedirs(output_folder)

# Save parameters used
with open(output_folder + "Parameters.txt", "w") as f:
    f.write("CLoNe parameters:\n")
    f.write("> P_dc: %f\n"%pdc)
    f.write("> Filtering fraction: %f\n"%filt)
    f.write("> n_resize: %f\n"%n_resize)
    f.write("\n")
    f.write("Structural ensemble:\n")
    f.write("> Trajectory: %s\n"%traj)
    f.write("> Topology: %s\n"%topo)
    if feat != "None":
        f.write("> Feature file: %s\n"%feat)
    else:
        f.write("> Selection: %s\n"%at_sel)
    f.write("> PCA: %i\n"%pca)

# Load trajectory
struct_ens = mdtraj.load(traj, top=topo)

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

# Save labels
with open(output_folder + "results.txt", "w") as f:
    [f.write("%i "%c) for c in clone.centers]
    f.write("\n")
    for l, core, r in zip(clone.labels_, clone.core_card, clone.rho):
        f.write("%i %i %f\n"%(l, core, r))

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

# Save frames of cluster centers and clusters as trajectories (XTC is lightest format)
unique_labels = range(len(clone.centers))
print("> Saving centers and sub-trajectories...")
[struct_ens[center].save(output_folder + "Center_%i.pdb"%(center_id + 1)) for center, center_id in zip(clone.centers, unique_labels)]
[struct_ens[clone.labels_ == lab].save(output_folder + "Cluster_%i.xtc"%(lab + 1)) for lab in unique_labels]

# Write VMD script for quick visualization
# > Centers
with open(output_folder + "load_centers_VMD.tcl", "w") as f:
    f.write("set center_list {}\n")
    [f.write("lappend center_list Center_%i.pdb\n"%(center_id + 1)) for center_id in unique_labels]
    f.write("set i 1\n")
    f.write("foreach center $center_list {\n")
    f.write("    mol new %s\n"%topo.split('/')[-1])
    f.write("    mol rename top \"Center $i\"\n")
    f.write("    incr i \n")
    if topo.endswith("pdb") or topo.endswith("gro"):
        f.write("    animate delete all\n")  # VMD adds a frame for topology GRO/PDB format, this removes it
    f.write("    mol addfile $center waitfor all\n")
    f.write("    # Modify these lines to customize representations:\n")
    f.write("    mol modstyle 0 top \"VDW\"\n")
    # Color by fragments (general)
    f.write("    mol modcolor 0 top \"Fragment\"\n")
    # Color by cluster ID
    f.write("    #mol modcolor 0 top \"ColorID\" $i\n")
    
    f.write("    # To add more representations, change 0 to 1 and so on. Below is to add a second representation:\n")
    f.write("    # mol addrep top\n")
    f.write("    # mol modselect 1 top \"fragment 0\"\n")
    f.write("    # mol modstyle 1 top \"Licorice\"\n")
    f.write("    # mol modcolor 1 top \"ColorID\" 32\n")
    f.write("    mol off top\n")
    f.write("}\n")

# > Clusters
with open(output_folder + "load_clusters_VMD.tcl", "w") as f:
    # Copy topology over
    shutil.copy(topo, output_folder)
    f.write("set cluster_list {}\n")
    [f.write("lappend cluster_list Cluster_%i.xtc\n"%(clust_id + 1)) for clust_id in unique_labels]
    f.write("set i 1\n")
    f.write("foreach cluster $cluster_list {\n")
    f.write("    mol new %s\n"%topo.split('/')[-1])
    f.write("    mol rename top \"Cluster $i\"\n")
    f.write("    incr i \n")
    if topo.endswith("pdb") or topo.endswith("gro"):
        f.write("    animate delete all\n")  # VMD adds a frame for topology GRO/PDB format, this removes it    f.write("    mol addfile $cluster waitfor all\n")
    f.write("    mol addfile $cluster waitfor all\n")
    f.write("    # Modify these lines to customize representations:\n")
    f.write("    mol modstyle 0 top \"VDW\"\n")
    f.write("    mol modcolor 0 top \"Fragment\"\n")
    f.write("    #mol modcolor 0 top \"ColorID\" $i\n")
    f.write("    # To add more representations, change 0 to 1 and so on:\n")
    f.write("    # mol addrep top\n")
    f.write("    # mol modselect 1 top \"fragment 0\"\n")
    f.write("    # mol modstyle 1 top \"Licorice\"\n")
    f.write("    # mol modcolor 1 top \"ColorID\" 32\n")
    f.write("    mol off top\n")
    f.write("}\n")

print("> Saved output files in: %s"%output_folder)
