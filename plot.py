import numpy as np
from itertools import cycle, islice
from scipy.stats.kde import gaussian_kde
from sklearn.decomposition import PCA

import statsmodels.api as sm
import matplotlib.pyplot as mplot
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def plot_clusters(clone, data, path, headers=None):
    color_list = np.array(['yellowgreen', 'orange',  'crimson', 'mediumpurple', 'deepskyblue', 'Aquamarine', 'DarkGoldenRod',
              'Khaki', 'SteelBlue', 'Olive', 'Violet', 'DarkSeaGreen', 'RosyBrown', 'LightPink', 'DodgerBlue',
              'lightcoral', 'chocolate', 'burlywood', 'cyan', 'olivedrab', 'palegreen', 'turquoise', 'gold', 'teal',
              'hotpink', 'moccasin', 'lawngreen', 'sandybrown', 'blueviolet', 'powderblue', 'plum', 'springgreen',
              'mediumaquamarine', 'rebeccapurple', 'peru', 'lightsalmon', 'khaki', 'sienna', 'lightseagreen', 'lightcyan'])
    colors = np.array(list(islice(cycle(color_list),len(clone.centers))))

    if headers is None:
        headers = ["C%i"%x for x in range(data.shape[1])]
    if data.shape[1] == 1:
        plot_1d_clusters(clone, data, path, headers, colors)
    elif data.shape[1] == 2:
        plot_2d_clusters(clone, data, path, headers, colors)
    else:
        plot_nd_clusters(clone, data, path, headers, colors)
    mplot.show()


def plot_1d_clusters(clone, data, path, headers, colors):
    centers = np.array(clone.centers)
    labels = np.array(clone.labels_)
    labels_all = np.array(clone.labels_all)
    core = clone.core_card
    rho = clone.rho

    # Mask for plotting
    assigned_mask = np.where(labels != -1)
    outliers_mask = np.where(labels == -1)

    # KDE
    kde = sm.nonparametric.KDEUnivariate(data.astype(np.float))
    kde.fit()

    # Sort some values for better visualization after
    arcore = np.argsort(core)
    s_cores = core[arcore]
    s_x = data[arcore]

    # Plot core
    mplot.figure(figsize=(4, 2))
    mplot.scatter(s_x, [0] * len(data), marker='|', linewidth=0.1, s=150, c=s_cores, cmap=cm.nipy_spectral)
    mplot.yticks([])
    mplot.xlabel(headers[0], fontsize=15)
    cbar = mplot.colorbar()
    mplot.tick_params(axis='x', which='major', length=8, width=2, labelsize=15)
    cbar.ax.tick_params(axis='x', which='major', length=8, width=2, labelsize=15)
    cbar.ax.set_xlabel("#core", fontsize=15)
    mplot.tight_layout()
    mplot.savefig(path + "/cores.png", dpi=300)
    
    # Plot clusters
    mplot.figure(figsize=(4, 4))
    mplot.scatter(data[assigned_mask, 0], [0] * len(data[assigned_mask]), marker='|', color=colors[labels[assigned_mask]])
    mplot.scatter(data[outliers_mask, 0], [0] * len(data[outliers_mask]), marker='x', s=10, color='black')
    mplot.scatter(data[centers, 0], [0] * len(data[centers]), marker='*', s=100, color='black')
    mplot.plot(kde.support, kde.density, c="grey", marker="None", linestyle="--")
    mplot.xlabel(headers[0])
    mplot.ylabel("KDE")

    mplot.tight_layout()
    mplot.savefig(path + "/clusters.png", dpi=300)


def plot_2d_clusters(clone, data, path, headers, colors):
    centers = np.array(clone.centers)
    labels = np.array(clone.labels_)
    core_card = clone.core_card
    rho = clone.rho

    # Core cardinality mapped to each point
    size = (3.5,4)
    cluster_fig = mplot.figure(figsize=size)

    arcore = np.argsort(core_card)
    s_cores = core_card[arcore]
    s_data = data[arcore]

    mplot.scatter(s_data[:, 0], s_data[:, 1], marker='o', c=s_cores, cmap=cm.nipy_spectral)
    mplot.xlabel("#core", labelpad=20, fontsize=15)
    cbar = mplot.colorbar(orientation='horizontal')
    cbar.ax.tick_params(axis='x', which='major', length=8, width=2, labelsize=15)
    cbar.ax.tick_params(axis='y', which='major', length=8, width=2, labelsize=15)
    mplot.scatter(data[centers, 0], data[centers, 1], marker='*', s=150, c='black', cmap=cm.nipy_spectral)
    mplot.xticks([])
    mplot.yticks([])
    mplot.tight_layout()
    mplot.savefig(path + "/cores.png")

    # Cluster plot by labels
    size = (4,4)
    cluster_fig = mplot.figure(figsize=size)
    ax_clus = cluster_fig.add_subplot(111)
    assigned_mask = np.where(labels != -1)
    outliers_mask = np.where(labels == -1)
    ax_clus.set_xlabel(headers[0])
    ax_clus.set_ylabel(headers[1])
    ax_clus.scatter(data[outliers_mask, 0], data[outliers_mask, 1], marker='x', s=10, color='black', alpha=0.7)
    ax_clus.scatter(data[assigned_mask, 0], data[assigned_mask, 1], marker='o', s=30, edgecolors='black', linewidth=0.1, color=colors[labels[assigned_mask]])
    ax_clus.scatter(data[centers, 0], data[centers, 1], marker='*', s=150, c='black', cmap=cm.nipy_spectral)
    mplot.tight_layout()
    mplot.savefig(path + "/clusters.png")


def plot_nd_clusters(clone, data, path, headers, colors):
    centers = np.array(clone.centers)
    nb_clust = len(centers)
    data_dim = data.shape[1]
    labels = np.array(clone.labels_)
    labels_all = np.array(clone.labels_all)
    core = clone.core_card
    rho = clone.rho

    # Mask for plotting
    assigned_mask = np.where(labels != -1)
    outliers_mask = np.where(labels == -1)

    fig_3d = mplot.figure(figsize=(4,4))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.scatter(data[outliers_mask, 0], data[outliers_mask, 1], data[outliers_mask, 2], marker='.', color='black')
    ax.scatter(data[assigned_mask, 0], data[assigned_mask, 1], data[assigned_mask, 2], marker='o', color=colors[labels[assigned_mask]])
    ax.set_xlabel(headers[0])
    ax.set_ylabel(headers[1])
    ax.set_zlabel(headers[2])
    mplot.savefig(path + "/clusters_3D.png", dpi=300)
    
    # Make plot
    fig = mplot.figure(figsize=(4,4))
    for n in range(data_dim):
        ax = fig.add_subplot(data_dim, 1, n+1)
        ax.set_xlabel("")

        # Plot distributions per cluster
        for clus_idx in range(len(centers)):
            cur_coords = np.array(data[labels == clus_idx, n],ndmin=2)
            p = ax.violinplot(cur_coords[0], [clus_idx+1], showmeans=False, showextrema=False, showmedians=False, widths=0.8)
            for pc in p['bodies']:
                pc.set_facecolor(colors[clus_idx])
                pc.set_edgecolor('black')
                pc.set_alpha(1)

        mplot.yticks(rotation=90, va="center")
        ax.set_ylabel(headers[n], labelpad=10, rotation=90)
        
        ax.grid(True, which='major', axis='y', linestyle='--', c='black')

        # Set x axis only for last dimension
        ax.set_xticks(np.arange(0,nb_clust+2, 1))
        ax.get_xaxis().set_visible(False)
        if n == data_dim - 1:
            ax.get_xaxis().set_visible(True)
            labels = []
            for n in range(nb_clust):
                labels.append("C%i"%(n+1))
            ax.set_xticks(np.arange(1,nb_clust+2, 1))
            ax.set_xticklabels(labels)
            mplot.tick_params(axis='x', pad=20)
        else:
            ax.get_xaxis().set_visible(False)
        ax.set_xlim(xmin=0.3, xmax=(0.8+nb_clust))
    mplot.xticks(rotation=90, va="center")
    mplot.tight_layout()
    mplot.savefig(path + "/clusters_violin.png", dpi=300)
