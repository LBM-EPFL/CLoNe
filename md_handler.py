import shutil
import numpy as np
import warnings
from configparser import ConfigParser


def load_md_args(parser):
    parser.add_argument("-c", type=str, default="DEFAULT", help="Loads parameters and filenames from md_config.ini if applicable")

    # CLoNe arguments
    parser.add_argument("-pdc", type=float, default=0, help="Relates to the sigma of Gaussian kernel used to compute local densities of each point. Use integers or half-integer values. Default: 4.")
    parser.add_argument("-n_resize", type=float, default=0, help="Neighbour matrix resize factor. Reduces the number of neighbours to consider when building the neighbour matrix by dividing the number of elements by the provided value. Defaults to 1 (i.e. all neighbours considered).")
    parser.add_argument("-filt", type=float, default=0, help="If enabled, filters out outliers.")
    parser.add_argument("-verbose", action='store_true', help="CLoNe verbosity")          

    # MD-specific arguments
    parser.add_argument("-traj", type=str, default="", help="Path and filename of trajectory to cluster")
    parser.add_argument("-topo", type=str, default="", help="Path and filename of topology file. Leave blank in case of multiframe PDB")
    parser.add_argument("-at_sel", type=str, default="name CA or name BB", help="Atom selection. Default to 'name CA or name BB'. Follows MDAnalysis' naming conventions. Resulting features used for clustering. If PCA is enabled, it will be applied on the selection.")
    parser.add_argument("-feat", type=str, default="", help="Text file containing features to use for clustering. If PCA is enabled, it will be applied on these features.")
    parser.add_argument("-pca", type=int, default=0, help="Whether to perform PCA on vectors. Value corresponds to the number of principal component to consider.")
    args = parser.parse_args()

    # Check if parameter sets already exists in md_confid.ini; else, load default.
    c_section = args.c
    config = ConfigParser()
    config.read('md_config.ini')
    if c_section not in config.sections():
        assert args.traj, "Invalid input arguments and no trajectory specified"
        print("> Loading default parameters")
        c_section = "DEFAULT"
    else:
        print("> Loading config for %s"%c_section)
    c = config[c_section]

    # Get values from md_config.ini
    pdc = c.getfloat('pdc')
    n_resize = c.getfloat('n_resize')
    filt = c.getfloat('filt')
    verbose = c.getboolean('verbose')
    traj = c.get('traj')
    topo = c.get('topo')
    at_sel = c.get('at_sel')
    feat = c.get('feat')
    pca = c.getint('pca')

    # Update from command-line arguments
    # CLoNe
    if args.pdc:
        pdc = args.pdc
    if args.n_resize:
        n_resize = args.n_resize
    if args.filt:
        filt = args.filt
    if args.verbose:
        verbose = args.verbose

    # MD-specific
    if args.traj:
        traj = args.traj
    if args.topo:
        topo = args.topo
    if args.at_sel:
        at_sel = args.at_sel
    if args.feat:
        feat = args.feat
    if args.pca:
        pca = args.pca
    return [pdc, n_resize, filt, verbose, traj, topo, at_sel, feat, pca, c_section]

def show_cluster_info(clone, data, path, headers):
    centers = clone.centers
    labels = clone.labels_
    labels_all = clone.labels_all
    core = clone.core_card
    rho = clone.rho

    outname = path + "Summary_clusters.txt"
    with open(outname, "a") as f:
        header = "  |  #center  |    Dens     Core  |"
        subh  =  "  |-----------|-------------------|"
        for h in headers:
            hs = h[:15].center(15)
            header = header + "      " + hs + "      |"
            subh  =    subh + "   center   median    IQR  |"
        header = header + "  # el  | -outl  |"
        subh   =   subh + "--------|--------|"
        
        top = "   " + "-" * (len(header) - 4)
        f.write(top + "\n" + header + "\n" + subh + "\n" + top + "\n")
        print(top + "\n" + header + "\n" + subh + "\n" + top)
        for cl in range(len(centers)):
            elem = len(labels_all[labels_all == cl])
            outliers = len(labels[labels == cl])
            line =  "  |%2i - %5i | %7.2f  %7i  "%\
                    (cl+1, centers[cl]+1, rho[centers[cl]], core[centers[cl]])
            iqr_list = []
            for dim in range(data.shape[1]):
                centr_val = data[centers[cl],dim]
                quartile1, median, quartile3 = np.percentile(data[labels == cl, dim], [25, 50, 75])
                iqr = quartile3 - quartile1
                line = line + "| %8.2f %8.2f %7.2f "%(centr_val, median, iqr)
            line = line + "| %6i | %6i |"%(elem, outliers)
            f.write(line + "\n")
            print(line)
        f.write(top + "\n")
        print(top)
