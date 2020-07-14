# CLoNe_clustering

    CLoNe - Clustering based on local density neighborhoods
    Laboratory for Biomolecular Modeling
    Ecole Polytechnique Federale de Lausanne (EPFL)
    Switzerland


For a quick overview, you can open our jupyter notebook, clones_examples.ipynb.

CLoNe is a clustering algorithm with highly general applicability. Based on the Density Peaks algorithm from Rodriguez and Laio (Science, 2014), it improves on it by requiring a single parameter, 'pdc', that is intuitive and easy to use. 'pdc' can be incremeted if there are too many clusters, and decremented if there are not enough. Integer values between 1 and 10 are usually enough, with many values leading to the same results in most cases.  

CLoNe first performs a Nearest Neighbour step to derive the local densities of every data point. Putative cluster centers are then identified as local density maxima. Then, CLoNe takes advantage of the Bhattacaryaa coefficient to merge clusters if needed and relies on a Bayes classifier to effectively remove outliers.

CLoNe was developed mostly for use with structural ensembles, such as those obtained from Molecular Dynamics simulations or integrative modeling attempts. Written in Python3.7, it can be used as any other clustering tool from the Scikit-learn package. When applied to structural ensembles, it will output helpful scripts for automatic loading of the results in the molecular visualization software VMD.

1-Requirements
----

CLoNe was developed with python 3.7 and the following libraries:
- scikit-learn 0.21.3
- scipy 1.3.1
- statsmodels 0.10.0
- numpy 1.17.2
- matplotlib 3.1.0
- mdtraj 1.9.3


2-Running tests
----
To test on a series of examples, please run:
    
    python run_data.py

Then select the dataset you want to cluster. Plots will be saved in the current folder.
You can use the same script with another dataset of your choice and by specifying a value for CLoNe's input parameter:
    
    python run_data.py examples/spiral_quartet.txt 2

Alternatively, you can use clone.py as you would any clustering algorithm from scikit-learn's cluster module. Here's the minimal code to use:

    from clone import CLoNe
    clone = CLoNe() # or e.g. CLoNe(pdc=8) to change the input parameter
    clone.fit(data)  # clone.labels_ to access results

Refer to 'run_test.py' or the Jupyter-notebook included for more information.

3a-Running on structural data
----
The script to run CLoNe is md_clone.py and requires at least a trajectory file to run (and topology if the trajectory is not a multi-pdb file). 
You can perform PCA on the atom selection (-at_sel, "name CA or name BB" by default) of your choice by using the "-pca N" option, where N is the amount of principal components you'd like to consider.
Alternatively, you can use a list of pre-computed features stored in a text file (first line are the headers, 1 frame per row, 1 feature per column). See examples of such files in the 'structural_ensembles/' folder if needed. 


    python run_structural.py -traj mytraj.xtc -topo mytopo.gro -pdc 5 -at_sel "name CA"
    python run_structural.py -traj mytraj.xtc -topo mytopo.gro -pdc 5 -at_sel "name CA" -pca 3
    python run_structural.py -traj mytraj.xtc -topo mytopo.gro -pdc 5 -feat "features.txt"
    python run_structural.py -traj mytraj.xtc -topo mytopo.gro -pdc 5 -feat "features.txt" -pca 2


The '-feat' and '-at_sel' options are mutually exclusive: -feat will take priority over -at_sel. If -pca is set and a feature file is provided, PCA will be done on the features loaded from the file.

For compatible trajectory and topology formats, see documentation of MDtraj.

The atom selection uses the syntax of MDTraj.

Alternatively, you can save specific parameter/file configurations in the 'md_config.ini' file. See the file for examples. This allows you to keep track and repeat some runs easily:

    python run_structural.py -c MY_SYSTEM
    python run_structural.py -c APP


3b-Output files
----
At the end of the run, plots are shown, statistical info is written in terminal, and the path where output files are saved is written in the terminal. CLoNe also creates an output folder containing several files. It is stored in the 'results/' folder, in another folder with either the name of the config used (-c option) or the name of the trajectory.

In that output folder, you'll find:
- Plots
- Cluster summary
- Parameters used
- Cluster centers as PDB files
- Cluster subtrajectories as XTC files, and the original topology to open files easily with the scripts below
- Two Tcl scripts for use with VMD in the output folder.
- PCA projected coordinates in a text file if PCA was done

To run the Tcl scripts, do in a terminal:
    
    vmd -e load_centers_VMD.tcl
    
    vmd -e load_clusters_VMD.tcl

By default, they show VDW and color by fragments. If you open these Tcl files, you'll find inside some examples to change the representations to your tastes.

4-Running on structural data from the paper
----
In a terminal:

    python run_structural.py -c APP
    python run_structural.py -c APP_bound

If you wish to repeat the results of other systems in the paper just to see plots and statistics (e.g., TEM1), you can use 'test_structural.py' instead of 'run_structural.py' as the trajectory files are too big to share here.

The '-c' argument will fetch the parameters and information from the corresponding section in the config file, md_config.ini. The value given to '-c' must be the same as the section name in the file.

You can add new sections for your own datasets in that config file by following the same format.



Enjoy !
