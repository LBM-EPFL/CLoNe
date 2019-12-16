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
    
    python run_test.py

Then select the dataset you want to cluster. Plots will be saved in the current folder.

Outside of these tests, you can use clone.py as you would any clustering algorithm from scikit-learn's cluster module.

    from clone import CLoNe
    clone = CLoNe()
    clone.fit(data)  # clone.labels_ to access results

Refer to 'run_test.py' or the Jupyter-notebook included for more information.


3-Running on structural data
----
To run CLoNe on structural examples:
   
    python md_clone.py -c APP_bound

If you wish to repeat the results of other systems in the paper just to see plots and statistics (e.g., TEM1), you can use 'run_structural.py' instead of 'md_clone.py'.

The '-c' argument will fetch the parameters and information from the corresponding section in the config file, md_config.ini. The value given to '-c' must be the same as the section name in the file.

You can add new structural examples in the same format in that config file.

You can update the specific value of parameters via the command line while still using a parameter set, such as the atom selection or the pdc parameter of CLoNe:

    python md_clone.py -c SYSTEM_NAME -pdc NEW_VALUE
    
or

    python md_clone.py -c SYSTEM_NAME -at_sel "name CA and resid 15 to 100"

A full example without using the config file would be:

    python md_clone.py -traj mytraj.xtc -topo mytopo.gro -pdc 5 -at_sel "name CA" -pca 3

This loads the trajectory "mytraj" and its topology, extract the CA atoms xyz coordinates, performs PCA on them and the clustering will be based on the first 3 principal components.

For compatible trajectory and topology formats, see documentation of MDtraj.

The atom selection uses the syntax of MDTraj. While you have basic examples above, please check their documentation for up to date selection syntax.

At the end of the run, plots are shown, statistical info is written in terminal, and the path where output files are saved is written in the terminal.

In the output folder, you'll find:
- Plots
- Cluster summary
- Parameters used
- Cluster centers as PDB files
- Cluster subtrajectories as XTC files, and the original topology to open files easily with the scripts below
- Two Tcl scripts for use with VMD in the output folder.

To run the Tcl scripts, do in a terminal:
    vmd -e load_centers_VMD.tcl
    vmd -e load_clusters_VMD.tcl

By default, they show VDW and color by fragments. If you open these Tcl files, you'll find inside some examples to change the representations to your tastes.

Enjoy !
