import operator
from copy import deepcopy
import numpy as np
from numpy.polynomial.polynomial import polyfit
from sklearn.neighbors import NearestNeighbors

class CLoNe(object):
    '''
    CLoNe clustering
    
    Parameters
    ----------
    factor_dc: float, optional, default: 4
        The only parameter that may need tuning.
        factor_dc is a percentage and will be used as index in the sorted, flattened neighbour matrix to extract the cut-off distance dc.
        In turn, dc is used in the gaussian kernel to compute local densities rho. Increasing (decreasing) factor_dc will increase (decrease)
        the reach of the gaussian kernel.

        Rule of thumb:
        > If you think there should be more clusters, decrease this value (integer increments are usually enough).
        > If you think there should be less clusters, increase this value.

    n_resize: int, optional, default: 1
        Divides the axis 1 of the neighbour matrix with n_resize. It decreases the number of neighbour computed for each point.
        This is optional and can be used to decreases computing time and memory use. This is useful for larger datasets.
        It may affect clustering results if the value is set too high, which may be corrected by adapting factor_dc above.
        Generally, a value of at least 2 is safe to use. As a rule of thumb, you should not resize the matrix so that its axis 1 becomes smaller than
        the largest/densest cluster (if no prior knowledge is available, then it is trial and error)

    filt: float, optional, default: 0.2
        If non-zero, enables the filtering of outliers.
        After cluster assignation, local densities are recomputed for each cluster separately. Points within a cluster whose local density are below
        the set threshold * local density of the cluster center are categorised as outliers and unassigned.

    n_int: int, optional, default: 10
        The computation of gnt occurs by measuring volumetric neighbour density at specific intervals of closest neighbours around each point.
        This sets this interval. There is no need to change it, but may decrease computation time without loss of accuracy for large datasets + large clusters.

    more_params: boolean, optional, default: False
        If enabled, more data will be available from the object after fitting the data. See attributes below.
        
    verbose: boolean, optional, default: False
        enables verbose output of CLoNe


    Attributes - single hierarchy
    ------------------------------
    labels_: array, shape (n_samples,)
        Labels of each point after outlier removal.
        Provided by default.

    labels_all_: array, shape (n_samples,)
        Labels of each point before outlier removal.
        Provided by default.

    centers_: array, shape (n, )
        Index of cluster centers.
        Provided by default.

    lnt_: array, (n_samples,)
        Local neighbour thresholds / hypothetical cluster sizes of each data point.
        Provided if more_params is set to True.

    rho_: array, (n_samples,)
        Local densities of each point as computed by a Gaussian kernel
        Provided if more_params is set to True.
        
    delta_: array, (n_samples,)
        Minimal distance to another point of higher density for each point
        Provided if more_params is set to True.

    r2_: array, (n_intervals,)
        Coefficient of determination of every pair of successive intervals of neighbour density (gnt calculation)
        Provided if more_params is set to True.

    neighbour_interval_: int
        Interval of neighbours used when computing neighbour density
        Provided if more_params is set to True.

    gnt_idx_: int
        Index of interval where density changed most in r2_
        Provided if more_params is set to True.

    gnt_: int
        Number of neighbours that lead to the global biggest drop in neighbour density
        Provided if more_params is set to True.

    neigh_density_: array
        Neighbour density around every point at every interval
        Provided if more_params is set to True.


    Attributes - multiple hierarchies
    ---------------------------------
    Same as single hierarchy - except many attributes become lists, ordered from smallest to biggest gnt
    

    Examples
    --------
    See Jupyter notebook "clone_examples" on GitHub


    Notes
    -----
    This algorithm is a variant of Clustering by fast search and find of density peaks (Rodriguez and Laio, 2015, Science).
    Cluster center determination is entirely automated, and is still based on their definition of cluster center (see rho and delta).
    The computation of the latter two has been heavily modified to accomodate for the automated determination of centers, namely the computation of lnt,
    the distance matrix that is now based on NearestNeighbours and the possibility to resize the neighbour matrix to save on time and memory, and the
    possibility to detect distinct meaningful hierarchies if applicable.
    
    '''

    def __init__(self, factor_dc=4,
                 n_resize=1,
                 filt=0.2,
                 hierarchy=False,
                 n_int=10,
                 more_params=False,
                 verbose=False):
                     
        self.factor_dc = factor_dc
        self.n_resize = n_resize
        self.filt = filt
        self.n_int = n_int
        self.more_params = more_params
        self.verbose = verbose
            
    def fit(self, vectors):
        data_dim = len(vectors[0])
        nb_el = len(vectors)
        self.vectors = vectors  # saved for easier plotting afterwards

        ## 1. Build distance matrix
        # Reduce size of distance matrix to be computed
        effective_nb_el = int(nb_el / self.n_resize)

        if self.verbose:
            print("> Computing ordered distance matrix using %.2f%% (%i) neighbours..."%(100.0 * effective_nb_el / nb_el, effective_nb_el))

        # Use kNN to get neighbours + idx
        nbrs = NearestNeighbors(n_neighbors=effective_nb_el, algorithm="auto", n_jobs=1, metric='l2').fit(vectors)
        knn_dist, knn_idx = nbrs.kneighbors(vectors)

        ## 2. Get global neighbour limit
        print(effective_nb_el, self.n_int)
        interval_list = np.arange(self.n_int - 1, effective_nb_el, self.n_int)
        n_intervals = len(interval_list)

        if self.verbose:
            print("> Computing gnt (%i intervals of %i neighbours)..."%(n_intervals, self.n_int))

        # Compute neighbour densities at every interval for all points
        neigh_interval_array = np.array([interval_list for x in range(nb_el)])
        interval_knn_dist = knn_dist[:, interval_list]
        neighbour_density = np.divide(neigh_interval_array, np.pi * np.power(interval_knn_dist, data_dim))
        
        # Get R2 between interval pairs
        mean = np.mean(neighbour_density, axis=0)
        std = np.std(neighbour_density, ddof=1, axis=0)
        z = (neighbour_density - mean) / std
        r2 = (np.sum(z[:, 1:] * z[:, :-1], axis=0) / (neighbour_density.shape[0] - 1)) ** 2

        # Fit a line to windows of r2, get local standard deviation
        r2_std = []
        w_size = 5
        for x in range(0, len(r2), 1):
            window = np.arange(max(0, x - w_size), min(len(r2) - 1, x + w_size + 1))
            r2_window = r2[window]
            b, m = polyfit(window, r2_window, 1)
            r = r2_window - m * window - b
            r2_std.append(np.std(r))
        r2_std = np.array(r2_std)

        # Initialize value at max R2
        min_r2 = max(r2)
        min_idx = n_intervals - 2

        # Get largest drop in R2 that exceed mean std (without tails where std is not fully defined)
        std_mean = np.mean(r2_std[w_size:-w_size])
        for x in range(1, len(r2) - 1, 1):
            if r2[x - 1] > r2[x] and r2[x + 1] > r2[x]:
                if r2_std[x] > std_mean and r2[x] < min_r2:
                    min_r2 = r2[x]
                    min_idx = x
        
        # Global neighbour limit (max size if no minima was found; can happen if n_resize is set too high)
        gnt = interval_list[min_idx]

        ## 3. Clustering data 
        # Get Gaussian sigma for local density based on neighbour matrix
        position = int(round(nb_el * (nb_el - 1) / 2 * self.factor_dc * 0.01))
        flat_dist = knn_dist.flatten()
        flat_dist.sort()
        dc = flat_dist[position * 2 + nb_el]

        # 3a. Compute local densities
        inv_dc = 1 / dc
        inv_gnt = 1 / gnt
        if self.verbose:
            print("> Computing local densities...")

        # Apply gaussian kernel on knn distances
        ordered_knn_dens = np.exp(-1 * np.square(knn_dist * inv_dc))

        # Get local densities by summing over rows
        summed_okd = ordered_knn_dens.cumsum(axis=1) - 1
        rho = summed_okd[:, -1]

        # Weight sum w/r general neighbor threshold
        weighted_summed_okd = summed_okd * inv_gnt

        # Detect where adding neighbors do not contribute enough anymore to the point's local density => local neighbour limits
        lnt_list = np.unravel_index(np.argmax(weighted_summed_okd > ordered_knn_dens, axis=1), weighted_summed_okd.shape)[1]

        # 3b. Get distances to nearest neighbour of higher density
        if self.verbose:
            print("> Computing distance to higher density...")

        delta = [-1] * nb_el
        nneigh = [0] * nb_el
        cl_centers = []
        labels = [-1] * nb_el
        nb_clust = 0
        for x in range(0, nb_el):
            cur_idx_list = knn_idx[x]
            cur_dist_list = knn_dist[x]
            is_set = False
            for y in range(1, effective_nb_el):
                # if y's density is higher, record corresponding distance as delta
                if rho[cur_idx_list[y]] > rho[x]:
                    is_set = True
                    nneigh[x] = cur_idx_list[y]
                    delta[x] = cur_dist_list[y]

                    # If y is beyond lnt; x is a center
                    if y > lnt_list[x]:
                        cl_centers.append(x)
                        nb_clust += 1
                    break

            # If nb_el was reduced, the nearest neighbour of higher density may be beyond the neighbour matrix
            # x is assigned info corresponding to its last neighbour
            # also covers the point of highest density, which would be a cluster center by definition in this approach
            if not is_set:
                nneigh[x] = cur_idx_list[y]
                delta[x] = cur_dist_list[y]

                # If y is beyond lnt; x is a center
                if y > lnt_list[x]:
                    cl_centers.append(x)
                    nb_clust += 1

        # Sort cluster centers according to local densities
        to_sort = []
        for x in range(nb_clust):
            to_sort.append([rho[cl_centers[x]], cl_centers[x]])
        to_sort = sorted(to_sort, key=operator.itemgetter(0))
        cl_centers = [el[1] for el in to_sort]
        for x in range(nb_clust):
            labels[cl_centers[x]] = x
        idx_rho_sorted = np.argsort(-np.array(rho))

        # Set highest density point delta to max delta
        delta[cl_centers[-1]] = np.amax(delta)
        
        ## 4. Cluster assignation
        # Assign points to the same cluster as its nearest neighbour of higher density
        # > do so in order of decreasing density
        if self.verbose:
            print("> Assigning points to clusters...")

        labels = np.array(labels)
        for x in range(nb_el):
            if labels[idx_rho_sorted[x]] == -1:
                labels[idx_rho_sorted[x]] = labels[nneigh[idx_rho_sorted[x]]]

        ## 5. Outlier removal
        labels_all = labels

        # Recompute local densities by cluster. Then, remove points whose density is <X% of their center.
        # > Doing so avoids clusters of low densities to be classified as mainly outliers if they're close a high density cluster.
        if self.filt:
            if self.verbose:
                print("> Filtering outliers...")
            for cl in range(len(cl_centers)):
                # Get points of cluster + new id of center
                cl_mask = np.where(labels == cl)[0]
                center = cl_centers[cl]
                idx_center = np.where(cl_mask==center)
                cl_v = vectors[cl_mask]

                # KNN only on cluster points
                nbrs = NearestNeighbors(n_neighbors=len(cl_v), algorithm="auto", n_jobs=1, metric='l2').fit(cl_v)
                cl_knn_dist, cl_knn_idx = nbrs.kneighbors(cl_v)
                cl_ordered_knn_dens = np.exp(-1 * np.square(cl_knn_dist * inv_dc))

                # Get local densities by summing over rows
                cl_summed_okd = cl_ordered_knn_dens.cumsum(axis=1) - 1
                cl_rho = cl_summed_okd[:, -1]
                
                # rho of center + threshold. 
                d_thresh = np.amax(cl_rho) * self.filt
                
                # Unassign points with d < d_thresh
                for i in range(len(cl_v)):
                    if cl_rho[i] < d_thresh:
                        labels[cl_mask[i]] = -1

        self.labels_ = labels
        self.labels_all_ = labels_all
        self.centers_ = cl_centers
        self.lnt_ = lnt_list
        self.rho_ = rho
        self.delta_ = delta
        self.r2_ = r2
        self.neighbour_interval_ = self.n_int
        self.gnt_idx_ = min_idx
        self.gnt_ = gnt + 1
        self.neigh_density_ = neighbour_density

    def fit_hierarchy(self, vectors):
        data_dim = len(vectors[0])
        nb_el = len(vectors)
        self.vectors = vectors  # saved for easier plotting afterwards

        ## 1. Build distance matrix
        # Reduce size of distance matrix to be computed
        effective_nb_el = int(nb_el / self.n_resize)

        if self.verbose:
            print("> Computing ordered distance matrix using %.2f%% (%i) neighbours..."%(100.0 * effective_nb_el / nb_el, effective_nb_el))

        # Use kNN to get neighbours + idx
        nbrs = NearestNeighbors(n_neighbors=effective_nb_el, algorithm="auto", n_jobs=1, metric='l2').fit(vectors)
        knn_dist, knn_idx = nbrs.kneighbors(vectors)

        ## 2. Get global neighbour limit
        interval_list = np.arange(self.n_int - 1, effective_nb_el, self.n_int)
        n_intervals = len(interval_list)

        if self.verbose:
            print("> Computing gnt (%i intervals of %i neighbours)..."%(n_intervals, self.n_int))

        # Compute neighbour densities at every interval for all points
        neigh_interval_array = np.array([interval_list for x in range(nb_el)])
        interval_knn_dist = knn_dist[:, interval_list]
        neighbour_density = np.divide(neigh_interval_array, np.pi * np.power(interval_knn_dist, data_dim))
        
        # Get R2 between interval pairs
        mean = np.mean(neighbour_density, axis=0)
        std = np.std(neighbour_density, ddof=1, axis=0)
        z = (neighbour_density - mean) / std
        r2 = (np.sum(z[:, 1:] * z[:, :-1], axis=0) / (neighbour_density.shape[0] - 1)) ** 2

        # Fit a line to windows of r2, get local standard deviation
        r2_std = []
        w_size = 5
        for x in range(0, len(r2), 1):
            window = np.arange(max(0, x - w_size), min(len(r2) - 1, x + w_size + 1))
            r2_window = r2[window]
            b, m = polyfit(window, r2_window, 1)
            r = r2_window - m * window - b
            r2_std.append(np.std(r))
        r2_std = np.array(r2_std)

        # Initialize value at max R2
        min_r2 = max(r2)
        min_idx = n_intervals - 2

        # Get drops in R2 that exceed mean std (without tails where std is not fully defined)
        # > keep largest drop. If other drops seem interesting, enable hierarchy
        gnt_list = []
        gnt_idx_list = []
        std_mean = np.mean(r2_std[w_size:-w_size])
        for x in range(1, len(r2) - 1, 1):
            if r2[x - 1] > r2[x] and r2[x + 1] > r2[x]:
                if r2_std[x] > std_mean:
                    gnt_list.append(interval_list[x])
                    gnt_idx_list.append(x)
                    if r2[x] < min_r2:
                        min_r2 = r2[x]
                        min_idx = x
        
        # Global neighbour limit if no levels were found
        if not len(gnt_list):
            gnt_list.append(interval_list[min_idx])
        
        ## 3. Clustering data by hierarchy
        level_check = []
        
        # Get Gaussian sigma for local density based on neighbour matrix
        position = int(round(nb_el * (nb_el - 1) / 2 * self.factor_dc * 0.01))
        flat_dist = knn_dist.flatten()
        flat_dist.sort()
        dc = flat_dist[position * 2 + nb_el]

        # Placeholders for results
        self.labels_ = []
        self.labels_all_ = []
        self.centers_ = []
        self.lnt_ = []
        self.rho_ = []
        self.delta_ = []
        self.gnt_ = []
        self.gnt_idx_ = []
        
        for gnt, gnt_idx in zip(gnt_list, gnt_idx_list):
            if self.verbose:
                print("\n> Clustering based on gnt=%i"%(gnt + 1))
                
            # 3a. Compute local densities
            inv_dc = 1 / dc
            inv_gnt = 1 / gnt
            if self.verbose:
                print("> Computing local densities...")

            # Apply gaussian kernel on knn distances
            ordered_knn_dens = np.exp(-1 * np.square(knn_dist * inv_dc))

            # Get local densities by summing over rows
            summed_okd = ordered_knn_dens.cumsum(axis=1) - 1
            rho = summed_okd[:, -1]

            # Weight sum w/r general neighbor threshold
            weighted_summed_okd = summed_okd * inv_gnt

            # Detect where adding neighbors do not contribute enough anymore to the point's local density => local neighbour limits
            lnt_list = np.unravel_index(np.argmax(weighted_summed_okd > ordered_knn_dens, axis=1), weighted_summed_okd.shape)[1]

            # 3b. Get distances to nearest neighbour of higher density
            if self.verbose:
                print("> Computing distance to higher density...")

            delta = [-1] * nb_el
            nneigh = [0] * nb_el
            cl_centers = []
            labels = [-1] * nb_el
            nb_clust = 0
            for x in range(0, nb_el):
                cur_idx_list = knn_idx[x]
                cur_dist_list = knn_dist[x]
                is_set = False
                for y in range(1, effective_nb_el):
                    # if y's density is higher, record corresponding distance as delta
                    if rho[cur_idx_list[y]] > rho[x]:
                        is_set = True
                        nneigh[x] = cur_idx_list[y]
                        delta[x] = cur_dist_list[y]

                        # If y is beyond lnt; x is a center
                        if y > lnt_list[x]:
                            cl_centers.append(x)
                            nb_clust += 1
                        break

                # If nb_el was reduced, the nearest neighbour of higher density may be beyond the neighbour matrix
                # x is assigned info corresponding to its last neighbour
                # also covers the point of highest density, which would be a cluster center by definition in this approach
                if not is_set:
                    nneigh[x] = cur_idx_list[y]
                    delta[x] = cur_dist_list[y]

                    # If y is beyond lnt; x is a center
                    if y > lnt_list[x]:
                        cl_centers.append(x)
                        nb_clust += 1

            # Check if same set of clusters was already found
            if cl_centers not in level_check:
                if self.verbose:
                    print("> Hierarchy is new ! %i clusters"%nb_clust)
                    print("  > Centers: %s"%cl_centers)
                level_check.append(deepcopy(cl_centers))
            else:
                if self.verbose:
                    print("> Repeated hierarchy; passing.")
                continue
                
            # Sort cluster centers according to local densities
            to_sort = []
            for x in range(nb_clust):
                to_sort.append([rho[cl_centers[x]], cl_centers[x]])
            to_sort = sorted(to_sort, key=operator.itemgetter(0))
            cl_centers = [el[1] for el in to_sort]
            for x in range(nb_clust):
                labels[cl_centers[x]] = x
            idx_rho_sorted = np.argsort(-np.array(rho))

            # Set highest density point delta to max delta
            delta[cl_centers[-1]] = np.amax(delta)
            
            ## 4. Cluster assignation
            # Assign points to the same cluster as its nearest neighbour of higher density
            # > do so in order of decreasing density
            if self.verbose:
                print("> Assigning points to clusters...")

            labels = np.array(labels)
            for x in range(nb_el):
                if labels[idx_rho_sorted[x]] == -1:
                    labels[idx_rho_sorted[x]] = labels[nneigh[idx_rho_sorted[x]]]

            ## 5. Outlier removal
            labels_all = labels

            # Recompute local densities by cluster. Then, remove points whose density is <X% of their center.
            # > Doing so avoids clusters of low densities to be classified as mainly outliers if they're close a high density cluster.
            if self.filt:
                if self.verbose:
                    print("> Filtering outliers...")
                for cl in range(len(cl_centers)):
                    # Get points of cluster + new id of center
                    cl_mask = np.where(labels == cl)[0]
                    center = cl_centers[cl]
                    idx_center = np.where(cl_mask==center)
                    cl_v = vectors[cl_mask]

                    # KNN only on cluster points
                    nbrs = NearestNeighbors(n_neighbors=len(cl_v), algorithm="auto", n_jobs=1, metric='l2').fit(cl_v)
                    cl_knn_dist, cl_knn_idx = nbrs.kneighbors(cl_v)
                    cl_ordered_knn_dens = np.exp(-1 * np.square(cl_knn_dist * inv_dc))

                    # Get local densities by summing over rows
                    cl_summed_okd = cl_ordered_knn_dens.cumsum(axis=1) - 1
                    cl_rho = cl_summed_okd[:, -1]
                    
                    # rho of center + threshold. 
                    d_thresh = np.amax(cl_rho) * self.filt
                    
                    # Unassign points with d < d_thresh
                    for i in range(len(cl_v)):
                        if cl_rho[i] < d_thresh:
                            labels[cl_mask[i]] = -1

            self.labels_.append(labels)
            self.labels_all_.append(labels_all)
            self.centers_.append(cl_centers)
            self.lnt_.append(lnt_list)
            self.rho_.append(rho)
            self.delta_.append(delta)
            self.gnt_.append(gnt)
            self.gnt_idx_.append(gnt_idx)
        
        self.neighbour_interval_ = self.n_int
        self.neigh_density_ = neighbour_density
        self.r2_ = r2
