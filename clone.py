import operator
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde


class CLoNe(object):
    '''
    CLoNe clustering

    Parameters
    ----------
    pdc: float, optional, default: 4
        The only input parameter.
        pdc is a percentage of all distances between neighbours.
        Used to extract the cut-off distance dc for the Gaussian kernel.
        dc is hence the distance at that percentile of the distances.

        Rule of thumb:
        > If you think there should be more clusters, decrease this value (integer increments are usually enough).
        > If you think there should be less clusters, increase this value.

    n_resize: int, optional, default: 4
        It decreases the number of neighbour computed for each point to N/n_resize.
        Only few neighbors are needed to compute what's needed by CLoNe.
        While perfectly optional, this can be used to decreases computing time and memory use.
        If set too high, some important points may not have the neighbours they need.

    filt: float, optional, default: 0.1
        Points whose local density is below 10% of the local density of their cluster center are classified as outliers.
        Setting it to 0 disables it (but disables removal of noise clusters as well)
        Change this value if you think it needs tuning.

    verbose: boolean, optional, default: False
        Enables verbose output of CLoNe


    Attributes
    ------------------------------
    labels_: array, shape (n_samples,)
        Labels of each point after outlier removal.

    labels_all_: array, shape (n_samples,)
        Labels of each point before outlier removal.

    centers_: array, shape (n, )
        Index of cluster centers.

    core_card_: array, (n_samples,)
        Core cardinalities of each data point.

    rho_: array, (n_samples,)
        Local densities of each point as computed by a Gaussian kernel


    Examples
    --------
    See Jupyter notebook "clone_examples" or "run_test.py".

    '''

    def __init__(self, pdc=4,
                 n_resize=4,
                 filt=0.1,
                 verbose=False):
        self.pdc = pdc
        self.n_resize = n_resize
        self.filt = filt
        self.verbose = verbose

    def fit(self, vectors):
        # 1. Build neighbour distance matrix
        # Reduce size of distance matrix to be computed
        # (reduces memory use + speeds things up)
        nb_el = len(vectors)
        effective_nb_el = int(nb_el / self.n_resize)

        if self.verbose:
            print("> Computing kNN...")

        # Use kNN to get neighbours + idx
        vectors = vectors.astype(np.float32)
        nbrs = NearestNeighbors(n_neighbors=effective_nb_el, algorithm="auto", n_jobs=1, metric='l2').fit(vectors)
        knn_dist, knn_idx = nbrs.kneighbors(vectors)
        knn_dist = knn_dist.astype(np.float32)
        knn_idx = knn_idx.astype(np.int32)

        # Get Gaussian sigma for local density based on neighbour matrix
        position = int(round(nb_el ** 2 * self.pdc * 0.01))
        if position >= np.prod(knn_dist.shape):
            position = -1
        dc = np.sort(knn_dist.flatten())[position]

        # 3. Compute local densities
        inv_dc = 1 / dc
        if self.verbose:
            print("> Computing local densities...")

        # Apply gaussian kernel on knn distances
        ordered_knn_dens = np.exp(-1 * np.square(knn_dist * inv_dc))

        # Get local densities by summing over rows
        summed_okd = ordered_knn_dens.cumsum(axis=1) - 1
        rho = summed_okd[:, -1].copy()

        # Weight contributions
        summed_okd /= int(round(self.pdc * 0.01 * nb_el))

        # Count contributing neighbours (= putative cluster cores)
        core_card = np.unravel_index(np.argmax(summed_okd > ordered_knn_dens, axis=1), summed_okd.shape)[1]

        # 4. Get real cluster centers
        if self.verbose:
            print("> Identifying real centers...")

        nneigh = [0] * nb_el  # Nearest neighbour of higher density; used for assignation
        centers = []
        labels = [-1] * nb_el
        nb_clust = 0
        for x in range(0, nb_el):
            cur_idx_list = knn_idx[x]
            is_set = False
            for y in range(1, effective_nb_el):
                # if y's density is higher, record it as nearest neigh of higher density
                if rho[cur_idx_list[y]] > rho[x]:
                    is_set = True
                    nneigh[x] = cur_idx_list[y]

                    # If y is not in the core of x, x is a center
                    if y > core_card[x] and core_card[x]:
                        centers.append(x)
                        nb_clust += 1
                    break

            # The point of highest density is a cluster center by definition
            # If by chance a few points have the same value that is the highest density, this takes care of it
            # (spiral quartet dataset is one such case)
            if not is_set:
                centers.append(x)
                nb_clust += 1

        if nb_clust == 1:
            print("Unique cluster found with set parameters.")
            self.labels_ = labels
            self.labels_all = labels
            self.centers = centers
            self.core_card = core_card
            self.rho = rho
            return

        # Sort cluster centers according to local densities
        to_sort = []
        for x in range(nb_clust):
            to_sort.append([rho[centers[x]], centers[x]])
        to_sort = sorted(to_sort, key=operator.itemgetter(0))
        centers = [el[1] for el in to_sort]
        for x in range(nb_clust):
            labels[centers[x]] = x
        idx_rho_sorted = np.argsort(-np.array(rho))

        # 4. Cluster assignation
        # Assign points to the same cluster as its nearest neighbour of higher density
        # > do so in order of decreasing density
        if self.verbose:
            print("> Assigning points to clusters...")

        labels = np.array(labels)
        for x in range(nb_el):
            if labels[idx_rho_sorted[x]] == -1:
                labels[idx_rho_sorted[x]] = labels[nneigh[idx_rho_sorted[x]]]

        # 5. Merging
        outliers = np.empty(0, dtype=int)
        core_points = np.empty(0, dtype=int)

        if self.verbose:
            print("> Merging centers...")

        # Get cluster cores
        core_dict = {}
        for cl in centers:
            lab = labels[cl]
            core_dict[lab] = []
            for j in knn_idx[cl, :core_card[cl]]:
                core_dict[lab].append(j)

        # ... border points
        border_points = {}
        for c1_idx in range(1, len(centers) - 1):
            c1 = centers[c1_idx]
            lab1 = labels[c1]
            mask1 = np.where(labels == lab1)[0]
            for p1 in mask1:
                for neigh in range(effective_nb_el):
                    if knn_dist[p1, neigh] > dc:
                        break
                    p2 = knn_idx[p1, neigh]
                    lab2 = labels[p2]
                    if lab2 == -1:
                        continue

                    if lab1 != lab2:
                        # Order labels by density to merge in order + avoid duplicate keys
                        if rho[c1] > rho[centers[lab2]]:
                            key = tuple((lab2, lab1))
                        else:
                            key = tuple((lab1, lab2))
                        if key not in border_points.keys():
                            border_points[key] = []
                        border_points[key].append(p1)
                        border_points[key].append(p2)

        # Merge based on Bhattacaryaa distance > 0.65
        cnt = 0
        for k in sorted(border_points.keys(), reverse=True):
            k0, k1 = k
            if labels[centers[k0]] == labels[centers[k1]]:
                continue
            border_points[k] = np.unique(border_points[k])
            mask0 = np.where(labels == labels[centers[k0]])[0]
            mask1 = np.where(labels == labels[centers[k1]])[0]
            try:
                kde_bord = gaussian_kde(core_card[border_points[k]])
                kde_core0 = gaussian_kde(core_card[core_dict[k0]])
                kde_core1 = gaussian_kde(core_card[core_dict[k1]])
            except Exception:
                continue
            mb = min(min(core_card[border_points[k]]), min(core_card[core_dict[k0]]), min(core_card[core_dict[k1]]))
            mc = max(max(core_card[border_points[k]]), max(core_card[core_dict[k0]]), max(core_card[core_dict[k1]]))

            support = np.linspace(mb, mc, 100)
            c2_0 = kde_core0.evaluate(support)
            c2_1 = kde_core1.evaluate(support)
            b2 = kde_bord.evaluate(support)

            c2_1 /= np.sum(c2_1)
            c2_0 /= np.sum(c2_0)
            b2 /= np.sum(b2)
            bc_0 = np.sum(np.sqrt(np.multiply(b2, c2_0)))
            bc_1 = np.sum(np.sqrt(np.multiply(b2, c2_1)))

            bc_01 = 0.5 * (bc_0 + bc_1)

            if bc_01 > 0.65:
                labels[mask0] = labels[centers[k1]]

        # Clean centers
        cur_lab = 0
        centers = []
        for ulab in np.unique(labels):
            if ulab == -1:
                continue
            mask_l = np.where(labels == ulab)[0]
            centers.append(mask_l[np.argmax(rho[mask_l])])
            labels[mask_l] = cur_lab
            cur_lab += 1

        # Get new core points
        for cl in centers:
            lab = labels[cl]
            for j in knn_idx[cl, :core_card[cl]]:
                core_points = np.append(core_points, j)

        # Removing outliers
        labels_all = labels.copy()
        outliers = np.array([], dtype=int)
        outlier_clusters = []
        if self.filt:
            if self.verbose:
                print("> Removing outliers...")

            # Recompute local densities and estimate noise
            # > recomputed to avoid bias from high density from other clusters if applicable
            for cl in centers:
                lab = labels[cl]
                mask = np.where(labels == lab)[0]

                # Isolate cluster + cut-off distance based on its neighborhood
                cl_v = vectors[mask]
                core_dc = knn_dist[cl, core_card[cl]]

                # KNN only on cluster points (reduce n_neighbors)
                nbrs = NearestNeighbors(n_neighbors=len(cl_v), algorithm="auto", n_jobs=1, metric='l2').fit(cl_v)
                cl_knn_dist, cl_knn_idx = nbrs.kneighbors(cl_v)

                # Gaussian based on core of center
                cl_ordered_knn_dens = np.exp(-1 * np.square(cl_knn_dist / core_dc))

                # Get local densities by summing over rows
                cl_summed_okd = cl_ordered_knn_dens.cumsum(axis=1) - 1
                cl_rho = cl_summed_okd[:, -1]

                # Set new densities and get noise approximation for this cluster
                rho[mask] = cl_rho
                rho_center = rho[cl]
                rho_thresh = self.filt * rho_center
                noise_mask = np.where(cl_rho < rho_thresh)[0]
                outliers = np.append(outliers, mask[noise_mask])
                labels[mask[noise_mask]] = -1

            # If enough noise, check for clusters from noise fluctuation and remove them
            # 2 is the lower limit for KDE to run...
            if len(outliers) > 2:
                # All core points
                for cl in centers:
                    lab = labels[cl]
                    cnt = 0
                    for j in knn_idx[cl, :core_card[cl]]:
                        cnt += 1
                        core_points = np.append(core_points, j)

                kde_noise = gaussian_kde(rho[outliers])
                kde_core = gaussian_kde(rho[core_points])

                mb = np.amin(rho)
                mc = np.amax(rho)
                support = np.linspace(mb, mc, 1000)

                log_pN = np.log(len(outliers) / len(vectors))
                log_pC = np.log(len(core_points) / len(vectors))

                for cl in centers:
                    lab = labels[cl]
                    mask_core = knn_idx[cl, :core_card[cl]]
                    rho_core = [rho[i] for i in mask_core if labels[i] == lab]
                    core_size = len(rho_core)
                    d_core = kde_core.evaluate(rho_core)
                    d_noise = kde_noise.evaluate(rho_core)

                    if len(d_noise[d_noise > 1e-16]) < core_size:
                        log_pN_pNX = np.NINF
                    else:
                        log_pN_pNX = np.sum(np.log(d_noise[d_noise > 1e-16])) + log_pN

                    if not len(d_core[d_core > 1e-16]):
                        log_pC_pCX = np.NINF
                    else:
                        log_pC_pCX = np.sum(np.log(d_core[d_core > 1e-16])) + log_pC

                    if log_pN_pNX > log_pC_pCX:
                        outlier_clusters.append(np.where(labels == lab)[0])
                        labels[labels == lab] = -1

                # Re-label after removing noise clusters
                init_c_idx = 0
                updated_centers = []
                for c in centers:
                    lab = labels[c]
                    if lab > -1:
                        labels_all[labels_all == lab] = init_c_idx
                        labels[labels == lab] = init_c_idx
                        init_c_idx += 1
                        updated_centers.append(c)
                centers = updated_centers

        self.rho = rho
        self.core_card = core_card
        self.centers = centers
        self.labels_ = labels
        self.labels_all = labels_all
