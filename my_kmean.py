import numpy as np
import faiss

# centroids, code
def skl_kmeans(x, ncentroids=10, niter=300, n_init=10, mini=False, reassign=0.01):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    if x.shape[0] > 1000 or mini:
        model = MiniBatchKMeans(n_clusters=ncentroids, max_iter=niter, n_init=n_init, init='k-means++', random_state=3,
                                batch_size=4096, reassignment_ratio=reassign, max_no_improvement=20, tol=1e-7,
                                verbose=1)
    else:
        model = KMeans(n_clusters=ncentroids, max_iter=niter, n_init=n_init, init='k-means++', random_state=3, tol=1e-7,
                       verbose=1)
    model.fit(x)
    return model.cluster_centers_, model.labels_.tolist()


def constrained_km(data, n_clusters=512):
    from k_means_constrained import KMeansConstrained
    num_data_points = len(data)
    size_min = max(1, min(len(data) // (n_clusters * 2), n_clusters // 4))
    size_max = min(num_data_points, n_clusters * 2)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max, max_iter=10, n_init=10,
                            n_jobs=10, verbose=True)
    clf.fit(data)
    return clf.cluster_centers_, clf.labels_.tolist()


def kmeans(x, ncentroids=10, niter=100):
    verbose = True
    x = np.array(x, dtype=np.float32)
    d = x.shape[1]
    model = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    model.train(x)
    D, I = model.index.search(x, 1)
    code = [i[0] for i in I.tolist()]
    return model.centroids, code