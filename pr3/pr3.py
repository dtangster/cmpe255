#!/usr/bin/env python

import networkx
import numpy
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def load_csr_dataset(filename):
    ptr = [0]
    idx = []
    val = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        length = len(line)
        for i in range(1, length, 2):
            idx.append(line[i-1])
            val.append(line[i])
        ptr.append(ptr[-1] + length / 2)
    return csr_matrix((val, idx, ptr), dtype=numpy.double)


def svd(data, new_dimension):
    transformer = TruncatedSVD(n_components=new_dimension)
    return transformer.fit_transform(data)


def dbscan(clusters, min_pts, eps):
    neighborhoods = []
    core = []
    border = []
    noise = []

    # Treat the centroid of each cluster as a point
    points = clusters.cluster_centers_
    cos_sims = cosine_similarity(points)

    # Find core points
    for i in range(len(points)):
        neighbors = []
        for p in range(len(points)):
            # If the distance is below eps, p is a neighbor
            if cos_sims[i][p] >= eps:
                neighbors.append(p)
        neighborhoods.append(neighbors)
        # If neighborhood has at least min_pts, i is a core point
        if len(neighbors) >= min_pts:
            core.append(i)

    print("core: ", core, len(core))

    # Find border points
    for i in range(len(points)):
        neighbors = neighborhoods[i]
        # Look at points that are not core points
        if len(neighbors) < min_pts:
            for j in range(len(neighbors)):
                # If one of its neighbors is a core, it is also in the core point's neighborhood,
                # thus it is a border point rather than a noise point
                if neighbors[j] in core:
                    border.append(i)
                    # Need at least one core point...
                    break

    print("border: ", border, len(border))

    # Find noise points
    for i in range(len(points)):
        if i not in core and i not in border:
            noise.append(i)

    print("noise", noise, len(noise))

    nodes = core + border
    graph = networkx.Graph()
    graph.add_nodes_from(nodes)

    # Create neighborhood
    for i in range(len(nodes)):
        for p in range(len(nodes)):
            # If the distance is below the threshold, add a link in the graph.
            if p != i and cos_sims[i][p] >= eps:
                graph.add_edges_from([(nodes[i], nodes[p])])

    return list(networkx.connected_components(graph))


def output_results(clusters, clusters_refined, filename):
    outliers_cluster = len(clusters_refined) + 1
    with open(filename, 'w') as f:
        for c in clusters.labels_:
            found = False
            for i, cr in enumerate(clusters_refined):
                if c in cr:
                    f.write(str(i+1) + '\n')
                    found = True
                    continue
            if not found:
                f.write(str(outliers_cluster) + '\n')


csr = load_csr_dataset('train_pr3.dat')
print("Shape: ", csr.shape)
#csr = svd(csr, 500)
clusters = MiniBatchKMeans(n_clusters=300).fit(csr)
print("KMeans clusters:", clusters.labels_, len(clusters.labels_))
clusters_refined = dbscan(clusters, 5, 0.35)
print("# clusters:", len(clusters_refined))
print("clusters: ", clusters_refined)

output_results(clusters, clusters_refined, 'results.txt')
