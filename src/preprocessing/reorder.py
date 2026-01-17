from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import numpy as np
import scanpy as sc

# --- Reorder theo covariance + hierarchical clustering ---
def reorder_gene_cov(Xmiss, method="average"):
    cov_mat = np.cov(Xmiss, rowvar=False)

    # Similarity normalization
    sim = np.asarray(cov_mat, dtype=float)
    smin, smax = sim.min(), sim.max()
    if smax > smin:
        sim = (sim - smin) / (smax - smin)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    order = leaves_list(Z)

    return  Xmiss[:, order], order



