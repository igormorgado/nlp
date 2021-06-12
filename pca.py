#!/usr/bin/env python
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    """
    M = X - X.mean(axis=0)
    covariance_matrix = np.cov(M, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)
    idx_sorted = np.argsort(np.abs(eigen_vals))
    idx_sorted_decreasing = idx_sorted[::-1]
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
    eigen_vecs_sorted = eigen_vecs.T[idx_sorted_decreasing]
    eigen_vecs_subset = eigen_vecs_sorted[:n_components]
    X_reduced = np.real(np.dot(X, eigen_vecs_subset.T))
    return X_reduced

if __name__ == "__main__":
    #%% Testing your function
    np.random.seed(1)
    X = np.random.rand(3, 10)
    X_reduced = compute_pca(X, n_components=2)
    print("Your original matrix was " + str(X.shape) + " and it became:")
    print(X_reduced)

    #%%
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(X_reduced[:,0], X_reduced[:,1])
    fig.show()
