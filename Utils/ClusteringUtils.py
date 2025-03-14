# Standard Library Imports
import os
import sys
from collections import defaultdict

# Environment Configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "3"

# Third-Party Library Imports
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import mixture
import numpy as np

# Local Imports
from . import VisualizationUtils as VU
from . import ElbowUtils as EU

# --------------------------------------------
# User Input Functions
# --------------------------------------------

def input_for_number_of_clusters_method(clustering_method):
    """
    Prompts user to select a method for determining the number of clusters.

    Returns:
        str: The selected method ('auto', 'partial', or 'defined').
    """

    if clustering_method != 'Uniform binning':
        method_mapping = {'1': 'auto', '2': 'partial', '3': 'defined'}
        valid_options = list(method_mapping.values()) + list(method_mapping.keys())

        while True:
            method = input("Select clustering method: auto [1] / partial [2] / defined [3]: ").strip()
            if method in valid_options:
                return method_mapping.get(method, method)  # Convert numeric input to string equivalent
            print(f"Invalid input: {method}. Choose from {', '.join(valid_options)}.")
    else:
        return 'defined'

def input_for_clustering_method():
    """
    Prompts user to select a clustering method.

    Returns:
        str: The selected clustering method ('kmeans' or 'gmm').
    """
    method_mapping = {'1': 'kmeans', '2': 'gmm', '3': 'Uniform binning'}
    valid_options = list(method_mapping.values()) + list(method_mapping.keys())

    while True:
        method = input("Select clustering method: kmeans [1] / gmm [2] / Uniform binning [3] ").strip()
        if method in valid_options:
            return method_mapping.get(method, method)  # Convert numeric input to string equivalent
        print(f"Invalid input: {method}. Choose from {', '.join(valid_options)}.")

# --------------------------------------------
# Clustering Functions
# --------------------------------------------

def get_number_of_clusters(method, xy, cluster_method, plot=False, save=False):
    """
    Determines the number of clusters based on the selected method.

    Args:
        method (str): The method for determining the number of clusters ('auto', 'partial', 'defined').
        xy (np.ndarray): Data points for clustering.
        cluster_method (str): Clustering method ('kmeans' or 'gmm').
        plot (bool): Whether to display the elbow plot.
        save (bool): Whether to save the elbow plot.

    Returns:
        int: The determined number of clusters.
    """
    if method not in ['auto', 'partial', 'defined']:
        raise ValueError(f"Invalid clustering method '{method}'.")

    if method in ['auto', 'partial']:
        cluster_metrics_dict = defaultdict(float)

        for n_clusters in range(2, 20):
            _, _, metric = get_clusters(xy, n_clusters, cluster_method)
            cluster_metrics_dict[n_clusters] = metric

        if method == 'auto':
            elbow_index = EU.find_elbow(list(cluster_metrics_dict.keys()), list(cluster_metrics_dict.values()))
            n_clusters = list(cluster_metrics_dict.keys())[elbow_index]

            VU.save_or_plot(
                lambda: (
                    plt.plot(list(cluster_metrics_dict.keys()), list(cluster_metrics_dict.values()), label="Cluster Metrics"),
                    plt.scatter(n_clusters, list(cluster_metrics_dict.values())[elbow_index], c='r', label="Elbow Point"),
                    plt.xlabel("Number of Clusters"),
                    plt.ylabel("Inertia" if cluster_method == 'kmeans' else "BIC"),
                    plt.title("Elbow Point"),
                    plt.grid(),
                    plt.legend()
                ),
                save_path="ElbowPoint.pdf" if save else None,
                plotting_graphs=plot
            )

            return n_clusters

        elif method == 'partial':
            VU.save_or_plot(
                lambda: (
                    plt.plot(list(cluster_metrics_dict.keys()), list(cluster_metrics_dict.values())),
                    plt.xlabel("Number of Clusters"),
                    plt.ylabel("Inertia" if cluster_method == 'kmeans' else "BIC"),
                    plt.title("Cluster Metrics"),
                    plt.grid()
                ),
                save_path="ClusterMetrics.pdf" if save else None,
                plotting_graphs=plot
            )

            while True:
                try:
                    return int(input("Enter the number of clusters: "))
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")

    elif method == 'defined':
        while True:
            try:
                return int(input("Enter the number of clusters: "))
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

def get_clusters(xy, n_clusters, method='gmm'):
    """
    Performs clustering using the specified method.

    Args:
        xy (np.ndarray): Data points for clustering.
        n_clusters (int): Number of clusters.
        method (str): Clustering method ('kmeans' or 'gmm').

    Returns:
        tuple: (Cluster labels, cluster centers, cluster metric)
    """
    if method not in ['kmeans', 'gmm', 'Uniform binning']:
        raise ValueError(f"Invalid clustering method '{method}'.")
    
    if method == 'Uniform binning':
        return get_uniform_binning_clusters(xy, n_clusters)

    return get_kmeans_clusters(xy, n_clusters) if method == 'kmeans' else get_gmm_clusters(xy, n_clusters)

def get_kmeans_clusters(xy, n_clusters):
    """
    Performs KMeans clustering.

    Args:
        xy (np.ndarray): Data points for clustering.
        n_clusters (int): Number of clusters.

    Returns:
        tuple: (Cluster labels, cluster centers, inertia metric)
    """
    kmeans = KMeans(n_clusters=n_clusters, verbose=0)
    kmeans.fit(xy)
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_

def get_gmm_clusters(xy, n_clusters):
    """
    Performs Gaussian Mixture Model (GMM) clustering.

    Args:
        xy (np.ndarray): Data points for clustering.
        n_clusters (int): Number of clusters.

    Returns:
        tuple: (Cluster labels, cluster centers, BIC metric)
    """
    gmm = mixture.GaussianMixture(n_components=n_clusters).fit(xy)
    metrics = gmm.bic(xy)
    labels = gmm.predict(xy)
    means = gmm.means_

    return labels, means, metrics

import numpy as np

def get_uniform_binning_clusters(xy, n_clusters,method='exact'):
    """
    Performs uniform binning clustering.
    
    This function divides the bounding box of the data uniformly into a grid with exactly
    n_clusters bins and assigns each point to the corresponding bin. For bins with no points,
    the bin center is used as the cluster center.
    
    Args:
        xy (np.ndarray): Data points for clustering with shape (n_samples, 2).
        n_clusters (int): Desired number of clusters.
    
    Returns:
        tuple: (cluster_labels, cluster_centers, metric)
            - cluster_labels (np.ndarray): An integer array of shape (n_samples,)
              indicating the cluster (bin) index for each point.
            - cluster_centers (np.ndarray): A (n_clusters, 2) array of cluster centers.
              For bins with no points, the bin center is used.
            - metric (float): Sum of squared distances from each point to its cluster center.
    """
    def get_grid_dims(n_clusters):
        """
        Returns a tuple (nx, ny) such that nx*ny == n_clusters.
        It searches for factors of n_clusters starting from floor(sqrt(n_clusters)) downwards.
        If no factors are found (i.e. n_clusters is prime), returns (1, n_clusters).
        """
        # Try to factor n_clusters into two factors that are as close as possible.
        for i in range(int(np.floor(np.sqrt(n_clusters))), 0, -1):
            if n_clusters % i == 0:
                return i, n_clusters // i
        return 1, n_clusters  # fallback if n_clusters is prime

    assert method in ['approximate', 'exact'], "method must be 'approximate' or 'exact'"

    # Ensure xy has shape (n_samples, 2)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be a 2D array with shape (n_samples, 2)")
    
    n_samples = xy.shape[0]

    if method == 'exact':
        # Determine grid dimensions such that nx*ny == n_clusters.
        nx, ny = get_grid_dims(n_clusters)
        total_bins = nx * ny
    
    elif method == 'approximate':
        # Determine grid dimensions:
        nx = int(np.ceil(np.sqrt(n_clusters)))
        ny = int(np.ceil(n_clusters / nx))
        total_bins = nx * ny

    # Compute the bounding box of the data.
    x_min, x_max = np.min(xy[:, 0]), np.max(xy[:, 0])
    y_min, y_max = np.min(xy[:, 1]), np.max(xy[:, 1])
    
    # Create uniform bin edges in x and y.
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    
    # For each point, find the corresponding bin index.
    # np.digitize returns indices in 1...len(bins), so subtract 1 for 0-based indexing.
    x_bin_indices = np.digitize(xy[:, 0], x_edges) - 1
    y_bin_indices = np.digitize(xy[:, 1], y_edges) - 1
    
    # Points that lie exactly on the right (or top) edge might get index == nx or ny; clip these.
    x_bin_indices = np.clip(x_bin_indices, 0, nx - 1)
    y_bin_indices = np.clip(y_bin_indices, 0, ny - 1)
    
    # Compute a single cluster label for each point.
    # Here we use label = (x_index * ny) + y_index so that labels run from 0 to (n_clusters - 1).
    labels = x_bin_indices * ny + y_bin_indices

    # Compute cluster centers and the sum-of-squared distances metric.
    cluster_centers = np.zeros((total_bins, 2))
    total_ssd = 0  # Sum of squared distances.
    
    for label in range(total_bins):
        mask = (labels == label)
        if np.any(mask):
            cluster_points = xy[mask]
            center = cluster_points.mean(axis=0)
            cluster_centers[label] = center
            # Compute sum of squared distances for points in this cluster.
            ssd = np.sum((cluster_points - center)**2)
            total_ssd += ssd
        else:
            # If no points fell into this bin, we assign the bin center.
            # Determine bin indices in the grid corresponding to this label.
            i = label // ny  # x bin index
            j = label % ny   # y bin index
            center_x = (x_edges[i] + x_edges[i+1]) / 2
            center_y = (y_edges[j] + y_edges[j+1]) / 2
            cluster_centers[label] = np.array([center_x, center_y])
    
    return labels, cluster_centers, total_ssd

'''def get_uniform_binning_clusters(xy, n_clusters):
    """
    Performs uniform binning clustering.
    
    This function divides the bounding box of the data uniformly into a grid and 
    assigns each point to the corresponding bin. The grid dimensions (number of bins
    in x and y) are chosen so that the total number of bins is at least n_clusters.
    
    Args:
        xy (np.ndarray): Data points for clustering with shape (n_samples, 2).
        n_clusters (int): Desired (approximate) number of clusters.
    
    Returns:
        tuple: (cluster_labels, cluster_centers, metric)
            - cluster_labels (np.ndarray): An integer array of shape (n_samples,)
              indicating the cluster (bin) index for each point.
            - cluster_centers (np.ndarray): A (total_bins, 2) array of cluster centers.
              For bins with no points, the bin center is used.
            - metric (float): Sum of squared distances from each point to its cluster center.
    """
    # Ensure xy has shape (n_samples, 2)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be a 2D array with shape (n_samples, 2)")
    
    n_samples = xy.shape[0]
    
    # Determine grid dimensions:
    # Let nx be the number of bins along the x-dimension and ny along the y-dimension.
    # A common approach is to set nx = ceil(sqrt(n_clusters)) and ny = ceil(n_clusters / nx).
    nx = int(np.ceil(np.sqrt(n_clusters)))
    ny = int(np.ceil(n_clusters / nx))
    total_bins = nx * ny  # This is the actual number of bins/clusters.
    
    # Compute the bounding box of the data.
    x_min, x_max = np.min(xy[:, 0]), np.max(xy[:, 0])
    y_min, y_max = np.min(xy[:, 1]), np.max(xy[:, 1])
    
    # Create uniform bin edges in x and y.
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)
    
    # For each point, find the corresponding bin index.
    # np.digitize returns indices in 1...len(bins), so subtract 1 for 0-based indexing.
    x_bin_indices = np.digitize(xy[:, 0], x_edges) - 1
    y_bin_indices = np.digitize(xy[:, 1], y_edges) - 1
    
    # Points that lie exactly on the right edge might get index == nx or ny; clip these.
    x_bin_indices = np.clip(x_bin_indices, 0, nx - 1)
    y_bin_indices = np.clip(y_bin_indices, 0, ny - 1)
    
    # Compute a single cluster label for each point.
    # Here we use label = i * ny + j so that labels run from 0 to (nx*ny - 1).
    labels = x_bin_indices * ny + y_bin_indices

    # Compute cluster centers and the sum-of-squared distances metric.
    cluster_centers = np.zeros((total_bins, 2))
    total_ssd = 0  # Sum of squared distances.
    
    for label in range(total_bins):
        mask = (labels == label)
        if np.any(mask):
            cluster_points = xy[mask]
            center = cluster_points.mean(axis=0)
            cluster_centers[label] = center
            # Compute sum of squared distances for points in this cluster.
            ssd = np.sum((cluster_points - center)**2)
            total_ssd += ssd
        else:
            # If no points fell into this bin, we assign the bin center.
            # Determine bin indices in the grid corresponding to this label.
            i = label // ny  # x bin index
            j = label % ny   # y bin index
            center_x = (x_edges[i] + x_edges[i+1]) / 2
            center_y = (y_edges[j] + y_edges[j+1]) / 2
            cluster_centers[label] = np.array([center_x, center_y])
    
    return labels, cluster_centers, total_ssd'''
