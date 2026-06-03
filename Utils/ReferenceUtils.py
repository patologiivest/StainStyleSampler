# Standard Library Imports
import random
from collections import defaultdict

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from sklearn.metrics import pairwise_distances
from PIL import Image
from matplotlib import colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm


# Local Imports
from . import VisualizationUtils as VU
from . import StainUtils as SU


def input_for_number_of_images() -> int:
    """
    Prompts the user to enter the number of images to select.
    Ensures input is a valid positive integer.

    Returns:
        int: The number of images to select.
    """
    while True:
        try:
            n_images = int(input("Enter the number of images to select: "))
            if n_images > 0:
                return n_images
            else:
                print("Error: Please enter a positive integer.")
        except ValueError:
            print("Error: Invalid input. Please enter a valid integer.")

def label_full_dataset(hist,hist_centers,embedding,non_empty_bin_coordinates,true_labels):
    """
    Meant to be run after the clustering and reference images were chosen
    
    Returns:
      List of labels, same length as embedding.
    """

    _, x_edges, y_edges = hist[0], hist[1], hist[2]
    
    hist_centers_x = hist_centers[:,0]
    hist_centers_y = hist_centers[:,1]
    
    hitx = np.digitize(embedding[:, 0], x_edges)-1
    hity = np.digitize(embedding[:, 1], y_edges)-1
    
    # Make sure indices are valid
    mask = (
        (hitx >= 0) & (hitx < len(hist_centers_x)) &
        (hity >= 0) & (hity < len(hist_centers_y))
    )
    hitx = hitx[mask]
    hity = hity[mask]

    bin_center_x = hist_centers_x[hitx]
    bin_center_y = hist_centers_y[hity]

    indexes_label = []
    for i in range(len(list(zip(bin_center_x, bin_center_y)))):
        point = (bin_center_x[i], bin_center_y[i])  
        distances = np.linalg.norm(non_empty_bin_coordinates - point, axis=1)
        min_idx = np.argmin(distances)
        indexes_label.append(min_idx)

    calculated_labels = []
    for idx in indexes_label:
        calculated_labels.append(true_labels[idx])
    
    return calculated_labels

#----------------------------------------------------------------------------
def get_references(
    embedding: np.ndarray, colors: np.ndarray, xy: np.ndarray, cluster_labels: np.ndarray = None, 
    cluster_centers: np.ndarray = None, images: np.ndarray = None, h: np.ndarray = None, 
    n_images: int = None, reference_mode: str = 'representative',color_mode: str = 'lab',embedding_name: str = 'UMAP',
    density_selection_mode: str = None, density_percentile_level: int = 2, plot: bool = False, save: bool = False) -> tuple:
    """
    Selects reference images based on different modes: 'random', 'representative', or 'density'.

    Args:
        embedding (np.ndarray): The 2D embedding coordinates (from UMAP or PCA).
        colors (np.ndarray): Color values associated with each data point.
        xy (np.ndarray): Original XY coordinates of each image in the embedding space.
        cluster_labels (np.ndarray, optional): Labels of the clusters. Required for 'representative' and modes.
        cluster_centers (np.ndarray, optional): Cluster center coordinates. Required for 'representative' mode.
        images (np.ndarray, optional): List of image file paths.
        h (np.ndarray, optional): Histogram data for mode.
        n_images (int, optional): Number of images to select in 'random' mode.
        reference_mode (str): Mode for selecting reference images. One of ['random', 'representative'].
        plot (bool): Whether to plot the selected reference images. Default is False.
        save (bool): Whether to save results. Default is False.

    Returns:
        tuple: Reference images and, if applicable, cluster statistics.
    """
    valid_modes = ['random', 'representative','density']
    if reference_mode not in valid_modes:
        raise ValueError(f"Invalid reference mode '{reference_mode}'. Choose from {valid_modes}")

    if reference_mode == 'random':
        return __get_random_references__(embedding, colors, xy, images, n_images, plot, save)

    elif reference_mode == 'representative':
        return __get_representative_references__(embedding, colors, xy, cluster_labels, cluster_centers, images, plot, save)
    
    elif reference_mode == 'density':
        return __get_density_references__(embedding,h, colors, embedding_name, images,plot=plot, save=save,n_images=n_images,selection_mode=density_selection_mode,density_percentile_level=density_percentile_level)
    return None

#----------------------------------------------------------------------------
def __get_random_references__(embedding, colors, xy, images, n_images, plot, save):
    """Handles random selection of reference images."""
    if n_images is None:
        raise ValueError("Number of images must be specified for random selection.")

    targets = np.random.choice(range(len(images)), n_images, replace=False)
    reference_files = images[targets]

    VU.save_or_plot(
        lambda: (
            plt.scatter(xy[:, 0], xy[:, 1], c='lightgrey'),
            plt.scatter(embedding[targets, 0], embedding[targets, 1], c=colors[targets], s=200, edgecolors="black"),
            plt.axis('off')
        ),
        save_path="Targets.pdf" if save else None,
        plotting_graphs=plot
    )

    VU.display_reference_images(
        reference_files,
        save_path="References.pdf" if save else None,
        plotting_graphs=plot
    )

    return reference_files
#----------------------------------------------------------------------------

def __get_representative_references__(embedding, colors, xy, cluster_labels, cluster_centers, images, plot, save):
    """Handles selection of representative images based on cluster centers."""
    if cluster_labels is None or cluster_centers is None:
        raise ValueError("Cluster labels and centers must be provided for representative reference selection.")

    def get_targets(centers: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """
        Finds the closest points in `xy` to each cluster center in `centers`.

        Args:
            centers (np.ndarray): Array of cluster center coordinates (shape: [n_clusters, 2]).
            xy (np.ndarray): Array of all data point coordinates (shape: [n_samples, 2]).

        Returns:
            np.ndarray: Indices of the closest points in `xy` to each center.
        """
        if centers is None or xy is None:
            raise ValueError("Both `centers` and `xy` must be provided.")
        
        if centers.shape[1] != xy.shape[1]:
            raise ValueError(f"Dimension mismatch: `centers` has shape {centers.shape}, but `xy` has shape {xy.shape}.")

        return np.argmin(pairwise_distances(centers, xy), axis=-1)
    
    
    targets = get_targets(cluster_centers, embedding[:, :2])

    VU.save_or_plot(
        lambda: (
            plt.scatter(xy[:, 0], xy[:, 1], c='lightgrey'),
            plt.scatter(embedding[targets, 0], embedding[targets, 1], c=colors[targets], s=200, edgecolors="black"),
            plt.axis('off')
        ),
        save_path="Clusters.pdf" if save else None,
        plotting_graphs=plot
    )

    reference_files = images[targets]

    VU.display_reference_images(
        reference_files,
        save_path="References.pdf" if save else None,
        plotting_graphs=plot
    )

    if save:
        pd.DataFrame(reference_files).to_csv("ReferenceFiles.csv", index=False)

    return reference_files
#----------------------------------------------------------------------------
def __get_density_references__(
    embedding, histogram, colors, embedding_name, images,
    plot, save, n_images=1, density_percentile_level=2,
    number_of_bins=100, selection_mode="original"
):
    """
    Select reference images based on density contours in 2D embedding space.

    selection_mode:
      - "original": Use all candidates from the highest-density region.
      - "sorted_regions": Try to select one from each region, then fill as needed.
    """

    # --- Helper: Density Map ---
    def get_density_map(embedding,histogram, number_of_bins):
        pad = 1
        x, y = embedding[:, 0], embedding[:, 1]
        hist, x_edges, y_edges = histogram[0],histogram[1],histogram[2]
        hist = gaussian_filter(hist, sigma=2.5)
        hist = np.pad(hist, pad, mode='constant', constant_values=0)

        x_bin_width = np.diff(x_edges[:2])[0]
        y_bin_width = np.diff(y_edges[:2])[0]
        x_edges = np.linspace(x_edges[0] - pad * x_bin_width,
                              x_edges[-1] + pad * x_bin_width,
                              hist.shape[1] + 1)
        y_edges = np.linspace(y_edges[0] - pad * y_bin_width,
                              y_edges[-1] + pad * y_bin_width,
                              hist.shape[0] + 1)

        bin_area = x_bin_width * y_bin_width
        density = (hist / np.sum(hist)) * bin_area
        density = density / np.max(density)

        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

        percentiles = [90, 95, 99]
        density_levels = [np.percentile(density, p) for p in percentiles]
        density_levels = sorted(density_levels)
        contour_colors = [cm.viridis(i / len(density_levels)) for i in range(len(density_levels))]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_gradient", contour_colors, N=256)
        norm = mcolors.Normalize(vmin=density_levels[0], vmax=density_levels[-1])

        return (hist, X, Y, x_edges, y_edges, density, density_levels,
                contour_colors, custom_cmap, norm, x_centers, y_centers, percentiles)

    # --- Helper: Get bin centers ---
    def get_bin_centers(x_edges, y_edges):
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X_grid, Y_grid = np.meshgrid(x_centers, y_centers)
        return np.vstack([X_grid.flatten(), Y_grid.flatten()]).T

    # --- Helper: Select candidates per region ---
    def select_points_from_regions(bin_points, paths):
        selected_points = []
        region_candidates = []
        for path in paths:
            inside = [pt for pt in bin_points if path.contains_point(pt, radius=-1e-3)]
            if inside:
                region_candidates.append(np.array(inside))
        # Pick a farthest-from-centroid point from each region
        for candidates in region_candidates:
            centroid = candidates.mean(axis=0)
            dists = np.linalg.norm(candidates - centroid, axis=1)
            selected_points.append(candidates[np.argmax(dists)])
        return selected_points, region_candidates

    # --- Helper: Original mode (use one region only) ---
    def select_points_from_first_region(bin_points, paths, n_images):
        for path in paths:
            inside = [pt for pt in bin_points if path.contains_point(pt, radius=-1e-3)]
            if inside:
                inside = np.array(inside)
                # Optionally, pick densest/farthest etc. For now: farthest sampling for spread.
                # If less than n_images, just take all.
                if len(inside) >= n_images:
                    selected = [inside[0]]
                    for _ in range(n_images - 1):
                        # Greedily pick the point farthest from those already selected
                        dists = np.min(np.linalg.norm(inside - selected[-1], axis=1))
                        candidate = inside[np.argmax(dists)]
                        selected.append(candidate)
                    return selected
                else:
                    return inside.tolist()
        return []

    # --- Helper: Add more points if needed, avoid duplicates ---
    def fill_extra_points(region_candidates, already_selected, n_needed):
        all_candidates = np.vstack(region_candidates) if region_candidates else np.array([])
        # Remove already selected
        extras = []
        for pt in all_candidates:
            if not any(np.allclose(pt, sel) for sel in already_selected):
                extras.append(pt)
        if len(extras) > n_needed:
            chosen = random.sample(extras, n_needed)
        else:
            chosen = extras
        return chosen

    # --- Helper: Map bin point to nearest embedding (unique indices) ---
    def map_bin_points_to_embedding(selected_bin_points, embedding):
        mapped_indices = []
        used_indices = set()
        for pt in selected_bin_points:
            dists = np.linalg.norm(embedding - pt, axis=1)
            sorted_idx = np.argsort(dists)
            for idx in sorted_idx:
                if idx not in used_indices:
                    mapped_indices.append(idx)
                    used_indices.add(idx)
                    break
        return mapped_indices

    # === MAIN EXECUTION ===
    (hist, X, Y, x_edges, y_edges, density, density_levels,
     contour_colors, custom_cmap, norm, x_centers, y_centers, percentiles) = get_density_map(
        embedding=embedding,histogram=histogram, number_of_bins=number_of_bins)

    fig_temp, ax_temp = plt.subplots(figsize=(8, 6))
    contour_set = ax_temp.contour(X, Y, density.T, levels=density_levels, colors=contour_colors)
    plt.close(fig_temp)

    desired_index = density_percentile_level
    paths = contour_set.collections[desired_index].get_paths()
    bin_points = get_bin_centers(x_edges, y_edges)

    if selection_mode == "sorted_regions":
        # 1. Pick one per region
        per_region_points, region_candidates = select_points_from_regions(bin_points, paths)
        n_regions = len(per_region_points)
        # 2. Fill more if needed
        if n_images > n_regions:
            n_extra = n_images - n_regions
            extra_points = fill_extra_points(region_candidates, per_region_points, n_extra)
            selected_bin_points = per_region_points + extra_points
        else:
            selected_bin_points = random.sample(per_region_points, min(n_images, n_regions))
    elif selection_mode == "original":
        selected_bin_points = select_points_from_first_region(bin_points, paths, n_images)
    else:
        raise ValueError(f"Invalid selection_mode: {selection_mode}")

    # 4. Map selected bin points to nearest embedding points (no duplicates)
    unique_indices = map_bin_points_to_embedding(selected_bin_points, embedding)
    # Optionally: fill up if not enough unique indices
    if len(unique_indices) < n_images:
        all_indices = set(range(len(embedding)))
        remaining = list(all_indices - set(unique_indices))
        if remaining:
            far_points = []
            remaining_indices = list(remaining)
            for _ in range(min(n_images - len(unique_indices), len(remaining_indices))):
                selected = embedding[unique_indices + far_points]
                candidates = embedding[remaining_indices]
                distances = np.linalg.norm(candidates[:, None, :] - selected[None, :, :], axis=2)
                candidate_position = int(np.argmax(np.min(distances, axis=1)))
                far_points.append(remaining_indices.pop(candidate_position))
            unique_indices.extend(far_points)

    targets = np.array(unique_indices[:n_images])
    reference_files = images[targets]

    # --- Visualization ---
    if plot or save:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=100, c=colors, label=f'{embedding_name} Points')
        contour_set = ax.contour(X, Y, density.T, levels=density_levels, colors=contour_colors)
        ax.scatter(embedding[targets, 0], embedding[targets, 1], marker='x', s=200,
                   c='red', linewidths=2, label='Reference Points')
        ax.set_title(f"{embedding_name} with Density Contours (Percentile Levels)")
        ax.axis('off')
        cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
        cb = ColorbarBase(cax, cmap=custom_cmap, norm=norm, orientation='vertical')
        cb.set_ticks(density_levels)
        cb.set_ticklabels([f'{100 - p}%' for p in percentiles])
        if save:
            plt.savefig(f"{embedding_name}_DensityReferences.pdf", bbox_inches="tight")
        if plot:
            plt.show()
        plt.close(fig)

    VU.save_or_plot(
        lambda: (
            plt.scatter(embedding[:, 0], embedding[:, 1], c='lightgrey'),
            plt.scatter(embedding[targets, 0], embedding[targets, 1], c=colors[targets], s=200, edgecolors="black"),
            plt.axis('off')
        ),
        save_path="DensityTargets.pdf" if save else None,
        plotting_graphs=plot
    )
    
    reference_files = images[targets]
    
    VU.display_reference_images(
        reference_files,
        save_path="DensityReferences.pdf" if save else None,
        plotting_graphs=plot
    )

    return reference_files, contour_set





