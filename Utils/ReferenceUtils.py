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

#----------------------------------------------------------------------------
def get_references(
    embedding: np.ndarray, colors: np.ndarray, xy: np.ndarray, cluster_labels: np.ndarray = None, 
    cluster_centers: np.ndarray = None, images: np.ndarray = None, h: np.ndarray = None, 
    n_images: int = None, reference_mode: str = 'representative',color_mode: str = 'lab',embedding_name: str = 'UMAP',
    density_selection_mode: str = None, density_percentile_level: int = 2, plot: bool = False, save: bool = False) -> tuple:
    """
    Selects reference images based on different modes: 'random', 'representative', 'grouped' or 'density'.

    Args:
        embedding (np.ndarray): The 2D embedding coordinates (from UMAP or PCA).
        colors (np.ndarray): Color values associated with each data point.
        xy (np.ndarray): Original XY coordinates of each image in the embedding space.
        cluster_labels (np.ndarray, optional): Labels of the clusters. Required for 'representative' and 'grouped' modes.
        cluster_centers (np.ndarray, optional): Cluster center coordinates. Required for 'representative' mode.
        images (np.ndarray, optional): List of image file paths.
        h (np.ndarray, optional): Histogram data for 'grouped' mode.
        n_images (int, optional): Number of images to select in 'random' mode.
        reference_mode (str): Mode for selecting reference images. One of ['random', 'representative', 'grouped'].
        plot (bool): Whether to plot the selected reference images. Default is False.
        save (bool): Whether to save results. Default is False.

    Returns:
        tuple: Reference images and, if applicable, cluster statistics.
    """
    valid_modes = ['random', 'representative', 'grouped','density']
    if reference_mode not in valid_modes:
        raise ValueError(f"Invalid reference mode '{reference_mode}'. Choose from {valid_modes}")

    if reference_mode == 'random':
        return __get_random_references__(embedding, colors, xy, images, n_images, plot, save)

    elif reference_mode == 'representative':
        return __get_representative_references__(embedding, colors, xy, cluster_labels, cluster_centers, images, plot, save)
    
    elif reference_mode == 'grouped':
        return __get_grouped_references__(embedding, xy, cluster_labels, cluster_centers, images, h, plot, save, color_mode)
    
    elif reference_mode == 'density':
        return __get_density_references__(embedding, colors, embedding_name, images,plot=plot, save=save,n_images=n_images,selection_mode=density_selection_mode,density_percentile_level=density_percentile_level)
        #embedding,colors,embedding_name,images,n_images, plot=plot, save=save
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

def __get_grouped_references__(embedding, xy, cluster_labels, cluster_centers, images, h, plot, save,color_mode):
    """Handles selection of grouped references using clustering and histogram data."""
    if cluster_labels is None or cluster_centers is None:
        raise ValueError("Cluster labels and centers must be provided for grouped reference selection.")
    if h is None:
        raise ValueError("Histogram data must be provided for grouped reference selection.")

    def find_bin_for_umap_point(point, hist):
        """
        Finds the corresponding bin for a given UMAP point in a histogram.

        Args:
            point (tuple or np.ndarray): The (x, y) coordinates of the UMAP point.
            hist (tuple): Histogram data, where hist[1] contains x bin edges and hist[2] contains y bin edges.

        Returns:
            tuple or None: (x_bin, y_bin) if the point falls within the histogram bins, else None.
        """
        if not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2:
            raise ValueError("Invalid input: 'point' must be a tuple or array with two elements (x, y).")

        if len(hist) < 3:
            raise ValueError("Invalid histogram data: Must contain at least three elements (counts, xedges, yedges).")

        xedges, yedges = hist[1], hist[2]

        # Get bin indices
        x_bin = np.digitize([point[0]], xedges, right=True)[0] - 1
        y_bin = np.digitize([point[1]], yedges, right=True)[0] - 1

        # Ensure bin indices are within valid range
        if 0 <= x_bin < len(xedges) - 1 and 0 <= y_bin < len(yedges) - 1:
            return x_bin, y_bin

        return None  # Point is out of bounds
    
    def build_image_to_bin_cluster_dict(images, umap_data, histogram, bin_coordinates, labels):
        """
        Build a dictionary mapping each image to its bin and cluster.

        Args:
            images (list): List of image file names or references.
            umap_data (np.ndarray): UMAP coordinates of all images (shape: [n_samples, 2]).
            histogram (tuple): The output of np.histogram2d (includes bin edges).
            bin_coordinates (np.ndarray): Coordinates of bins.
            labels (np.ndarray): Cluster labels for each bin.

        Returns:
            dict: A dictionary where keys are image names and values are dicts with 'bin' and 'cluster'.
        """
        if len(images) == 0 or images is None or not umap_data.any() or not histogram or not labels.any():
            raise ValueError("Invalid inputs: Ensure all required parameters contain valid data.")

        if len(images) != len(umap_data):
            raise ValueError("Mismatch: `images` and `umap_data` must have the same length.")

        # Map bin index to cluster
        bin_to_cluster = {tuple(coord): labels[i] for i, coord in enumerate(bin_coordinates)}

        # Calculate bin center coordinates
        x_centers = (histogram[1][:-1] + histogram[1][1:]) / 2
        y_centers = (histogram[2][:-1] + histogram[2][1:]) / 2

        image_to_bin_cluster = {}

        for i, point in enumerate(umap_data):
            bin_index = find_bin_for_umap_point(point, histogram)

            if bin_index:
                bin_coord = (x_centers[bin_index[0]], y_centers[bin_index[1]])
                cluster = bin_to_cluster.get(tuple(bin_coord), None)
                image_to_bin_cluster[images[i]] = {"bin": tuple(bin_index), "cluster": cluster}
            else:
                image_to_bin_cluster[images[i]] = {"bin": None, "cluster": None}  # Out-of-bounds case

        return image_to_bin_cluster
    
    def compute_cluster_statistics(cluster_examples,color_mode):
        """Computes mean and standard deviation of LAB values for each cluster."""
        cluster_stats = {}

        for cluster, examples in cluster_examples.items():
            stats = np.empty((len(examples), 2, 3))  # Shape matches `getavgstd_extended`
            for i, example in enumerate(examples):
                img = np.array(Image.open(example))
                stats[i] = np.asarray(SU.calculate_avg_std(img,mode=color_mode))
            cluster_stats[cluster] = np.mean(stats, axis=0)

        return cluster_stats

    # Build the image-to-bin-cluster dictionary
    image_to_bin_cluster = build_image_to_bin_cluster_dict(images, embedding, h, xy, cluster_labels)

    # Select 1 random image from each cluster for visualization
    cluster_examples = defaultdict(list)
    for image, data in image_to_bin_cluster.items():
        cluster_examples[data["cluster"]].append(image)

    representative_images = [random.choice(images) for images in cluster_examples.values()]

    VU.display_reference_images(
        representative_images,
        save_path="RepresentativeImages.pdf" if save else None,
        plotting_graphs=plot
    )

    print("The images displayed are not representative of the grouped data and are meant just for visualization.")

    # Compute the average and standard deviation of LAB values in each cluster
    cluster_stats = compute_cluster_statistics(cluster_examples,color_mode=color_mode)

    if save:
        pd.DataFrame(cluster_stats).to_csv("ClusterStats.csv", index=False)

    return representative_images, cluster_stats
#----------------------------------------------------------------------------
def __get_density_references__(embedding, colors, embedding_name, images,
                               plot, save, n_images=1, density_percentile_level=2,
                               number_of_bins=100, selection_mode="original"):
    """Handles selection of reference images based on density.
    
    Parameters:
      selection_mode: 
          - "original" (default): use the original approach.
          - "sorted_regions": sample one candidate from every region defined by the closed contour polygon
                              at the desired contour level, then if n_images is greater than the number of regions,
                              fetch additional candidates from within these regions in a density-sorted manner.
                              In both modes, if a candidate lies too near a boundary, an interior point is used.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors as mcolors
    from matplotlib.colorbar import ColorbarBase
    from scipy.ndimage import gaussian_filter
    from shapely.geometry import Polygon as ShapelyPolygon, Point

    def get_density_map(embedding, number_of_bins):
        histogram_padding_settings = 1
        x, y = embedding[:, 0], embedding[:, 1]
        
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=number_of_bins)
        hist = gaussian_filter(hist, sigma=2.5)
        hist = np.pad(hist, histogram_padding_settings, mode='constant', constant_values=0)

        x_bin_width = np.diff(x_edges[:2])[0]
        y_bin_width = np.diff(y_edges[:2])[0]
        
        x_edges = np.linspace(x_edges[0] - histogram_padding_settings * x_bin_width,
                              x_edges[-1] + histogram_padding_settings * x_bin_width,
                              hist.shape[1] + 1)
        y_edges = np.linspace(y_edges[0] - histogram_padding_settings * y_bin_width,
                              y_edges[-1] + histogram_padding_settings * y_bin_width,
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
        num_levels = len(density_levels)
        contour_colors = [cm.viridis(i / num_levels) for i in range(num_levels)]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", contour_colors, N=256)
        norm = mcolors.Normalize(vmin=density_levels[0], vmax=density_levels[-1])
        
        return (hist, X, Y, x_edges, y_edges, density, density_levels, 
                contour_colors, custom_cmap, norm, x_centers, y_centers, percentiles)


    def get_bin_points_in_contour(embedding, x_edges, y_edges, contour_set, desired_index=2, num_points=3):
        """
        Selects n_images candidate points (num_points) from the histogram grid inside the specified contour level,
        trying to pick at most one per contour path if possible. Not completely working
        """
        
        def farthest_point_sampling(points, n):
            """
            Greedily select n points from the array 'points' so that the chosen points are as far apart as possible.
            """
            points = np.array(points)
            if len(points) == 0:
                return []
            if n >= len(points):
                return points.tolist()
            selected = [points[0]]
            selected_indices = [0]
            while len(selected) < n:
                best_candidate = None
                best_distance = -np.inf
                for i in range(len(points)):
                    if i in selected_indices:
                        continue
                    d = min(np.linalg.norm(points[i] - p) for p in selected)
                    if d > best_distance:
                        best_distance = d
                        best_candidate = i
                selected.append(points[best_candidate])
                selected_indices.append(best_candidate)
            return np.array(selected).tolist()
        
        n_images = num_points
        # Compute bin centers from the edges.
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X_grid, Y_grid = np.meshgrid(x_centers, y_centers)
        bin_points = np.vstack([X_grid.flatten(), Y_grid.flatten()]).T

        # Get the contour collection at the desired index and its paths.
        collection = contour_set.collections[desired_index]
        paths = collection.get_paths()
        n_paths = len(paths)
        print(f"Number of paths: {n_paths}")

        # Build dictionary: for each path (region), gather candidate points.
        region_candidates = {}
        for i, path in enumerate(paths):
            region_candidates[i] = []
            for point in bin_points:
                if path.contains_point(point, radius=-1e-3):
                    region_candidates[i].append(point)
        
        # For each region, if there are candidate points, choose a primary candidate.
        # Here we choose the candidate that is farthest from the region's centroid.
        primary_candidates = {}  # key: region index, value: candidate point
        extra_candidates = {}    # key: region index, value: list of additional candidate points
        for i, pts in region_candidates.items():
            if len(pts) > 0:
                pts_arr = np.array(pts)
                centroid = pts_arr.mean(axis=0)
                # Compute distances from each candidate to the centroid.
                dists = np.linalg.norm(pts_arr - centroid, axis=1)
                primary_idx = np.argmax(dists)
                primary_candidates[i] = pts_arr[primary_idx]
                # Extra candidates: all others (if any)
                extra_candidates[i] = [pt for j, pt in enumerate(pts_arr) if j != primary_idx]
        
        # Convert primary candidates to a list.
        primary_list = list(primary_candidates.values())
        num_regions = len(primary_list)

        if num_regions >= n_images:
            # We have enough distinct regions: select n_images among the primary candidates using farthest point sampling.
            selected_points = farthest_point_sampling(primary_list, n_images)
        else:
            # Use all primary candidates.
            selected_points = primary_list.copy()
            remaining_needed = n_images - len(selected_points)
            # Gather extra candidates from all regions.
            extras = []
            for i in extra_candidates:
                extras.extend(extra_candidates[i])
            # If extras exist, use farthest point sampling to select remaining_needed candidates.
            if len(extras) > remaining_needed:
                extra_selected = farthest_point_sampling(extras, remaining_needed)
            else:
                extra_selected = extras
            selected_points.extend(extra_selected)
            # If we end up with too many (unlikely), randomly downsample.
            if len(selected_points) > n_images:
                selected_points = random.sample(selected_points, n_images)

        return np.array(selected_points)
        
        # --- Original Implementation ---
        
        #inside_points = []
        #for point in bin_points:
        #    for path in paths:
        #        if path.contains_point(point, radius=-1e-6):
        #            inside_points.append(point)
        #            break
        #inside_points = np.array(inside_points)
        #if inside_points.shape[0] < num_points:
        #    print("Not enough bin points found strictly inside the specified contour level; returning all found points.")
        #    return inside_points
        #else:
        #    selected_indices = np.random.choice(inside_points.shape[0], num_points, replace=False)
        #    return inside_points[selected_indices]

    def get_region_candidates(x_edges, y_edges, density, contour_set, desired_index=2):
        """
        Function to be implemented. Order the areas encircled by the contour polygons at the desired_index
        by their density, and return the points within these regions in order of decreasing density.
        Make sure to sample at least one point from each region.

        """
        return None

    def adjust_point_in_contour(pt, contour_set, desired_index=2, tol_factor=0.05):
        """
        For a given point, find the first contour polygon (at desired_index) that contains it.
        If the distance from the point to the polygon's boundary is less than a tolerance,
        return the polygon's representative (interior) point; otherwise, return the original point.
        """
        from shapely.geometry import Polygon as ShapelyPolygon, Point
        collection = contour_set.collections[desired_index]
        for path in collection.get_paths():
            if path.contains_point(pt, radius=-1e-6):
                poly = ShapelyPolygon(path.vertices)
                if not poly.is_valid or poly.is_empty:
                    continue
                minx, miny, maxx, maxy = poly.bounds
                tol = tol_factor * max(maxx - minx, maxy - miny)
                if poly.exterior.distance(Point(pt)) < tol:
                    return np.array(poly.representative_point())
                break
        return pt

    def map_bin_center_to_original(selected_bin_points, x_edges, y_edges, embedding):
        mapped_points = []
        for point in selected_bin_points:
            i = np.searchsorted(x_edges, point[0]) - 1
            j = np.searchsorted(y_edges, point[1]) - 1
            x_low, x_high = x_edges[i], x_edges[i + 1]
            y_low, y_high = y_edges[j], y_edges[j + 1]
            mask = ((embedding[:, 0] >= x_low) & (embedding[:, 0] < x_high) &
                    (embedding[:, 1] >= y_low) & (embedding[:, 1] < y_high))
            points_in_bin = embedding[mask]
            if points_in_bin.shape[0] > 0:
                rep_point = points_in_bin.mean(axis=0)
                mapped_points.append(rep_point)
            else:
                mapped_points.append(point)
        return np.array(mapped_points)

    def build_density_targets(embedding, mapped_points):
        targets = []
        for point in mapped_points:
            distances = np.linalg.norm(embedding - point, axis=1)
            closest_index = np.argmin(distances)
            targets.append(closest_index)
        return np.array(targets)
    
    # --- Main Execution ---
    (hist, X, Y, x_edges, y_edges, density, density_levels, 
     contour_colors, custom_cmap, norm, x_centers, y_centers, percentiles) = get_density_map(
            embedding=embedding, number_of_bins=number_of_bins)

    fig_temp, ax_temp = plt.subplots(figsize=(8, 6))
    contour_set = ax_temp.contour(X, Y, density.T, levels=density_levels, colors=contour_colors)
    plt.close(fig_temp)
    
    if selection_mode == "original":
        selected_bin_points = get_bin_points_in_contour(embedding, x_edges, y_edges, 
                                                        contour_set, desired_index=density_percentile_level, 
                                                        num_points=n_images)
        # Adjust each selected point to ensure it's not too close to the boundary.
        selected_bin_points = [adjust_point_in_contour(pt, contour_set, desired_index=density_percentile_level) 
                               for pt in selected_bin_points]
    elif selection_mode == "sorted_regions":
        print (f"Mode not yet implemented: {selection_mode}")
        # Default to original mode
        selected_bin_points = get_bin_points_in_contour(embedding, x_edges, y_edges, 
                                                        contour_set, desired_index=density_percentile_level, 
                                                        num_points=n_images)
        # Adjust each selected point to ensure it's not too close to the boundary.
        selected_bin_points = [adjust_point_in_contour(pt, contour_set, desired_index=density_percentile_level) 
                               for pt in selected_bin_points]
    else:
        raise ValueError(f"Invalid selection mode: {selection_mode}")

    mapped_points = map_bin_center_to_original(selected_bin_points, x_edges, y_edges, embedding)
    targets = build_density_targets(embedding, mapped_points)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=100, c=colors, label=f'{embedding_name} Points')
    contour_set = ax.contour(X, Y, density.T, levels=density_levels, colors=contour_colors)
    ax.set_xlabel(f"{embedding_name} 1")
    ax.set_ylabel(f"{embedding_name} 2")
    ax.set_title(f"{embedding_name} with Density Contours (Percentile Levels)")
    ax.scatter(np.array(mapped_points)[:, 0], np.array(mapped_points)[:, 1], marker='x', s=200,
               c='red', linewidths=2, label='Mapped Bin Points')
    ax.axis('off')
    cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cb = ColorbarBase(cax, cmap=custom_cmap, norm=norm, orientation='vertical')
    cb.set_ticks(density_levels)
    cb.set_ticklabels([f'{100 - p}%' for p in percentiles])
    plt.show()
    
    if not plot:
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





