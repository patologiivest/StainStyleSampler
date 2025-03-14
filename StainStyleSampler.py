# Standard Library Imports
import os
import glob
import random
from collections import defaultdict

# Third-Party Library Imports
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.filters import threshold_otsu
from scipy.stats import kurtosis, skew, entropy
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn import mixture
from umap.umap_ import UMAP
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from matplotlib.colors import Normalize
from joblib import Parallel,delayed
from tqdm_joblib import tqdm_joblib,ParallelPbar
import tqdm
from natsort import os_sorted
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.path import Path

# HistomicsTK Imports
from histomicstk.preprocessing.color_conversion import (
    rgb_to_lab, lab_mean_std, rgb_to_hsi
)
from histomicstk.preprocessing.color_deconvolution import (
    stain_color_map, rgb_separate_stains_macenko_pca, color_deconvolution, find_stain_index
)
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask, threshold_multichannel
)
from histomicstk.utils import simple_mask

# Local Imports
import Utils.StainUtils as SU
import Utils.VisualizationUtils as VU
import Utils.ReferenceUtils as RU
import Utils.ClusteringUtils as CU

class StainStyleSampler():
    
    def __init__(self):
        pass

    def build_features(self, dataset_path: str
                       , fraction: float = None, mode: str = 'lab', background_removal:bool = True,
                       stain_deconv: bool = None,split_stains: bool = None,multiprocessing: bool = True) -> None:
        """
        Build features and color representations from images in the dataset.

        Args:
            dataset_path (str): Path to the dataset containing images.
            fraction (float, optional): Fraction of images to sample (0 < fraction <= 1). Defaults to None (use all images).
            mode (str): Color mode for feature extraction ('lab', 'rgb', 'hsv', 'hsi'). Defaults to 'lab'.
            stain_deconv (bool, optional): Whether to use stain deconvolution. Defaults to None.
            split_stains (bool, optional): Whether to split stain features. Defaults to None.
        """

        def calculate_features_len(mode: str, stain_deconv: bool, stain_split: bool) -> int:
            """
            Calculate the length of features based on mode, stain_deconv, and stain_split.

            Args:
                mode (str): The color mode ('lab', 'rgb', 'hsv', 'hsi').
                stain_deconv (bool): Whether to use stain deconvolution.
                stain_split (bool): Whether to split stain features.

            Returns:
                int: The length of features.
            """
            # Validate mode
            valid_modes = ['lab', 'rgb', 'hsv', 'hsi']
            if mode not in valid_modes:
                raise ValueError(f"Invalid mode '{mode}'. Supported modes are: {valid_modes}")

            # Determine feature length
            if not stain_deconv:
                if mode not in ['lab']:
                    return 12  # Mode-based features
                else:
                    return 6
            elif not stain_split:
                return 8  # Stain deconvolution without split
            else:
                return 24  # Stain deconvolution with split
            
        def process_individual_image(image_path: str,mode: str,background_removal:bool,stain_deconv: bool, split_stains: bool) -> tuple:
            """
            Process individual image to extract features and colors.

            Args:
                image_path (str): Path to the image.
                mode (str): Color mode for feature extraction ('lab', 'rgb', 'hsv', 'hsi').
                stain_deconv (bool): Whether to use stain deconvolution.
                split_stains (bool): Whether to split stain features.

            Returns:
                tuple: Tuple containing features and colors.
            """
            img = Image.open(image_path).resize((512, 512))
            img_array = np.array(img)[:,:,:3]

            # Compute color features
            features = SU.get_stain_features(img_array, mode, background_removal,stain_deconv, split_stains)

            # Compute mean RGB vector for scatterplot
            colors = np.mean(img_array.reshape((512*512,3)), axis=0)/255.0

            return features, colors,image_path

        assert background_removal in (None, True, False), "Background removal argument must be either None, True, or False"
        assert stain_deconv in (None,True,False), "Stain deconv argument must be either None,True or False"
        assert split_stains in (None,True,False), "Split stains argument must be either None,True or False"
        assert multiprocessing in (None,True,False), "Multiprocessing argument must be either None,True or False"

        if dataset_path is None:
            raise ValueError("Please provide a valid dataset path.")
        
        self.dataset_path = dataset_path
        self.mode = mode
        # ----------------------------------
        # Build list of images in the given path
        all_images = np.array(os_sorted(glob.glob(f"{dataset_path}\**\*.png",recursive=True)))

        if len(all_images) == 0:
            raise FileNotFoundError(f"No images found in the specified path: {dataset_path}")

        # If a fraction is specified, randomly select a subset
        if fraction is not None and fraction != 1:
            if not (0 < fraction < 1):
                raise ValueError("Fraction must be a value between 0 and 1.")
            
            num_images = int(len(all_images) * fraction)
            selected_indices = np.random.choice(len(all_images), size=num_images, replace=False)
            self.images = all_images[selected_indices]
        else:
            self.images = all_images

        # ----------------------------------
        # Initialize arrays for features and colors
        num_images = len(self.images)
        len_features = calculate_features_len(mode, stain_deconv, split_stains)
        
        self.features = np.empty((num_images, len_features))
        self.colors = np.empty((num_images, 3))
        analysed_images = np.empty((num_images,1),dtype=object)

        # ----------------------------------
        # Compute features for each image
        if multiprocessing:
            results = ParallelPbar("Extracting features")(n_jobs=8)(
                delayed(process_individual_image)(image_path, mode,background_removal, stain_deconv, split_stains)
                    for image_path in self.images
            )

            for i, (features, colors,image_path) in enumerate(results):
                self.features[i] = features
                self.colors[i] = colors
                analysed_images[i] = image_path

        else:
            for i, image_path in enumerate(tqdm.tqdm(self.images, desc="Computing Features")):
                features, colors = process_individual_image(image_path, mode,background_removal, stain_deconv, split_stains)
                
                self.features[i] = features
                self.colors[i] = colors
                analysed_images[i] = image_path
        
        # Normalize features to [0, 1]
        self.features = (self.features - np.min(self.features, axis=0)) / (np.max(self.features, axis=0) - np.min(self.features, axis=0))
        
        # ------------------------------------------------------
        # Build the DataFrame
        # ------------------------------------------------------
        # Convert numpy arrays to lists (optional, but common for DataFrame storage)
    
        # Create a DataFrame with columns: image_path, colors, features
        self.df = pd.DataFrame({
            'image_path': analysed_images.tolist(),
            'colors': self.colors.tolist(),
            'features': self.features.tolist()
        })
        
        return self.features,self.colors,self.df,analysed_images
    
    def build_embedding_map(self, mode: str = 'UMAP', plot: bool = False, save: bool = False, pca_components: int = None) -> None:
        """
        Build a 2D embedding map using PCA or UMAP.

        Args:
            mode (str): Embedding mode ('UMAP' or 'PCA'). Default is 'UMAP'.
            plot (bool): Whether to plot the embedding. Default is False.
            save (bool): Whether to save the embedding plot. Default is False.
            pca_components (int): Number of components for PCA. Default is None (use all components).

        Returns:
            None
        """
        assert mode in ['UMAP', 'PCA'], "Invalid mode, choose between 'UMAP' and 'PCA'."
        # Reset PCA and UMAP embeddings
        self.PCA = None
        self.UMAP = None

        # Perform PCA prefiltration
        pca = PCA(n_components=pca_components) if pca_components else PCA()
        self.PCA = pca.fit_transform(self.features)

        if mode == 'PCA':
            embedding, title, save_path = self.PCA, f"PCA embedding ({self.mode})", "PCA.pdf"
        else:
            # Perform UMAP on PCA-filtered data
            umap = UMAP(random_state=13)
            embedding = umap.fit_transform(self.PCA)
            self.UMAP = np.array(list(zip(embedding[:,0],embedding[:,1])))
            
            embedding = self.UMAP
            title, save_path = f"UMAP embedding ({self.mode})", "UMAP.pdf"
            # Clear PCA embedding to save memory
            self.PCA = None

        # Save or plot embedding
        VU.save_or_plot(
            lambda: (
                plt.scatter(embedding[:, 0], embedding[:, 1], c=self.colors),
                plt.title(title),
                plt.axis('off')
            ),
            save_path=save_path if save else None,
            plotting_graphs=plot
        )

        return embedding

    
    def build_2d_histogram(self, bins: int = 100, plot: bool = False, save: bool = False) -> None:
        """
        Build a 2D histogram using PCA or UMAP embeddings.

        Args:
            bins (int): Number of bins for the histogram. Default is 100.
            plot (bool): Whether to plot the histogram. Default is False.
            save (bool): Whether to save the histogram plot. Default is False.

        Returns:
            None
        """
        # Determine which embedding method was used
        embedding = None
        embedding_name = None

        if self.UMAP is not None and self.PCA is None:
            embedding, embedding_name = self.UMAP, "UMAP"
        elif self.PCA is not None and self.UMAP is None:
            embedding, embedding_name = self.PCA, "PCA"
        else:
            raise ValueError("No valid embedding found. Ensure either PCA or UMAP has been computed.")

        # Compute histogram range dynamically
        range_ = [
            [np.min(embedding[:, 0]) - 1e-5, np.max(embedding[:, 0]) + 1e-5],
            [np.min(embedding[:, 1]) - 1e-5, np.max(embedding[:, 1]) + 1e-5]
        ]

        # Compute 2D histogram
        self.HISTOGRAM = hist2d(embedding[:, 0], embedding[:, 1], bins=bins, range=range_)

        # Save or plot the histogram
        VU.save_or_plot(
            lambda: (
                plt.imshow(self.HISTOGRAM[0], aspect='auto'),
                plt.axis('off'),
                plt.title(f"2D Histogram ({embedding_name})")
            ),
            save_path="2Dhistogram.pdf" if save else None,
            plotting_graphs=plot
        )

    def build_bins_coordinates(self, plot: bool = False, save: bool = False) -> None:
        """
        Compute center coordinates for non-empty histogram bins.

        Args:
            plot (bool): Whether to plot the bin coordinates. Default is False.
            save (bool): Whether to save the plot. Default is False.

        Returns:
            None
        """
        if self.HISTOGRAM is None:
            raise ValueError("HISTOGRAM has not been computed. Run `build_2d_histogram` first.")

        histogram, x_edges, y_edges = self.HISTOGRAM[0], self.HISTOGRAM[1], self.HISTOGRAM[2]

        # Compute center coordinates of bins
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Get indices of non-empty bins
        non_empty_indices = np.column_stack(np.where(histogram > 0))

        # Extract corresponding coordinates
        x_coords = x_centers[non_empty_indices[:, 0]]
        y_coords = y_centers[non_empty_indices[:, 1]]

        # Store bin coordinates
        self.XY = np.column_stack((x_coords, y_coords))

        # Save or plot bin coordinates
        VU.save_or_plot(
            lambda: (
                plt.scatter(self.XY[:, 0], self.XY[:, 1]),
                plt.axis('off'),
                plt.title("Non-Empty Bin Coordinates")
            ),
            save_path="BinsCoordinates.pdf" if save else None,
            plotting_graphs=plot
        )

    def get_image_references(self, reference_mode: str = 'representative',
                              density_selection_mode: str = 'original', density_percentile_level: int = 2,
                         plot: bool = False, save: bool = False) -> None:
        """
        Selects reference images based on different modes: 'random', 'density', 'representative', or 'grouped'.
        Automatically determines whether to use UMAP or PCA for clustering.

        Args:
            reference_mode (str): Reference selection mode ('random', 'density', 'representative', 'grouped').
            density_percentile_level (int): Percentile level for density-based selection. Default is 2.
            plot (bool): Whether to plot results. Default is False.
            save (bool): Whether to save plots. Default is False.

        Returns:
            None
        """
        # Initialize attributes
        self.ReferenceFiles = None
        self.ClusterStats = None
        self.nImages = None
        self.N_Clusters = None

        # Validate reference_mode
        valid_modes = ['random', 'density', 'representative', 'grouped']
        if reference_mode not in valid_modes:
            raise ValueError(f"Invalid reference mode '{reference_mode}'. Choose from {valid_modes}")

        # Determine embedding method automatically (UMAP or PCA)
        if self.UMAP is not None and self.PCA is None:
            embedding, embedding_name = self.UMAP, "UMAP"
        elif self.PCA is not None and self.UMAP is None:
            embedding, embedding_name = self.PCA, "PCA"
        else:
            raise ValueError("No valid embedding found. Ensure either PCA or UMAP has been computed.")

        # Handle different reference modes
        if reference_mode == 'random':
            self.__handle_random_reference__(embedding, embedding_name, plot, save)
        elif reference_mode == 'representative':
            self.__handle_representative_reference__(embedding,self.XY, embedding_name, plot, save)
        elif reference_mode == 'grouped':
            assert embedding_name == 'UMAP', "Grouped reference selection was only implemented for UMAP embedding."
            self.__handle_grouped_reference__(embedding, self.XY, embedding_name, plot, save)
        elif reference_mode == 'density':
            self.__handle_density_reference__(embedding,embedding_name,density_selection_mode,density_percentile_level,plot,save)
        


    def __handle_random_reference__(self, embedding: np.ndarray, embedding_name: str, plot: bool, save: bool) -> None:
        """Handles reference selection using a random sampling method."""
        self.nImages = RU.input_for_number_of_images()
        self.ReferenceFiles = RU.get_references(
            embedding, self.colors, self.XY, None, None, self.images, 
            n_images=self.nImages, reference_mode='random', plot=plot, save=save
        )

    def __handle_representative_reference__(self, embedding: np.ndarray,XY: np.ndarray, embedding_name: str, plot: bool, save: bool) -> None:
        """Handles reference selection using a representative sampling method."""

        method_for_clustering = CU.input_for_clustering_method()
        method_for_number_of_clusters = CU.input_for_number_of_clusters_method(method_for_clustering)

        # Define number of clusters
        self.__define_number_of_clusters__(method=method_for_number_of_clusters, clustering_method=method_for_clustering, plot=plot, save=save)

        # Perform clustering
        self.__perform_clustering__(clustering_method=method_for_clustering, plot=plot, save=save)

        # Scatter plot of clusters
        VU.save_or_plot(
            lambda: (
                plt.scatter(self.XY[:, 0], self.XY[:, 1], c=self.cluster_labels),
                plt.axis('off'),
                plt.title(f"Clusters ({embedding_name})")
            ),
            save_path="Clusters.pdf" if save else None,
            plotting_graphs=plot
        )

        # Select reference images based on clustering method
        self.ReferenceFiles = RU.get_references(
            embedding, self.colors, self.XY, self.cluster_labels, self.cluster_centers, 
            self.images, reference_mode='representative', plot=plot, save=save
        )

    def __handle_grouped_reference__(self,embedding: np.ndarray, XY: np.ndarray, embedding_name: str, plot: bool, save: bool) -> None:
        """Handles reference selection using a grouped sampling method."""
        method_for_clustering = CU.input_for_clustering_method()
        method_for_number_of_clusters = CU.input_for_number_of_clusters_method(method_for_clustering)
        
        # Define number of clusters
        self.__define_number_of_clusters__(method=method_for_number_of_clusters, clustering_method=method_for_clustering, plot=plot, save=save)

        # Perform clustering
        self.__perform_clustering__(clustering_method=method_for_clustering, plot=plot, save=save)

        # Scatter plot of clusters
        VU.save_or_plot(
            lambda: (
                plt.scatter(self.XY[:, 0], self.XY[:, 1], c=self.cluster_labels),
                plt.axis('off'),
                plt.title(f"Clusters ({embedding_name})")
            ),
            save_path="Clusters.pdf" if save else None,
            plotting_graphs=plot
        )

        # Select reference images based on clustering method
        self.ReferenceFiles,self.ClusterStats = RU.get_references(embedding,self.colors,XY,self.cluster_labels,self.cluster_centers,
                                                                  self.images,h=self.HISTOGRAM,reference_mode='grouped',plot=plot,save=save,embedding_name=embedding_name)

    def __handle_density_reference__(self,embedding: np.ndarray,embedding_name: str,density_selection_mode: str, density_percentile_level: int,plot: bool,save: bool) -> None:
        # Input the number of images to retrieve
        self.nImages = RU.input_for_number_of_images()

        self.ReferenceFiles,self.contours = RU.get_references(
            embedding, self.colors, None, None,
              None, self.images,n_images=self.nImages,
                reference_mode='density', plot=plot,
                  save=save,embedding_name=embedding_name,
                  density_selection_mode=density_selection_mode, density_percentile_level=density_percentile_level
        )

#--------------------------------------------------------------------------------------------------------------
    def __define_number_of_clusters__(self,method='auto',clustering_method='gmm',plot=False,save=False):
        self.N_Clusters = CU.get_number_of_clusters(method, self.XY, clustering_method,plot=plot,save=save)
        return None
    
    def __perform_clustering__(self,clustering_method='gmm',plot=False,save=False):
        self.cluster_labels, self.cluster_centers, self.cluster_metrics = CU.get_clusters(self.XY, self.N_Clusters, clustering_method)
        return None
#--------------------------------------------------------------------------------------------------------------
    # Define all the getters
    def get_images(self):
        return self.images
    
    def get_features(self):
        return self.features
    
    def get_features_structured(self):
        # Build a dictionary for each "channel" metric contained inside the features
        # for example: RGB -> np mean == [mean_R, mean_G, mean_B]
        return
    
    def get_colors(self):
        return self.colors
    
    def get_umap(self):
        return self.UMAP
    
    def get_pca(self):  
        return self.PCA
    
    def get_embedding_method(self):
        if isinstance(self.UMAP, np.ndarray) and self.UMAP.size > 0 and not isinstance(self.PCA, np.ndarray):
            return 'UMAP'
        elif isinstance(self.PCA, np.ndarray) and self.PCA.size > 0 and not isinstance(self.UMAP, np.ndarray):
            return 'PCA'
        else:
            raise ValueError("Both UMAP and PCA embeddings are either unset or invalid. Ensure one is properly computed.")
    
    def get_histogram(self):
        return self.HISTOGRAM
    
    def get_bins_coordinates(self):
        return self.XY
    
    def get_reference_files(self):
        return self.ReferenceFiles
    
    def get_cluster_labels(self):
        return self.cluster_labels
    
    def get_cluster_centers(self):
        return self.cluster_centers
    
    def get_cluster_metrics(self):
        return self.cluster_metrics
    
    def get_cluster_stats(self):
        return self.ClusterStats
    
    def get_density(self):
        return self.density
    
    def get_percentiles(self):
        return self.percentiles
    
    def get_contours(self,as_polygons=False):
        def contour_to_multipolygon(contour_set):
            """
            Converts a Matplotlib contour set into a Shapely MultiPolygon.
            
            Args:
                contour_set: The object returned by matplotlib's contour() function.
            
            Returns:
                A dictionary where each contour level is associated with a MultiPolygon.
            """
            level_polygons = {}  # Dictionary to store MultiPolygons per level

            levels = contour_set.levels if hasattr(contour_set, "levels") else [None] * len(contour_set.collections)

            for level_index, collection in enumerate(contour_set.collections):
                level_value = levels[level_index]
                polygons = []

                for path in collection.get_paths():
                    vertices = path.vertices  # Extract contour path points
                    codes = path.codes  # Extract path instructions (MOVETO, LINETO, CLOSEPOLY)

                    # Identify where each contour starts and ends
                    start_indices = np.where(codes == 1)[0]  # Find MOVETO (start of new contour)
                    
                    for i in range(len(start_indices)):
                        start_idx = start_indices[i]
                        end_idx = start_indices[i + 1] if i + 1 < len(start_indices) else len(vertices)

                        contour_segment = vertices[start_idx:end_idx]  # Get contour segment

                        # Ensure it forms a valid polygon (closed shape)
                        if len(contour_segment) > 2 and np.array_equal(contour_segment[0], contour_segment[-1]):
                            polygon = Polygon(contour_segment)
                            if polygon.is_valid and not polygon.is_empty:
                                polygons.append(polygon)

                # Store polygons as a MultiPolygon for this contour level
                if polygons:
                    level_polygons[level_value] = MultiPolygon(polygons)

            return level_polygons  # Returns a dictionary with contour level -> MultiPolygon mapping
        
        if as_polygons:
            return contour_to_multipolygon(self.contours)
        else:
            return self.contours
    
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
    def get_targets(self):
        return self.targets
    
    def get_nImages(self):
        return self.nImages
    
    def get_N_Clusters(self):
        return self.N_Clusters
