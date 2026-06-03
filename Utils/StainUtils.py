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
import tqdm

# HistomicsTK Imports
from histomicstk.preprocessing.color_conversion import rgb_to_lab, rgb_to_hsi
from histomicstk.saliency.tissue_detection import threshold_multichannel

# Tiatoolbox Import
from tiatoolbox.tools.stainnorm import MacenkoNormalizer

def get_background_mask(img: np.ndarray) -> np.ndarray:
    """
    Generate a background/foreground mask for the input image.

    Args:
        img (np.ndarray): Input RGB image.
        method (str): Method to create the mask ('Old' or 'New').

    Returns:
        np.ndarray: Binary mask where white/background pixels are labeled as 1.
    """
    try:
        background_mask, _ = threshold_multichannel(rgb_to_hsi(img), {
                    'hue': {'min': 0, 'max': 1.0},
                    'saturation': {'min': 0, 'max': 0.2},
                    'intensity': {'min': 220, 'max': 255},
                }, just_threshold=True)
        foreground_mask = np.where((1-background_mask)>0,1,0)
    except:
        background_mask = np.zeros_like(img)[:,:,0]
        foreground_mask = np.ones_like(img)[:,:,0]
    
    return background_mask

def get_stains_deconvoluted(img: np.ndarray, I_0: int = 255) -> tuple:
    """
    Perform color deconvolution to separate stains in an image.

    Args:
        img (np.ndarray): Input RGB image.
        I_0 (int): Maximum intensity value. Default is 255 for 8-bit images.

    Returns:
        tuple: Stain matrix
    """
    normalizer = MacenkoNormalizer()
    normalizer.fit(img)

    stain_matrix = (normalizer.stain_matrix_target).reshape((6,))
    stain_maxC_target = (normalizer.maxC_target).reshape((2,))
    combined_array = np.concatenate((stain_matrix, stain_maxC_target))
    
    return combined_array

def validate_feature_extraction_params(stain_deconv: bool, mode: str) -> None:
    """
    Validate the parameters for feature extraction.

    Args:
        stain_deconv (bool): Whether to use stain deconvolution.
        mode (str): Color mode for feature extraction.

    Raises:
        AssertionError: If parameters are inconsistent.
    """
    # Ensure 'mode' is valid
    valid_modes = ['lab', 'rgb', 'hsv', 'hsi']
    assert mode in valid_modes, f"Invalid mode '{mode}'. Supported modes are: {valid_modes}"

    # Validate conditions based on stain_deconv
    if stain_deconv:
        assert True, "When 'stain_deconv' is True, 'mode' is optional."
    else:
        assert mode in valid_modes, "If 'stain_deconv' is False, a valid 'mode' must be specified."

def mode_conversion(mode: str, img: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
    """
    Convert the input image to the specified color mode based on the background mask.

    Args:
        mode (str): Color mode ('lab', 'rgb', 'hsv', 'hsi').
        img (np.ndarray): Input RGB image.
        background_mask (np.ndarray): Binary mask where white (background) pixels are labeled as 1.

    Returns:
        np.ndarray: Image converted to the specified color mode.
    """
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """ Normalize the image to [0, 1] range per channel. """
        min_val = image.min(axis=(0,), keepdims=True)
        max_val = image.max(axis=(0,), keepdims=True)
        normalized = (image - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
        return normalized

    if mode == 'lab':
        converted = rgb_to_lab(img)[background_mask == 0]
    elif mode == 'rgb':
        converted = img[background_mask == 0]
    elif mode in ['hsv', 'hsi']:
        converted = rgb_to_hsi(img)[background_mask == 0]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return normalize_image(converted)


def get_stain_features(img,mode: str = 'lab',background_removal:bool = None,stain_deconv: bool = None) -> list:
    """
    Analyze image features based on color space or stain deconvolution.

    Args:
        img (np.ndarray): Input RGB image.
        mode (str): Color space mode ('lab', 'rgb', 'hsv', 'hsi').
        stain_deconv (bool): Whether to apply stain deconvolution.

    Returns:
        list: Flattened list of computed features (mean, std, kurtosis, skew).
    """
    # Validate input parameters
    try:
        validate_feature_extraction_params(stain_deconv, mode)
    except AssertionError as e:
        print(f"Error: {e}")
        return None

    if background_removal: 
        background_mask = get_background_mask(img)
    else:
        background_mask = np.zeros_like(img)[:,:,0]

    # For plotting
    stain_color = np.mean(img[background_mask == 0],axis=0)

    if not stain_deconv:
        # Apply color space conversion based on mode
        img_array_background_removed = mode_conversion(mode,img,background_mask)

        if mode not in ['lab']:
            # Compute features for the entire image
            return list(np.concatenate((
                np.mean(img_array_background_removed, axis=0),
                np.std(img_array_background_removed, axis=0),
                kurtosis(img_array_background_removed, axis=0),
                skew(img_array_background_removed, axis=0)
            ))),stain_color

        else:
            return list(np.concatenate((
                np.mean(img_array_background_removed,axis=0),
                np.std(img_array_background_removed,axis=0)
            ))),stain_color
    
    else:
        # Calculate stain deconvolution matrix
        stain_matrix = get_stains_deconvoluted(img)
        return (list(stain_matrix)),stain_color
