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

def get_background_mask(img: np.ndarray, method: str = 'Old') -> np.ndarray:
    """
    Generate a white mask for the input image using the specified method.

    Args:
        img (np.ndarray): Input RGB image.
        method (str): Method to create the mask ('Old' or 'New').

    Returns:
        np.ndarray: Binary mask where white (background) pixels are labeled as 1.
    """
    def old_method(img: np.ndarray) -> np.ndarray:
        """Create a white mask using the 'Old' method with thresholding."""
        white_mask, _ = threshold_multichannel(rgb_to_hsi(img), {
            'hue': {'min': 0, 'max': 1.0},
            'saturation': {'min': 0, 'max': 0.2},
            'intensity': {'min': 220, 'max': 255},
        }, just_threshold=True)
        return white_mask

    def new_method(img: np.ndarray) -> np.ndarray:
        """Create a white mask using the 'New' method with a simple mask."""
        return np.where(simple_mask(img),0,1)

    # Ensure the method is valid
    assert method in ['Old', 'New'], "method must be one of 'Old' or 'New'"

    # Choose the appropriate method
    if method == 'Old':
        return old_method(img)
    elif method == 'New':
        try:
            return new_method(img)
        except:
            return old_method(img)
        
def estimate_I_0(img: np.ndarray, sample_fraction: float = 1.0) -> int:
    """
    Estimate the maximum intensity value (I_0) of an image by sampling a fraction of pixels.

    Args:
        img (np.ndarray): Input RGB image.
        sample_fraction (float): Fraction of pixels to sample (range [0, 1]).

    Returns:
        int: Estimated maximum intensity value (I_0), based on the 95th percentile.
    """
    def sample_pixels_from_mask(mask: np.ndarray, fraction: float) -> np.ndarray:
        """Sample a fraction of pixels from a given mask."""
        non_zero_indices = np.nonzero(mask.flatten())[0]
        if non_zero_indices.size == 0:
            # Fallback: if no non-zero pixels, return indices of all pixels.
            return np.arange(mask.size)
        float_samples = fraction * non_zero_indices.size
        num_samples = int(np.floor(float_samples))
        num_samples += np.random.binomial(1, float_samples - num_samples)
        if num_samples == 0:
            # Fallback: ensure at least one sample is taken.
            num_samples = non_zero_indices.size
        sampled_indices = np.random.choice(non_zero_indices, num_samples, replace=False)
        return sampled_indices

    def compute_I_0(samples: np.ndarray) -> int:
        """Compute the 95th percentile intensity value and clip to valid range."""
        if samples.size == 0:
            # Fallback default value for empty samples.
            print("Warning: No samples available for computing I_0. Returning default value 255.")
            return 255
        I_0 = np.percentile(samples, 95, axis=0)
        I_0 = np.clip(I_0, 0, 255)
        return int(np.median(I_0))

    # Generate background mask and sample indices
    bgnd_mask = get_background_mask(img)
    sample_indices = sample_pixels_from_mask(bgnd_mask, sample_fraction)

    # Convert image to a linear pixel array and sample pixels
    img_pixels = img.reshape(-1, 3)
    sampled_pixels = img_pixels[sample_indices, :]

    # Fallback: if no pixels were sampled, use the entire image
    if sampled_pixels.size == 0:
        print("Warning: No sampled pixels found, falling back to using entire image pixels.")
        sampled_pixels = img_pixels

    # Calculate and return I_0
    return compute_I_0(sampled_pixels)

def get_stains_deconvoluted(img: np.ndarray, I_0: int = 255) -> tuple:
    """Not fully tested yet. Use with caution."""
    """
    Perform color deconvolution to separate stains in an image.

    Args:
        img (np.ndarray): Input RGB image.
        I_0 (int): Maximum intensity value. Default is 255 for 8-bit images.

    Returns:
        tuple: Separated stain images (stain_1, stain_2).
    """
    def extract_stain(stain_name: str, w_est: np.ndarray, deconv_result: np.ndarray) -> np.ndarray:
        """Extract a specific stain based on its name."""
        stain_index = find_stain_index(color_map[stain_name], w_est)
        return I_0 - deconv_result.Stains[:, :, stain_index]

    assert I_0 > 0 or I_0 is None or I_0 == 'auto', "I_0 must be a positive integer or 'auto'"

    # Define color map and stains
    color_map = stain_color_map
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # for cases with only two stains

    if I_0 is None or 'auto':
        I_0 = estimate_I_0(img)

    # Perform Macenko PCA-based stain separation
    w_est = rgb_separate_stains_macenko_pca(img, I_0)
    deconv_result = color_deconvolution(img, w_est, I_0)

    # Extract cytoplasm and nuclei stains
    stain_1 = extract_stain(stains[1], w_est, deconv_result)  # Cytoplasm (eosin)
    stain_2 = extract_stain(stains[0], w_est, deconv_result)  # Nuclei (hematoxylin)

    return stain_1, stain_2

def validate_feature_extraction_params(stain_deconv: bool, split_stains: bool, mode: str) -> None:
    """
    Validate the parameters for feature extraction.

    Args:
        stain_deconv (bool): Whether to use stain deconvolution.
        split_stains (bool): Whether to split stain features.
        mode (str): Color mode for feature extraction.

    Raises:
        AssertionError: If parameters are inconsistent.
    """
    # Ensure 'mode' is valid
    valid_modes = ['lab', 'rgb', 'hsv', 'hsi']
    assert mode in valid_modes, f"Invalid mode '{mode}'. Supported modes are: {valid_modes}"

    # Validate conditions based on stain_deconv and split_stains
    if split_stains:
        assert stain_deconv, "If 'split_stains' is True, 'stain_deconv' must also be True."
        assert mode in valid_modes, "If 'split_stains' is True, a valid 'mode' must also be specified."
    elif stain_deconv:
        assert True, "When 'stain_deconv' is True, 'mode' is optional."
    else:
        assert mode in valid_modes, "If 'stain_deconv' is False, a valid 'mode' must be specified."

# Map modes to conversion functions 
# # White mask is applied to remove background pixels when calculating the features
# This code bypasses errors when the image is fully background (all white) but ideally your dataset should not contain such examples

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

    def handle_all_background_images(img, mode):
        fake_background = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_RGB2GRAY)

        if mode == 'lab':
            converted = rgb_to_lab(img)[fake_background == 0]
        elif mode == 'rgb':
            converted = img[fake_background == 0]
        elif mode in ['hsv', 'hsi']:
            converted = rgb_to_hsi(img)[fake_background == 0]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return normalize_image(converted)

    if np.any(background_mask == 0):  # There are foreground pixels
        if mode == 'lab':
            converted = rgb_to_lab(img)[background_mask == 0]
        elif mode == 'rgb':
            converted = img[background_mask == 0]
        elif mode in ['hsv', 'hsi']:
            converted = rgb_to_hsi(img)[background_mask == 0]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return normalize_image(converted)
    else:  # If the whole image is background
        return handle_all_background_images(img, mode)


def get_stain_features(img,mode: str = 'lab',background_removal:bool = None,stain_deconv: bool = None, split_stains: bool = None) -> list:
    """
    Analyze image features based on color space or stain deconvolution.

    Args:
        img (np.ndarray): Input RGB image.
        mode (str): Color space mode ('lab', 'rgb', 'hsv', 'hsi').
        stain_deconv (bool): Whether to apply stain deconvolution.
        split_stains (bool): Whether to split analysis for individual stains.

    Returns:
        list: Flattened list of computed features (mean, std, kurtosis, skew).
    """
    # Validate input parameters
    try:
        validate_feature_extraction_params(stain_deconv, split_stains, mode)
    except AssertionError as e:
        print(f"Error: {e}")
        return None
        
    def threshold_stain(stain: np.ndarray,shape: np.uint8 =3) -> np.ndarray:
        """Apply Otsu thresholding to isolate significant pixels."""
        """Returns a mask of the significant pixels either binary or 3d"""
        assert shape in [3,2]

        threshold = threshold_otsu(stain)
        mask = stain > threshold
        if shape == 3:
            return mask[:,:,np.newaxis]
        else:
            return mask

    if background_removal:
        # Get white mask to remove background pixels from analysis
        background_mask = get_background_mask(img)
    else:
        background_mask = np.zeros_like(img)[:,:,0]

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
                )))

            else:
                return list(np.concatenate((
                    np.mean(img_array_background_removed,axis=0),
                    np.std(img_array_background_removed,axis=0)
                )))
    
    else:
        if not split_stains:
            # Apply stain deconvolution
            stain_1, stain_2 = get_stains_deconvoluted(img, I_0=estimate_I_0(img))
            
            # Compute features for both stains, grouping features by stain.
            # For each stain, compute: Mean, Std, Kurtosis, and Skew.
            features_stain1 = [
                np.mean(stain_1.flatten()),
                np.std(stain_1.flatten()),
                kurtosis(stain_1.flatten()),
                skew(stain_1.flatten())
            ]
            features_stain2 = [
                np.mean(stain_2.flatten()),
                np.std(stain_2.flatten()),
                kurtosis(stain_2.flatten()),
                skew(stain_2.flatten())
            ]
            
            # Concatenate the two sets so that the first half is for stain 1 and the second half is for stain 2.
            return features_stain1 + features_stain2
        else:
            # Apply stain deconvolution
            stain_1,stain_2 = get_stains_deconvoluted(img,I_0=estimate_I_0(img))

            # Compute the rgb images using the stains as masks
            mask_stain_1,mask_stain_2 = threshold_stain(stain_1),threshold_stain(stain_2)
            img_stain_1 = np.where(mask_stain_1,img,0)
            img_stain_2 = np.where(mask_stain_2,img,0)

            # Convert to the appropriate color space
            img_stain_1,img_stain_2 = mode_conversion(mode,img_stain_1,background_mask),mode_conversion(mode,img_stain_2,background_mask)

            # Compute features for both stains (should i return separately or concatenated?) (shape: feature[0]->stain1, feature[1]->stain2)
            return list(np.concatenate((
                np.mean(img_stain_1, axis=0),
                np.std(img_stain_1, axis=0),
                kurtosis(img_stain_1, axis=0),
                skew(img_stain_1, axis=0),
                np.mean(img_stain_2, axis=0),
                np.std(img_stain_2, axis=0),
                kurtosis(img_stain_2, axis=0),
                skew(img_stain_2, axis=0)
            )))
        
def calculate_avg_std(img: np.ndarray, mode: str = 'lab') -> tuple:
    """
    Calculate the average and standard deviation of an image in the specified color mode.

    Args:
        img (np.ndarray): Input RGB image.
        mode (str): Color mode ('lab', 'rgb', 'hsv', 'hsi').

    Returns:
        tuple: A tuple containing two lists (avg, std) for the mean and standard deviation of each channel.
    """
    # Code adapdted from https://github.com/yiqings/RandStainNA

    # Generate the white mask
    white_mask = get_background_mask(img)

    if mode is not None:
        # Convert the image to the specified color mode (excluding background)
        img = mode_conversion(mode, img, white_mask)

    # Calculate the average and standard deviation of each channel
    avg = np.mean(img, axis=0).tolist()
    std = np.std(img, axis=0).tolist()

    return (avg, std)

def quick_loop_extended(image, image_avg, image_std, temp_avg, temp_std, is_hed=False):
    """
    Applies stain normalization by transferring statistical properties from one image to another.

    Args:
        image (np.ndarray): Input image to be normalized.
        image_avg (np.ndarray): Mean values of the source image.
        image_std (np.ndarray): Standard deviation values of the source image.
        temp_avg (np.ndarray): Mean values of the target image.
        temp_std (np.ndarray): Standard deviation values of the target image.
        is_hed (bool): If True, assumes image is in HED space (range [0,1]), otherwise LAB/HSV (range [0,255]).

    Returns:
        np.ndarray: Stain-normalized image.
    """


    # Copy input image to avoid modifying original data
    original_image = image.copy()

    # Normalize stain characteristics
    transferred_image = (np.asarray(image) - np.asarray(image_avg)) * (np.asarray(temp_std) / np.asarray(image_std)) + np.asarray(temp_avg)

    # Preserve white background
    white_mask = get_background_mask(original_image)
    transferred_image[white_mask == 1] = original_image[white_mask == 1]

    # Clip values based on color space
    if not is_hed:  # LAB/HSV in range [0,255]
        transferred_image = np.clip(transferred_image, 0, 255).astype(np.uint8)

    return transferred_image

def transfer_image_style_extended(source, target):
    """
    Transfers the stain style from the target image to the source image.

    Args:
        source (np.ndarray or PIL.Image): Source image.
        target (np.ndarray or PIL.Image): Target image.

    Returns:
        np.ndarray: Stain-normalized source image.
    """
    # Convert input images to numpy arrays if necessary
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    # Compute mean and standard deviation for source and target
    source_avg, source_std = calculate_avg_std(source)
    target_avg, target_std = calculate_avg_std(target)

    # Determine if the source image is in HED space (range [0,1]) or LAB/HSV (range [0,255])
    is_hed = np.max(source) <= 1.0

    # Apply stain normalization
    return quick_loop_extended(source, source_avg, source_std, target_avg, target_std, is_hed)
