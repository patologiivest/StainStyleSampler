# Standard Library Imports
import os
import glob
import random
import collections
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

STAIN_COLOR_MAP = {
    'hematoxylin': np.array([0.65, 0.70, 0.29], dtype=np.float64),
    'eosin': np.array([0.07, 0.99, 0.11], dtype=np.float64),
    'dab': np.array([0.27, 0.57, 0.78], dtype=np.float64),
    'null': np.array([0.0, 0.0, 0.0], dtype=np.float64),
}

_RGB_TO_LMS = np.array([
    [0.3811, 0.5783, 0.0402],
    [0.1967, 0.7244, 0.0782],
    [0.0241, 0.1288, 0.8444],
])
_LMS_TO_LAB = np.dot(
    np.array([
        [1 / (3 ** 0.5), 0, 0],
        [0, 1 / (6 ** 0.5), 0],
        [0, 0, 1 / (2 ** 0.5)],
    ]),
    np.array([
        [1, 1, 1],
        [1, 1, -2],
        [1, -1, 0],
    ]),
)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert RGB image data to HistomicsTK's Ruderman LAB space."""
    m, n = img.shape[:2]
    rgb = np.reshape(img, (m * n, 3))
    lms = np.dot(_RGB_TO_LMS, np.transpose(rgb))
    lms[lms == 0] = np.spacing(1)
    lab = np.dot(_LMS_TO_LAB, np.log(lms))
    return np.reshape(lab.transpose(), (m, n, 3))


def rgb_to_hsi(img: np.ndarray) -> np.ndarray:
    """Convert RGB image data to HistomicsTK's HSI representation."""
    img = np.moveaxis(img, -1, 0)
    if len(img) not in (3, 4):
        raise ValueError(
            'Expected 3-channel RGB or 4-channel RGBA image;'
            f' received a {len(img)}-channel image'
        )
    img = img[:3]
    hues = (
        np.arctan2(3 ** 0.5 * (img[1] - img[2]), 2 * img[0] - img[1] - img[2])
        / (2 * np.pi)
    ) % 1
    intensities = img.mean(0)
    saturations = np.where(
        intensities, 1 - img.min(0) / np.maximum(intensities, 1e-10), 0
    )
    return np.stack([hues, saturations, intensities], -1)


def convert_image_to_matrix(img: np.ndarray) -> np.ndarray:
    """Convert an image to HistomicsTK's channel-by-pixel matrix format."""
    if img.ndim == 2:
        return img
    return img.reshape((-1, img.shape[-1])).T


def convert_matrix_to_image(matrix: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert a channel-by-pixel matrix back to an image."""
    if len(shape) == 2:
        return matrix
    return matrix.T.reshape(shape[:-1] + (matrix.shape[0],))


def exclude_nonfinite(matrix: np.ndarray) -> np.ndarray:
    """Drop matrix columns containing NaN or infinite values."""
    return matrix[:, np.isfinite(matrix).all(axis=0)]


def threshold_multichannel(
        img: np.ndarray, thresholds: dict, channels: list = None,
        just_threshold: bool = False, get_tissue_mask_kwargs: dict = None) -> tuple:
    """Threshold a multi-channel image using HistomicsTK-compatible semantics."""
    channels = ['hue', 'saturation', 'intensity'] if channels is None else channels
    if get_tissue_mask_kwargs is None:
        get_tissue_mask_kwargs = {
            'n_thresholding_steps': 1,
            'sigma': 5.0,
            'min_size': 10,
        }

    mask = np.ones(img.shape[:2])
    for axis, channel_name in enumerate(channels):
        channel = img[..., axis].copy()
        mask[channel < thresholds[channel_name]['min']] = 0
        mask[channel >= thresholds[channel_name]['max']] = 0

    if just_threshold or (np.unique(mask).shape[0] < 1):
        labeled = mask
    else:
        get_tissue_mask_kwargs['deconvolve_first'] = False
        labeled, mask = get_tissue_mask(mask, **get_tissue_mask_kwargs)

    return labeled, mask


def get_tissue_mask(
        thumbnail_img: np.ndarray, deconvolve_first: bool = False,
        stain_unmixing_routine_kwargs: dict = None,
        n_thresholding_steps: int = 1, sigma: float = 0.0,
        min_size: int = 500) -> tuple:
    """Create a tissue mask from a thumbnail image."""
    from scipy import ndimage
    from skimage.filters import gaussian

    stain_unmixing_routine_kwargs = (
        {} if stain_unmixing_routine_kwargs is None else stain_unmixing_routine_kwargs
    )

    if deconvolve_first and (len(thumbnail_img.shape) == 3):
        stain_unmixing_routine_kwargs['stains'] = ['hematoxylin', 'eosin']
        stains, _, _ = color_deconvolution_routine(thumbnail_img, **stain_unmixing_routine_kwargs)
        thumbnail = 255 - stains[..., 0]
    elif len(thumbnail_img.shape) == 3:
        thumbnail = 255 - cv2.cvtColor(thumbnail_img, cv2.COLOR_BGR2GRAY)
    else:
        thumbnail = thumbnail_img

    for _ in range(n_thresholding_steps):
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail, sigma=sigma, output=None, mode='nearest', preserve_range=True
            )
        try:
            threshold = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:
            threshold = 0
        thumbnail[thumbnail < threshold] = 0

    mask = 0 + (thumbnail > 0)
    labeled, _ = ndimage.label(mask)
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    discard = np.in1d(labeled, unique[counts < min_size]).reshape(labeled.shape)
    labeled[discard] = 0
    mask = labeled == unique[np.argmax(counts)]
    return labeled, mask


def simple_mask(img: np.ndarray, bandwidth: float = 2, bgnd_std: float = 2.5,
                tissue_std: float = 30, min_peak_width: float = 10,
                max_peak_width: float = 25, fraction: float = 0.10,
                min_tissue_prob: float = 0.05) -> np.ndarray:
    """Segment foreground tissue with HistomicsTK's grayscale GMM approach."""
    from scipy import signal
    from scipy.optimize import fmin_slsqp
    from scipy.stats import norm
    from skimage import color
    from sklearn.neighbors import KernelDensity

    gray_img = (255 * color.rgb2gray(img)).astype(np.uint8)
    num_samples = int(fraction * gray_img.size)
    sampled_intensities = np.random.choice(gray_img.flatten(), num_samples)[:, np.newaxis]

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sampled_intensities)
    x_hist = np.linspace(0, 255, 256)[:, np.newaxis]
    y_hist = np.exp(kde.score_samples(x_hist))[:, np.newaxis]
    y_hist = y_hist / sum(y_hist)
    y_hist = np.flipud(y_hist)

    peaks = signal.find_peaks_cwt(y_hist.flatten(), np.arange(min_peak_width, max_peak_width))
    background_peak = peaks[0]
    if len(peaks) > 1:
        tissue_peak = peaks[y_hist[peaks[1:]].argmax() + 1]
    else:
        tissue_peak = x_hist[int(np.round(0.66 * x_hist.size))].item()

    background_scale = estimate_variance(x_hist, y_hist, background_peak)
    if background_scale == -1:
        background_scale = bgnd_std

    tissue_scale = estimate_variance(x_hist, y_hist, tissue_peak)
    if tissue_scale == -1:
        tissue_scale = tissue_std

    mix = y_hist[background_peak] * (background_scale * (2 * np.pi) ** 0.5)
    try:
        if len(mix) == 1:
            mix = mix[0]
    except Exception:
        pass

    x_hist = x_hist.flatten()
    y_hist = y_hist.flatten()

    def gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p):
        background = norm(loc=mu1, scale=sigma1)
        tissue = norm(loc=mu2, scale=sigma2)
        return p * background.pdf(x) + (1 - p) * tissue.pdf(x)

    def gaussian_residuals(parameters, y, x):
        mu1, mu2, sigma1, sigma2, p = parameters
        y_hat = gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p)
        return sum((y - y_hat) ** 2)

    parameters = fmin_slsqp(
        gaussian_residuals,
        [background_peak, tissue_peak, background_scale, tissue_scale, mix],
        args=(y_hist, x_hist),
        bounds=[(0, 255), (0, 255), (np.spacing(1), 10), (np.spacing(1), 50), (0, 1)],
        iprint=0,
    )

    mu_background, mu_tissue, sigma_background, sigma_tissue, p = parameters
    background = norm(loc=mu_background, scale=sigma_background)
    tissue = norm(loc=mu_tissue, scale=sigma_tissue)
    p_background = p * background.pdf(x_hist)
    p_tissue = (1 - p) * tissue.pdf(x_hist)

    difference = p_tissue - p_background
    candidates = np.nonzero(difference >= 0)[0]
    filtered = np.nonzero(x_hist[candidates] > mu_background)
    ml_threshold = x_hist[candidates[filtered[0]][0]]

    endpoints = np.asarray(tissue.interval(1 - min_tissue_prob / 2))
    ml_threshold = 255 - ml_threshold
    endpoints = np.sort(255 - endpoints)

    mask = (
        (gray_img <= ml_threshold)
        & (gray_img >= endpoints[0])
        & (gray_img <= endpoints[1])
    )
    return mask.astype(np.uint8)


def estimate_variance(x: np.ndarray, y: np.ndarray, peak: float) -> float:
    """Estimate a histogram peak's standard deviation from its FWHM."""
    peak = int(peak)
    left = peak
    while y[left] > y[peak] / 2 and left >= 0:
        left -= 1
        if left == -1:
            break

    right = peak
    while y[right] > y[peak] / 2 and right < y.size:
        right += 1
        if right == y.size:
            break

    if left != -1 and right != y.size:
        left_slope = y[left + 1] - y[left] / (x[left + 1] - x[left])
        left = (y[peak] / 2 - y[left]) / left_slope + x[left]
        right_slope = y[right] - y[right - 1] / (x[right] - x[right - 1])
        right = (y[peak] / 2 - y[right]) / right_slope + x[right]
        scale = (right - left) / 2.355
    if left == -1:
        if right == y.size:
            scale = -1
        else:
            right_slope = y[right] - y[right - 1] / (x[right] - x[right - 1])
            right = (y[peak] / 2 - y[right]) / right_slope + x[right]
            scale = 2 * (right - x[peak]) / 2.355
    if right == y.size:
        if left == -1:
            scale = -1
        else:
            left_slope = y[left + 1] - y[left] / (x[left + 1] - x[left])
            left = (y[peak] / 2 - y[left]) / left_slope + x[left]
            scale = 2 * (x[peak] - left) / 2.355

    try:
        if len(scale) == 1:
            scale = scale[0]
    except Exception:
        pass
    return scale


def _normalize_stain_matrix(w: np.ndarray) -> np.ndarray:
    return w / _stain_magnitude(w)


def _stain_magnitude(w: np.ndarray) -> np.ndarray:
    return np.sqrt((w ** 2).sum(0))


def _get_principal_components(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.svd(matrix.astype(float), full_matrices=False)[0].astype(matrix.dtype)


def _complement_stain_matrix(w: np.ndarray) -> np.ndarray:
    stain0 = w[:, 0]
    stain1 = w[:, 1]
    stain2 = np.cross(stain0, stain1)
    return np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T


def _get_angles(matrix: np.ndarray) -> np.ndarray:
    matrix = _normalize_stain_matrix(matrix)
    return (1 - matrix[1]) * np.sign(matrix[0])


def _argpercentile(values: np.ndarray, percentile: float) -> int:
    index = min(int(percentile * values.size + 0.5), values.size - 1)
    return np.argpartition(values, index)[index]


def rgb_to_sda(img: np.ndarray, I_0: int, allow_negatives: bool = False) -> np.ndarray:
    """Transform RGB data to HistomicsTK SDA space."""
    is_matrix = img.ndim == 2
    if is_matrix:
        img = img.T
    if I_0 is None:
        img = img.astype(float) + 1
        I_0 = 256

    img = np.maximum(img, 1e-10)
    sda = -np.log(img / (1.0 * I_0)) * 255 / np.log(I_0)
    if not allow_negatives:
        sda = np.maximum(sda, 0)
    return sda.T if is_matrix else sda


def sda_to_rgb(img: np.ndarray, I_0: int) -> np.ndarray:
    """Transform HistomicsTK SDA data back to RGB-like intensity space."""
    is_matrix = img.ndim == 2
    if is_matrix:
        img = img.T

    old_od_mode = I_0 is None
    if old_od_mode:
        I_0 = 256

    rgb = I_0 ** (1 - img / 255.0)
    return (rgb.T if is_matrix else rgb) - old_od_mode


def separate_stains_macenko_pca(
        img_sda: np.ndarray, minimum_magnitude: float = 16,
        min_angle_percentile: float = 0.01,
        max_angle_percentile: float = 0.99,
        mask_out: np.ndarray = None) -> np.ndarray:
    """Estimate stain matrix from an SDA image with HistomicsTK's Macenko PCA method."""
    matrix = convert_image_to_matrix(img_sda)

    if mask_out is not None:
        keep_mask = np.equal(mask_out[..., None], False)
        keep_mask = np.tile(keep_mask, (1, 1, 3))
        keep_mask = convert_image_to_matrix(keep_mask)
        matrix = matrix[:, keep_mask.all(axis=0)]

    matrix = exclude_nonfinite(matrix)
    principal_components = _get_principal_components(matrix)
    projected = principal_components.T[:-1].dot(matrix)
    filtered = projected[:, _stain_magnitude(projected) > minimum_magnitude]
    angles = _get_angles(filtered)

    def get_percentile_vector(percentile):
        return principal_components[:, :-1].dot(
            filtered[:, _argpercentile(angles, percentile)]
        )

    min_vector = get_percentile_vector(min_angle_percentile)
    max_vector = get_percentile_vector(max_angle_percentile)
    return _complement_stain_matrix(
        _normalize_stain_matrix(np.array([min_vector, max_vector]).T)
    )


def rgb_separate_stains_macenko_pca(img: np.ndarray, I_0: int, *args, **kwargs) -> np.ndarray:
    """Estimate a stain matrix from RGB input using HistomicsTK's Macenko PCA method."""
    return separate_stains_macenko_pca(rgb_to_sda(img, I_0), *args, **kwargs)


def find_stain_index(reference_stain: np.ndarray, w_est: np.ndarray) -> int:
    """Find the estimated stain column closest to a reference stain vector."""
    dot_products = np.dot(
        _normalize_stain_matrix(np.array(reference_stain)),
        _normalize_stain_matrix(np.array(w_est)),
    )
    return int(np.argmax(np.abs(dot_products)))


def color_deconvolution(img: np.ndarray, w_est: np.ndarray, I_0: int = None) -> tuple:
    """Perform HistomicsTK-compatible color deconvolution."""
    w_est = np.array(w_est)
    if w_est.shape[1] < 3:
        complemented = np.zeros((w_est.shape[0], 3))
        complemented[:, :w_est.shape[1]] = w_est
        w_est = complemented

    if np.linalg.norm(w_est[:, 2]) <= 1e-16:
        complemented = _complement_stain_matrix(w_est)
    else:
        complemented = w_est

    complemented = _normalize_stain_matrix(complemented)
    inverse = np.linalg.pinv(complemented)
    matrix = convert_image_to_matrix(img)[:3]
    sda_forward = rgb_to_sda(matrix, I_0)
    sda_deconvolved = np.dot(inverse, sda_forward)
    sda_inverse = sda_to_rgb(sda_deconvolved, 255 if I_0 is not None else None)
    stains_float = convert_matrix_to_image(sda_inverse, img.shape)
    stains = stains_float.clip(0, 255).astype(np.uint8)

    unmixed = collections.namedtuple('Unmixed', ['Stains', 'StainsFloat', 'Wc'])
    return unmixed(stains, stains_float, complemented)


def color_deconvolution_routine(img: np.ndarray, W_source: np.ndarray = None,
                                mask_out: np.ndarray = None, **kwargs) -> tuple:
    """Small local equivalent of HistomicsTK's deconvolution routine."""
    if W_source is None:
        W_source = stain_unmixing_routine(img, mask_out=mask_out, **kwargs)
    stains, stains_float, complemented = color_deconvolution(img, w_est=W_source, I_0=None)
    if mask_out is not None:
        for channel in range(3):
            stains[..., channel][mask_out] = 255
            stains_float[..., channel][mask_out] = 255.0
    return stains, stains_float, complemented


def stain_unmixing_routine(
        img: np.ndarray, stains: list = None,
        stain_unmixing_method: str = 'macenko_pca',
        stain_unmixing_params: dict = None,
        mask_out: np.ndarray = None) -> np.ndarray:
    """Estimate and order a stain matrix for color deconvolution."""
    stains = ['hematoxylin', 'eosin'] if stains is None else stains
    stain_unmixing_params = {} if stain_unmixing_params is None else stain_unmixing_params

    if stain_unmixing_method.lower() != 'macenko_pca':
        raise ValueError('Unknown/Unimplemented deconvolution method.')

    stain_unmixing_params['I_0'] = None
    stain_unmixing_params['mask_out'] = mask_out
    matrix = rgb_separate_stains_macenko_pca(img, **stain_unmixing_params)
    return _reorder_stains(matrix, stains=stains)


def _reorder_stains(w: np.ndarray, stains: list = None) -> np.ndarray:
    stains = ['hematoxylin', 'eosin'] if stains is None else stains
    assert len(stains) == 2, 'Only two-stain matrices are supported for now.'
    first = find_stain_index(STAIN_COLOR_MAP[stains[0]], w)
    second = 1 - first
    return np.stack([w[..., channel] for channel in (first, second, 2)], -1)

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
    def extract_stain(stain_name: str, w_est: np.ndarray, deconv_result: tuple) -> np.ndarray:
        """Extract a specific stain based on its name."""
        stain_index = find_stain_index(color_map[stain_name], w_est)
        return I_0 - deconv_result.Stains[:, :, stain_index]

    assert I_0 is None or I_0 == 'auto' or I_0 > 0, "I_0 must be a positive integer or 'auto'"

    # Define color map and stains
    color_map = STAIN_COLOR_MAP
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']         # for cases with only two stains

    if I_0 is None or I_0 == 'auto':
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
