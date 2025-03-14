import numpy as np


def normalize_data(x, y):
    """Normalizes x and y values to the range [0,1]."""
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    return x_norm, y_norm

def find_elbow_kneedle(x, y):
    """
    Detects the elbow point using the Kneedle method.

    Args:
        x (np.ndarray): X values.
        y (np.ndarray): Y values.

    Returns:
        int: Index of the detected elbow point.
    """
    x_norm, y_norm = normalize_data(x, y)
    distances = y_norm - (x_norm + (1 - x_norm))
    return np.argmax(np.abs(distances))

def find_elbow_second_derivative(x, y):
    """
    Detects the elbow point using the second derivative method.

    Args:
        x (np.ndarray): X values.
        y (np.ndarray): Y values.

    Returns:
        int: Index of the detected elbow point.
    """
    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    return np.argmax(np.abs(second_derivative))

def find_elbow_projection(x, y):
    """
    Detects the elbow point using the perpendicular projection method.

    Args:
        x (np.ndarray): X values.
        y (np.ndarray): Y values.

    Returns:
        int: Index of the detected elbow point.
    """
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line_vec = end - start

    distances = [
        np.linalg.norm(np.array([x[i], y[i]]) - (start + (np.dot(np.array([x[i], y[i]]) - start, line_vec) / np.dot(line_vec, line_vec)) * line_vec))
        for i in range(len(x))
    ]
    return np.argmax(distances)

def find_elbow(x, y, method="projection"):
    """
    Determines the elbow point using the specified method.

    Args:
        x (np.ndarray): X values.
        y (np.ndarray): Y values.
        method (str): Method to use ('kneedle', 'second_derivative', 'projection').

    Returns:
        int: Index of the detected elbow point.
    """
    methods = {
        "kneedle": find_elbow_kneedle,
        "second_derivative": find_elbow_second_derivative,
        "projection": find_elbow_projection,
    }

    if method not in methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {list(methods.keys())}.")

    return methods[method](x, y)
