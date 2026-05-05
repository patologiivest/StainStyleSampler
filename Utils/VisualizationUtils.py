from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def save_or_plot(fig_func, save_path=None, plotting_graphs=True):
    """
    Handles saving or showing a plot based on user preferences.
    Args:
        fig_func: Function that generates the plot.
        save_path: File path to save the plot. If None, does not save.
        plotting_graphs: Whether to show the plot interactively.
    """
    fig_func()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    if plotting_graphs:
        plt.show()
    else:
        plt.close()
    return None

def display_reference_images(files, save_path=None, plotting_graphs=True, n_clusters=10):
    """
    Displays reference images in a grid format.

    Args:
        files (list of str): List of image file paths.
        save_path (str, optional): Path to save the figure (PDF format). Default is None.
        plotting_graphs (bool): Whether to display the images. Default is True.
        n_clusters (int): Number of images to display. Default is 10.

    Returns:
        None
    """
    #if not files:
    #    raise ValueError("No image files provided.")

    try:
        num_images = min(n_clusters, len(files))
        num_rows = (num_images // 5) + (1 if num_images % 5 else 0)  # Dynamically calculate rows

        fig, axes = plt.subplots(num_rows, 5, figsize=(25, 5 * num_rows))
        axes = axes.flatten()  # Flatten in case of a single row

        for i, ax in enumerate(axes[:num_images]):
            img = np.array(Image.open(files[i]))[:, :, :3]
            ax.imshow(img)
            ax.axis('off')

        # Hide unused subplots if images are fewer than grid slots
        for j in range(num_images, len(axes)):
            fig.delaxes(axes[j])

        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")

        if plotting_graphs:
            plt.show()
        else:
            plt.close()
    except:
        raise ValueError("No image files provided.")

def plot_image_groups(source_image, stain_converted_images, n_clusters):
    """
    Plots stain-converted images in a grouped grid layout with dynamic column count.

    Args:
        source_image (np.ndarray or PIL.Image.Image): The source image to display.
        stain_converted_images (dict): Dictionary of converted images where keys are file names
                                        and values are images (np.ndarray or PIL.Image.Image).
        n_clusters (int): Number of clusters (converted images).
    """
    if not stain_converted_images:
        raise ValueError("No stain-converted images provided.")

    references_per_group = min(5, n_clusters)  # Maximum 5 references per group
    num_groups = max(int(np.ceil(n_clusters / references_per_group)), 1)

    # Sort images by filename
    sorted_converted_images = sorted(stain_converted_images.items(), key=lambda x: x[0])

    for group_idx in range(num_groups):
        start_idx = group_idx * references_per_group
        end_idx = min(start_idx + references_per_group, n_clusters)
        current_images = sorted_converted_images[start_idx:end_idx]

        num_columns = len(current_images) + 1  # +1 for the source image
        fig, axes = plt.subplots(2, num_columns, figsize=(4 * num_columns, 10))

        # First row: reference images
        for col in range(num_columns):
            axes[0, col].axis('off')
            axes[1, col].axis('off')

        # Show source image in the second row, first column
        axes[1, 0].imshow(source_image)
        axes[1, 0].set_title("Source Image", fontsize=10)

        # Populate grid with reference & converted images
        for col, (file, image) in enumerate(current_images, start=1):
            ref_image = Image.open(file)
            axes[0, col].imshow(ref_image)
            axes[0, col].set_title(f"Reference {start_idx + col}", fontsize=10)

            axes[1, col].imshow(image)
            axes[1, col].set_title(f"Converted {start_idx + col}", fontsize=10)

        plt.tight_layout()
        plt.show()
