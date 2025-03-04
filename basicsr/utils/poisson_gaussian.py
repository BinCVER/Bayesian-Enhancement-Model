import torch

# Define the Poisson-Gaussian noise addition function
def add_poisson_gaussian_noise(input_image, poisson_scale=500.0, gaussian_std=0.01):
    """
    Add Poisson-Gaussian noise to the input image.

    Args:
        input_image: Input image tensor with shape (B, C, H, W).
        poisson_scale: Scaling factor for Poisson noise; higher values result in stronger Poisson noise.
        gaussian_std: Standard deviation of the Gaussian noise.

    Returns:
        Noisy image tensor.
    """
    # Normalize the input image to a non-negative range (Poisson distribution requires non-negative values)
    normalized_image = input_image - input_image.min()
    normalized_image = normalized_image / (normalized_image.max() + 1e-8)  # Prevent division by zero

    # Add Poisson noise (generate a Poisson-distributed random value for each pixel)
    poisson_noise = torch.poisson(normalized_image * poisson_scale) / poisson_scale

    # Add Gaussian noise
    gaussian_noise = torch.randn_like(input_image) * gaussian_std

    # Combine the noise
    noisy_image = poisson_noise + gaussian_noise

    return noisy_image
