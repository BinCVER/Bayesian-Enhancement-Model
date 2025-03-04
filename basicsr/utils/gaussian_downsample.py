import torch
import torch.nn.functional as F
import torch.fft

# def gaussian_blur(image, kernel_size, sigma, device='cuda'):
#     """
#     Apply Gaussian blur to an image.
#
#     Args:
#         image: Input image tensor with shape (B, C, H, W).
#         kernel_size: Size of the Gaussian kernel.
#         sigma: Standard deviation of the Gaussian blur.
#         device: Device to perform computation (default: 'cuda').
#
#     Returns:
#         Blurred image tensor with shape (B, C, H, W).
#     """
#     B, C, H, W = image.shape
#
#     # Generate a 1D Gaussian kernel
#     grid = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
#     gaussian_kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
#     gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # Normalize
#
#     # Construct a 2D Gaussian kernel
#     gaussian_kernel_2d = gaussian_kernel[:, None] * gaussian_kernel[None, :]
#     gaussian_kernel_2d = gaussian_kernel_2d[None, None, :, :].repeat(C, 1, 1, 1)
#     gaussian_kernel_2d = gaussian_kernel_2d.to(device)  # Move to specified device
#
#     # Apply Gaussian blur
#     padding = kernel_size // 2
#     padded_image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
#     blurred = F.conv2d(padded_image, gaussian_kernel_2d, padding=0, groups=C)
#     # Alternative method:
#     # blurred = F.conv2d(image, gaussian_kernel_2d, padding=kernel_size // 2, groups=C)
#
#     return blurred


def fft_gaussian_blur(image, sigma):
    """
    Apply Gaussian blur using Fast Fourier Transform (FFT).

    Args:
        image: Input image tensor with shape (B, C, H, W).
        sigma: Standard deviation of the Gaussian blur.

    Returns:
        Blurred image tensor with shape (B, C, H, W).
    """
    device = image.device
    B, C, H, W = image.shape

    # Construct the Gaussian kernel
    x = torch.linspace(-(W // 2), W // 2 - 1, W, device=device)  # Ensure symmetry
    y = torch.linspace(-(H // 2), H // 2 - 1, H, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    gaussian_kernel = torch.exp(-0.5 * (X**2 + Y**2) / sigma**2)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # Normalize

    # Perform frequency shift for center alignment
    gaussian_kernel = torch.fft.ifftshift(gaussian_kernel)

    # Compute the Fourier transform of the Gaussian kernel
    fft_kernel = torch.fft.fft2(gaussian_kernel, s=(H, W))  # (H, W)

    # Expand the Gaussian kernel to match the number of channels
    fft_kernel = fft_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    fft_kernel = fft_kernel.expand(B, C, -1, -1)       # (B, C, H, W)

    # Compute the Fourier transform of the input image
    fft_image = torch.fft.fft2(image)  # (B, C, H, W)

    # Multiply in the frequency domain
    fft_result = fft_image * fft_kernel  # (B, C, H, W)

    # Perform inverse Fourier transform
    blurred_image = torch.fft.ifft2(fft_result).real  # (B, C, H, W)

    return blurred_image

# Cache dictionary for Gaussian kernels
_gaussian_kernel_cache = {}

def get_gaussian_kernel(kernel_size, sigma, channels, device):
    """
    Retrieve the Gaussian kernel (supports caching).

    Args:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian blur.
        channels: Number of channels.
        device: Device where the tensor is stored.

    Returns:
        Gaussian kernel tensor with shape (C, 1, H, W).
    """
    global _gaussian_kernel_cache

    # Define the cache key
    key = (kernel_size, sigma, channels, device)
    if key in _gaussian_kernel_cache:
        return _gaussian_kernel_cache[key]

    # Generate a 1D Gaussian kernel
    grid = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=device)
    gaussian_kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Construct a 2D Gaussian kernel
    gaussian_kernel_2d = gaussian_kernel[:, None] * gaussian_kernel[None, :]
    gaussian_kernel_2d = gaussian_kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)

    # Cache the result
    _gaussian_kernel_cache[key] = gaussian_kernel_2d
    return gaussian_kernel_2d


def gaussian_blur(image, kernel_size, sigma):
    """
    Apply Gaussian blur to an image.

    Args:
        image: Input image tensor with shape (B, C, H, W).
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian blur.

    Returns:
        Blurred image tensor with shape (B, C, H, W).
    """
    device = image.device
    B, C, H, W = image.shape

    # Retrieve the Gaussian kernel (from cache)
    gaussian_kernel_2d = get_gaussian_kernel(kernel_size, sigma, C, device)

    # Apply Gaussian blur
    padding = kernel_size // 2
    padded_image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
    blurred = F.conv2d(padded_image, gaussian_kernel_2d, padding=0, groups=C)

    return blurred

def downsample_fft(image, scale_factor=0.125, kernel_size=17, sigma=4):
    """
    Perform downsampling with Gaussian blur using FFT.

    Args:
        image: Input image tensor with shape (B, C, H, W).
        kernel_size: (Deprecated) Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian blur.

    Returns:
        Downsampled image tensor with shape (B, C, H//8, W//8).
    """
    current_image = image
    # for _ in range(3):  # Perform 3 iterations of 2× downsampling
    #     current_image = fft_gaussian_blur(current_image, sigma)
    #     # blurred = gaussian_blur(current_image, kernel_size, sigma)
    #     current_image = F.interpolate(current_image, scale_factor=0.5, mode='bilinear', align_corners=False)

    current_image = fft_gaussian_blur(current_image, sigma)
    current_image = F.interpolate(current_image, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    return current_image
