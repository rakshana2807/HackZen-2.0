"""Image enhancement algorithms for road-scene visibility enhancement.

Functions:
- apply_gamma_correction: Non-linear luminance expansion.
- apply_clahe: Contrast Limited Adaptive Histogram Equalization on LAB L-channel.
- apply_low_light_pipeline: Composite pipeline combining bilateral noise reduction, gamma, and CLAHE.
- apply_simulated_fog: Atmospheric scattering simulation for demonstration testing only.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def validate_image(image: Optional[np.ndarray]) -> np.ndarray:
    """Validate that input is a valid non-empty uint8 numpy image.

    Raises:
        ValueError: If input is None, empty, or not a valid image array.
    """
    if image is None:
        raise ValueError("Input image is None.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(image)}.")
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image has zero dimensions.")
    if image.dtype != np.uint8:
        # Normalize or cast safely if needed, or raise
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            return (image * 255.0).astype(np.uint8)
        raise ValueError(f"Image dtype must be uint8, found {image.dtype}.")
    return image


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply gamma correction using a precomputed lookup table.

    Args:
        image: BGR or Grayscale uint8 image.
        gamma: Positive float. gamma > 1 brightens shadows; gamma < 1 darkens highlights.

    Returns:
        Enhanced copy of the image.
    """
    img = validate_image(image)
    if gamma <= 0:
        raise ValueError(f"Gamma must be strictly positive, got {gamma}")

    # Build lookup table mapping [0, 255] to corrected values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    For color images, the image is converted to LAB color space, CLAHE is applied
    only to the Lightness (L) channel, and the image is converted back to BGR.
    For grayscale images, CLAHE is applied directly.

    Args:
        image: Input uint8 image (BGR or 1-channel).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced copy of the image.
    """
    img = validate_image(image)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than 0.")
    if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        raise ValueError("tile_grid_size dimensions must be positive integers.")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        return clahe.apply(img)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        res = clahe.apply(img[:, :, 0])
        return np.expand_dims(res, axis=2)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Convert BGR to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        # BGRA support
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        enhanced_bgr = apply_clahe(bgr, clip_limit, tile_grid_size)
        return cv2.merge((enhanced_bgr, alpha))
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")


def apply_low_light_pipeline(
    image: np.ndarray,
    gamma: float = 1.4,
    clip_limit: float = 2.0,
    denoise: bool = True
) -> np.ndarray:
    """Composite enhancement pipeline designed for nighttime or under-exposed road scenes.

    Pipeline stages:
    1. Edge-preserving bilateral filter (optional) to suppress sensor noise in dark scenes.
    2. Non-linear gamma expansion to recover shadow detail.
    3. LAB-space CLAHE to enhance local feature contrast (road edges, signs, vehicles).

    Args:
        image: Input uint8 image.
        gamma: Gamma coefficient (> 1 brightens).
        clip_limit: CLAHE clip limit.
        denoise: Whether to run an edge-preserving filter before enhancement.

    Returns:
        Enhanced copy of the image.
    """
    img = validate_image(image)

    # 1. Bilateral filter preserves sharp edges while smoothing sensor noise in dark regions
    if denoise:
        if len(img.shape) == 3 and img.shape[2] == 3:
            filtered = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)
        else:
            filtered = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)
    else:
        filtered = img

    # 2. Gamma expansion
    gamma_corrected = apply_gamma_correction(filtered, gamma=gamma)

    # 3. CLAHE enhancement
    enhanced = apply_clahe(gamma_corrected, clip_limit=clip_limit, tile_grid_size=(8, 8))
    return enhanced


def apply_simulated_fog(
    image: np.ndarray,
    fog_density: float = 0.5,
    fog_brightness: int = 210
) -> np.ndarray:
    """Atmospheric scattering approximation for demonstration and testing purposes ONLY.

    Note: This is an artificial synthetic fog simulation and does NOT represent
    real environmental meteorological instrumentation.

    Formula:
        I_fog = I * (1 - beta) + AtmosphericLight * beta
    where beta is scaled by distance/depth approximation or uniform density.

    Args:
        image: Input uint8 image.
        fog_density: Fog intensity factor in [0.0, 1.0].
        fog_brightness: Grayscale intensity of atmospheric light (0-255).

    Returns:
        Simulated fog image.
    """
    img = validate_image(image)
    fog_density = max(0.0, min(1.0, float(fog_density)))
    if fog_density == 0.0:
        return img.copy()

    h, w = img.shape[:2]

    # Create a vertical depth gradient heuristic (horizon is foggy, foreground is clearer)
    # y = 0 is sky/horizon (foggiest), y = h is bumper/foreground
    y_coords = np.linspace(0.8, 0.2, h, dtype=np.float32).reshape(h, 1)
    # Atmospheric attenuation map beta(x, y)
    beta = y_coords * fog_density
    beta = np.clip(beta, 0.0, 0.95)

    if len(img.shape) == 3 and img.shape[2] == 3:
        beta = np.expand_dims(beta, axis=2)
        atmospheric_color = np.array([fog_brightness, fog_brightness, fog_brightness], dtype=np.float32)
        fogged = img.astype(np.float32) * (1.0 - beta) + atmospheric_color * beta
    else:
        fogged = img.astype(np.float32) * (1.0 - beta) + float(fog_brightness) * beta

    return np.clip(fogged, 0, 255).astype(np.uint8)
