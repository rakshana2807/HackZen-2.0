import pytest
import numpy as np
import cv2
from vision.enhancement import (
    validate_image,
    apply_gamma_correction,
    apply_clahe,
    apply_low_light_pipeline,
    apply_simulated_fog
)


def test_validate_image_valid():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    validated = validate_image(img)
    assert validated.shape == (100, 100, 3)
    assert validated.dtype == np.uint8


def test_validate_image_invalid():
    with pytest.raises(ValueError, match="Input image is None"):
        validate_image(None)

    with pytest.raises(ValueError, match="Expected numpy.ndarray"):
        validate_image("not an array")  # type: ignore

    with pytest.raises(ValueError, match="zero dimensions"):
        validate_image(np.array([]))


def test_gamma_correction_output_validity():
    # Constant mid-gray image
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    enhanced = apply_gamma_correction(img, gamma=2.0)

    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8
    # Gamma > 1 should brighten the image
    assert enhanced.mean() > img.mean()

    # Gamma < 1 should darken
    darkened = apply_gamma_correction(img, gamma=0.5)
    assert darkened.mean() < img.mean()


def test_gamma_invalid_parameter():
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="Gamma must be strictly positive"):
        apply_gamma_correction(img, gamma=0.0)
    with pytest.raises(ValueError, match="Gamma must be strictly positive"):
        apply_gamma_correction(img, gamma=-1.5)


def test_clahe_dimensions_and_type():
    # 3-channel BGR test
    img_bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    out_bgr = apply_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(4, 4))
    assert out_bgr.shape == img_bgr.shape
    assert out_bgr.dtype == np.uint8

    # 1-channel Grayscale test
    img_gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    out_gray = apply_clahe(img_gray, clip_limit=2.0, tile_grid_size=(4, 4))
    assert out_gray.shape == img_gray.shape
    assert out_gray.dtype == np.uint8


def test_low_light_pipeline():
    dark_scene = np.full((80, 80, 3), 25, dtype=np.uint8)
    enhanced = apply_low_light_pipeline(dark_scene, gamma=1.5, clip_limit=2.0, denoise=True)
    assert enhanced.shape == dark_scene.shape
    assert enhanced.dtype == np.uint8
    # Enhanced low-light image should have higher mean luminance than raw dark scene
    assert enhanced.mean() > dark_scene.mean()


def test_simulated_fog():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    foggy = apply_simulated_fog(img, fog_density=0.6, fog_brightness=200)
    assert foggy.shape == img.shape
    assert foggy.dtype == np.uint8
    # Fog adds bright veil, so mean luminance increases
    assert foggy.mean() > img.mean()
