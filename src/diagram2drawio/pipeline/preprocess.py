"""Image preprocessing module for diagram2drawio."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from diagram2drawio.config import DiagramConfig

logger = logging.getLogger(__name__)


def load_image(
    image_path: Path,
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load and normalize an image for processing.

    Args:
        image_path: Path to the input image file.
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        Tuple of (original_bgr, grayscale, scale_factor).

    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If image cannot be read.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image in BGR format
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Failed to read image: {image_path}")

    logger.info(f"Loaded image: {original.shape[1]}x{original.shape[0]}")

    # Save debug artifact
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_input.png"), original)

    # Normalize resolution if needed
    scale_factor = 1.0
    h, w = original.shape[:2]

    if w > config.max_width:
        scale_factor = config.max_width / w
        new_w = config.max_width
        new_h = int(h * scale_factor)
        original = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Downscaled to: {new_w}x{new_h} (scale={scale_factor:.3f})")

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "01_norm.png"), original)
        cv2.imwrite(str(debug_dir / "02_gray.png"), gray)

    return original, gray, scale_factor


def binarize(
    gray_image: np.ndarray,
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Binarize a grayscale image for shape detection.

    Args:
        gray_image: Grayscale input image.
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        Binary image (0 = background, 255 = ink/foreground).
    """
    # Step 1: Denoise with median blur
    if config.denoise_strength > 0:
        denoised = cv2.medianBlur(gray_image, config.denoise_strength)
    else:
        denoised = gray_image.copy()

    # Step 2: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.binarize_threshold_block_size,
        config.binarize_c,
    )

    # Step 3: Morphological operations to clean up
    # Opening removes small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Closing fills small gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "03_binary.png"), binary)
        cv2.imwrite(str(debug_dir / "04_binary_clean.png"), cleaned)

    return cleaned
