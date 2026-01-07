"""Text masking module for diagram2drawio."""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from diagram2drawio.config import DiagramConfig
from diagram2drawio.ir import TextIR

logger = logging.getLogger(__name__)


def create_text_mask(
    image_shape: tuple,
    texts: List[TextIR],
    config: DiagramConfig,
) -> np.ndarray:
    """
    Create a binary mask covering all text regions.

    Args:
        image_shape: Shape of the image (height, width).
        texts: List of detected text regions.
        config: Pipeline configuration.

    Returns:
        Binary mask (255 = text region, 0 = background).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for text in texts:
        x, y, tw, th = text.bbox
        # Apply padding
        x1 = max(0, int(x) - config.ocr_padding)
        y1 = max(0, int(y) - config.ocr_padding)
        x2 = min(w, int(x + tw) + config.ocr_padding)
        y2 = min(h, int(y + th) + config.ocr_padding)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Dilate to cover anti-aliasing artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def remove_text(
    binary_image: np.ndarray,
    texts: List[TextIR],
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Remove text regions from a binary image.

    This prevents text characters from being detected as shapes.

    Args:
        binary_image: Binary input image (255 = ink).
        texts: List of detected text regions.
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        Binary image with text regions removed.
    """
    if not texts:
        logger.info("No text regions to remove")
        return binary_image.copy()

    # Create text mask
    mask = create_text_mask(binary_image.shape, texts, config)

    if debug_dir:
        cv2.imwrite(str(debug_dir / "07_text_mask.png"), mask)

    # Remove text pixels by setting them to background (0)
    result = binary_image.copy()
    result[mask > 0] = 0

    if debug_dir:
        cv2.imwrite(str(debug_dir / "08_binary_no_text.png"), result)

    logger.info(f"Removed {len(texts)} text regions from binary image")

    return result
