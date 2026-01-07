"""OCR module for diagram2drawio."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from diagram2drawio.config import DiagramConfig
from diagram2drawio.ir import TextIR

logger = logging.getLogger(__name__)


class OcrBackend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def detect_text(
        self, image: np.ndarray, config: DiagramConfig
    ) -> List[Tuple[List[Tuple[int, int]], str, float]]:
        """
        Detect text in an image.

        Args:
            image: Input image (grayscale or BGR).
            config: Pipeline configuration.

        Returns:
            List of (polygon, text, confidence) tuples.
        """
        pass


class EasyOcrBackend(OcrBackend):
    """EasyOCR-based text detection backend."""

    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize EasyOCR reader.

        Args:
            languages: List of languages to detect. Defaults to English.
        """
        import easyocr

        self.languages = languages or ["en"]
        self.reader = easyocr.Reader(self.languages, gpu=False)
        logger.info(f"EasyOCR initialized with languages: {self.languages}")

    def detect_text(
        self, image: np.ndarray, config: DiagramConfig
    ) -> List[Tuple[List[Tuple[int, int]], str, float]]:
        """Detect text using EasyOCR."""
        results = self.reader.readtext(image)
        filtered = []

        for bbox, text, confidence in results:
            if confidence >= config.ocr_confidence_threshold:
                # Convert bbox to list of tuples
                polygon = [(int(p[0]), int(p[1])) for p in bbox]
                filtered.append((polygon, text, confidence))
            else:
                logger.debug(f"Filtered low-confidence text: '{text}' ({confidence:.2f})")

        return filtered


# Global singleton for OCR backend
_ocr_backend: Optional[OcrBackend] = None


def get_ocr_backend() -> OcrBackend:
    """Get or create the OCR backend singleton."""
    global _ocr_backend
    if _ocr_backend is None:
        _ocr_backend = EasyOcrBackend()
    return _ocr_backend


def run_ocr(
    image: np.ndarray,
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> List[TextIR]:
    """
    Run OCR on an image and return detected text regions.

    Args:
        image: Input image (grayscale preferred).
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        List of TextIR objects representing detected text.
    """
    if not config.enable_ocr:
        logger.info("OCR disabled, skipping")
        return []

    backend = get_ocr_backend()
    results = backend.detect_text(image, config)

    texts = []
    for polygon, text, confidence in results:
        # Compute bounding box from polygon
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y

        text_ir = TextIR(
            text=text,
            bbox=(float(x), float(y), float(w), float(h)),
            polygon=[(float(p[0]), float(p[1])) for p in polygon],
            confidence=confidence,
        )
        texts.append(text_ir)

    logger.info(f"OCR detected {len(texts)} text regions")

    # Save debug artifacts
    if debug_dir:
        import json

        # Save JSON
        ocr_data = [{"text": t.text, "bbox": t.bbox, "confidence": t.confidence} for t in texts]
        with open(debug_dir / "05_ocr.json", "w") as f:
            json.dump(ocr_data, f, indent=2)

        # Save overlay image
        overlay = image.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        for text_ir in texts:
            x, y, w, h = text_ir.bbox
            cv2.rectangle(
                overlay,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                overlay,
                text_ir.text[:20],
                (int(x), int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imwrite(str(debug_dir / "06_ocr_overlay.png"), overlay)

    return texts
