"""Pipeline package for diagram2drawio."""

from diagram2drawio.pipeline.preprocess import load_image, binarize
from diagram2drawio.pipeline.ocr import run_ocr, OcrBackend, EasyOcrBackend
from diagram2drawio.pipeline.text_mask import create_text_mask, remove_text
from diagram2drawio.pipeline.shapes import detect_shapes
from diagram2drawio.pipeline.connectors import detect_connectors
from diagram2drawio.pipeline.arrowheads import detect_arrowheads
from diagram2drawio.pipeline.associate import associate_edges_to_nodes
from diagram2drawio.pipeline.labels import assign_labels

__all__ = [
    "load_image",
    "binarize",
    "run_ocr",
    "OcrBackend",
    "EasyOcrBackend",
    "create_text_mask",
    "remove_text",
    "detect_shapes",
    "detect_connectors",
    "detect_arrowheads",
    "associate_edges_to_nodes",
    "assign_labels",
]
