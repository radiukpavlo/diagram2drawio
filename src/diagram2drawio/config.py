"""Configuration models for diagram2drawio."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DiagramConfig(BaseModel):
    """Configuration for the diagram conversion pipeline."""

    # I/O paths
    input_path: Path = Field(..., description="Path to the input image file")
    output_path: Path = Field(..., description="Path for the output .drawio file")
    debug_dir: Optional[Path] = Field(
        default=None, description="Directory to save debug artifacts"
    )

    # Processing toggles
    enable_ocr: bool = Field(default=True, description="Enable OCR text extraction")
    enable_arrowheads: bool = Field(default=True, description="Enable arrowhead detection")

    # Image preprocessing thresholds
    max_width: int = Field(default=3000, description="Max width before downscaling")
    binarize_threshold_block_size: int = Field(
        default=21, description="Block size for adaptive threshold (must be odd)"
    )
    binarize_c: int = Field(default=5, description="Constant subtracted from threshold")
    denoise_strength: int = Field(default=3, description="Median blur kernel size")

    # Shape detection thresholds
    min_shape_area: int = Field(default=100, description="Minimum contour area in pixels")
    epsilon_approx_poly_factor: float = Field(
        default=0.04, description="Factor for polygon approximation"
    )
    circularity_threshold: float = Field(
        default=0.85, description="Threshold for ellipse detection"
    )

    # OCR thresholds
    ocr_confidence_threshold: float = Field(
        default=0.4, description="Minimum OCR confidence to keep text"
    )
    ocr_padding: int = Field(default=5, description="Padding around OCR boxes for masking")

    # Connector/edge thresholds
    snap_distance_threshold: int = Field(
        default=15, description="Max distance for snapping edges to nodes"
    )
    min_connector_length: int = Field(
        default=20, description="Minimum length for a valid connector"
    )

    # Arrowhead detection
    arrowhead_max_area: int = Field(
        default=500, description="Maximum area for arrowhead contour"
    )
    arrowhead_min_area: int = Field(default=30, description="Minimum area for arrowhead contour")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
