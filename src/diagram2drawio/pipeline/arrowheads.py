"""Arrowhead detection module for diagram2drawio."""

import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from diagram2drawio.config import DiagramConfig
from diagram2drawio.ir import EdgeIR

logger = logging.getLogger(__name__)


def detect_arrowhead_at_point(
    binary_image: np.ndarray,
    point: Tuple[float, float],
    config: DiagramConfig,
    search_radius: int = 20,
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    Detect if there's an arrowhead near a point.

    Args:
        binary_image: Binary image.
        point: (x, y) point to search near.
        config: Pipeline configuration.
        search_radius: Radius to search around the point.

    Returns:
        Tuple of (centroid, direction_angle) if found, None otherwise.
    """
    h, w = binary_image.shape[:2]
    x, y = int(point[0]), int(point[1])

    # Extract region of interest
    x1 = max(0, x - search_radius)
    y1 = max(0, y - search_radius)
    x2 = min(w, x + search_radius)
    y2 = min(h, y + search_radius)

    roi = binary_image[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Find contours in the ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Check area bounds for arrowhead
        if not (config.arrowhead_min_area <= area <= config.arrowhead_max_area):
            continue

        # Approximate to polygon
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)

        # Arrowheads are typically triangular (3 vertices) or have a sharp tip
        if 3 <= len(approx) <= 5:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1

                # Compute direction from point to centroid
                dx = cx - point[0]
                dy = cy - point[1]
                angle = math.atan2(dy, dx)

                return ((cx, cy), angle)

    return None


def detect_arrowheads(
    binary_image: np.ndarray,
    edges: List[EdgeIR],
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> List[EdgeIR]:
    """
    Detect arrowheads and update edge directions.

    Args:
        binary_image: Binary input image.
        edges: List of detected edges.
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        Updated list of edges with arrowhead information.
    """
    if not config.enable_arrowheads:
        logger.info("Arrowhead detection disabled")
        return edges

    arrowhead_info = []

    for edge in edges:
        if len(edge.points) < 2:
            continue

        # Check start point
        start = edge.points[0]
        start_arrow = detect_arrowhead_at_point(binary_image, start, config)

        # Check end point
        end = edge.points[-1]
        end_arrow = detect_arrowhead_at_point(binary_image, end, config)

        if start_arrow:
            edge.has_arrowhead_source = True
            arrowhead_info.append({"edge_id": edge.id, "end": "source", "pos": start_arrow[0]})
        else:
            edge.has_arrowhead_source = False

        if end_arrow:
            edge.has_arrowhead_target = True
            arrowhead_info.append({"edge_id": edge.id, "end": "target", "pos": end_arrow[0]})
        else:
            edge.has_arrowhead_target = False

    detected_count = sum(1 for e in edges if e.has_arrowhead_source or e.has_arrowhead_target)
    logger.info(f"Detected arrowheads on {detected_count}/{len(edges)} edges")

    # Save debug artifacts
    if debug_dir:
        with open(debug_dir / "14_arrowheads.json", "w") as f:
            json.dump(arrowhead_info, f, indent=2)

        # Save overlay
        overlay = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        for edge in edges:
            if len(edge.points) >= 2:
                if edge.has_arrowhead_source:
                    cv2.circle(
                        overlay,
                        (int(edge.points[0][0]), int(edge.points[0][1])),
                        8,
                        (255, 0, 255),
                        2,
                    )
                if edge.has_arrowhead_target:
                    cv2.circle(
                        overlay,
                        (int(edge.points[-1][0]), int(edge.points[-1][1])),
                        8,
                        (0, 255, 255),
                        2,
                    )

        cv2.imwrite(str(debug_dir / "15_arrowheads_overlay.png"), overlay)

    return edges
