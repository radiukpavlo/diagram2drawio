"""Shape detection module for diagram2drawio."""

import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from diagram2drawio.config import DiagramConfig
from diagram2drawio.ir import NodeIR, ShapeType

logger = logging.getLogger(__name__)


def classify_shape(contour: np.ndarray, config: DiagramConfig) -> Tuple[ShapeType, float]:
    """
    Classify a contour into a shape type.

    Args:
        contour: OpenCV contour.
        config: Pipeline configuration.

    Returns:
        Tuple of (ShapeType, confidence).
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return ShapeType.UNKNOWN, 0.0

    # Approximate polygon
    epsilon = config.epsilon_approx_poly_factor * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    area = cv2.contourArea(contour)
    if area == 0:
        return ShapeType.UNKNOWN, 0.0

    # Calculate circularity: 4 * pi * area / perimeter^2
    circularity = (4 * math.pi * area) / (perimeter * perimeter)

    # Bounding rect for aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0

    # Classification logic
    if circularity > config.circularity_threshold:
        return ShapeType.ELLIPSE, circularity

    if num_vertices == 4:
        # Check if it's a diamond (rotated square) or rectangle
        # For diamond, check if vertices are near edges, not corners
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]

        if rect_area > 0:
            # Rotated rectangle fit
            box = cv2.boxPoints(rect)
            box_perimeter = cv2.arcLength(box.astype(np.float32), True)

            # Check rotation angle
            angle = abs(rect[2])
            if 35 < angle < 55:
                # Rotated ~45 degrees - likely a diamond
                return ShapeType.DIAMOND, 0.85

        # Check corner angles for rectangle vs diamond
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % 4][0]
            p3 = approx[(i + 2) % 4][0]

            v1 = p1 - p2
            v2 = p3 - p2
            dot = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)

            if mag1 > 0 and mag2 > 0:
                cos_angle = dot / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = math.degrees(math.acos(cos_angle))
                angles.append(angle_deg)

        if angles:
            avg_angle = sum(angles) / len(angles)
            # If angles are around 90 degrees, it's a rectangle
            if 80 < avg_angle < 100:
                return ShapeType.RECTANGLE, 0.9
            else:
                return ShapeType.DIAMOND, 0.8

        return ShapeType.RECTANGLE, 0.7

    if num_vertices > 8:
        # Many vertices often means rounded rectangle or ellipse
        if circularity > 0.7:
            return ShapeType.ELLIPSE, circularity

        # Check if it looks like a rounded rectangle
        # Rounded rectangles have high extent
        if extent > 0.85:
            return ShapeType.ROUNDED_RECTANGLE, 0.8

    # Check for rounded rectangle by analyzing convexity
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area > 0:
        solidity = area / hull_area
        if solidity > 0.9 and extent > 0.8:
            return ShapeType.ROUNDED_RECTANGLE, solidity

    return ShapeType.RECTANGLE, 0.5


def detect_shapes(
    binary_image: np.ndarray,
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> List[NodeIR]:
    """
    Detect shapes in a binary image.

    Args:
        binary_image: Binary input image (255 = ink).
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        List of detected NodeIR objects.
    """
    # Find contours
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    logger.info(f"Found {len(contours)} raw contours")

    nodes = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by minimum area
        if area < config.min_shape_area:
            continue

        # Get bounding rect
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very thin contours (likely lines, not shapes)
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else float("inf")
        if aspect > 10:
            logger.debug(f"Filtered thin contour with aspect ratio {aspect:.1f}")
            continue

        # Classify shape
        shape_type, confidence = classify_shape(contour, config)

        # Extract polygon points
        polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]

        node = NodeIR(
            shape=shape_type,
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            polygon=polygon,
            confidence=confidence,
        )
        nodes.append(node)

    logger.info(f"Detected {len(nodes)} shapes")

    # Save debug artifacts
    if debug_dir:
        # Save JSON
        nodes_data = [
            {
                "id": n.id,
                "shape": n.shape.value,
                "bbox": [n.x, n.y, n.w, n.h],
                "confidence": n.confidence,
            }
            for n in nodes
        ]
        with open(debug_dir / "09_nodes.json", "w") as f:
            json.dump(nodes_data, f, indent=2)

        # Save overlay
        overlay = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        colors = {
            ShapeType.RECTANGLE: (0, 255, 0),
            ShapeType.ROUNDED_RECTANGLE: (0, 255, 255),
            ShapeType.ELLIPSE: (255, 0, 0),
            ShapeType.DIAMOND: (255, 0, 255),
            ShapeType.UNKNOWN: (128, 128, 128),
        }

        for node in nodes:
            color = colors.get(node.shape, (255, 255, 255))
            cv2.rectangle(
                overlay,
                (int(node.x), int(node.y)),
                (int(node.x + node.w), int(node.y + node.h)),
                color,
                2,
            )
            cv2.putText(
                overlay,
                f"{node.shape.value[:3]}",
                (int(node.x), int(node.y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        cv2.imwrite(str(debug_dir / "10_nodes_overlay.png"), overlay)

    return nodes
