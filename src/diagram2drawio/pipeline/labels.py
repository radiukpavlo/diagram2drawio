"""Label assignment module for diagram2drawio."""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from diagram2drawio.config import DiagramConfig
from diagram2drawio.ir import EdgeIR, NodeIR, TextIR

logger = logging.getLogger(__name__)


def point_in_rect(point: tuple, rect: tuple) -> bool:
    """Check if a point is inside a rectangle."""
    px, py = point
    rx, ry, rw, rh = rect
    return rx <= px <= (rx + rw) and ry <= py <= (ry + rh)


def rect_overlap(rect1: tuple, rect2: tuple) -> float:
    """Calculate overlap ratio between two rectangles."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate intersection
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    area1 = w1 * h1

    return intersection / area1 if area1 > 0 else 0.0


def distance_to_polyline(point: tuple, polyline: List[tuple]) -> float:
    """Calculate minimum distance from a point to a polyline."""
    if len(polyline) < 2:
        if polyline:
            dx = point[0] - polyline[0][0]
            dy = point[1] - polyline[0][1]
            return (dx * dx + dy * dy) ** 0.5
        return float("inf")

    min_dist = float("inf")

    for i in range(len(polyline) - 1):
        p1 = np.array(polyline[i])
        p2 = np.array(polyline[i + 1])
        pt = np.array(point)

        # Vector from p1 to p2
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            dist = np.linalg.norm(pt - p1)
        else:
            # Project point onto line segment
            t = max(0, min(1, np.dot(pt - p1, line_vec) / (line_len * line_len)))
            projection = p1 + t * line_vec
            dist = np.linalg.norm(pt - projection)

        min_dist = min(min_dist, dist)

    return min_dist


def assign_labels(
    texts: List[TextIR],
    nodes: List[NodeIR],
    edges: List[EdgeIR],
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
    original_image: Optional[np.ndarray] = None,
) -> tuple:
    """
    Assign text labels to nodes and edges.

    Args:
        texts: List of detected text regions.
        nodes: List of detected nodes.
        edges: List of detected edges.
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.
        original_image: Optional original image for debug overlay.

    Returns:
        Tuple of (updated_nodes, updated_edges, updated_texts).
    """
    assigned_texts = []

    for text in texts:
        text_center = (
            text.bbox[0] + text.bbox[2] / 2,
            text.bbox[1] + text.bbox[3] / 2,
        )

        best_node = None
        best_overlap = 0.0

        # Check overlap with nodes
        for node in nodes:
            node_rect = (node.x, node.y, node.w, node.h)
            overlap = rect_overlap(text.bbox, node_rect)

            if overlap > best_overlap:
                best_overlap = overlap
                best_node = node

        # If significant overlap, assign to node
        if best_overlap > 0.3 and best_node:
            if best_node.label:
                best_node.label += " " + text.text
            else:
                best_node.label = text.text

            text.assigned_to = "node"
            text.assigned_id = best_node.id
            assigned_texts.append(text)
            continue

        # Otherwise, check proximity to edges
        best_edge = None
        best_dist = 50.0  # Max distance for edge labels

        for edge in edges:
            if not edge.points:
                continue

            dist = distance_to_polyline(text_center, edge.points)
            if dist < best_dist:
                best_dist = dist
                best_edge = edge

        if best_edge:
            if best_edge.label:
                best_edge.label += " " + text.text
            else:
                best_edge.label = text.text

            text.assigned_to = "edge"
            text.assigned_id = best_edge.id
            assigned_texts.append(text)

    node_labels = sum(1 for n in nodes if n.label)
    edge_labels = sum(1 for e in edges if e.label)
    logger.info(f"Assigned {node_labels} node labels, {edge_labels} edge labels")

    # Save debug artifacts
    if debug_dir and original_image is not None:
        overlay = original_image.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        for text in texts:
            x, y, w, h = [int(v) for v in text.bbox]
            if text.assigned_to == "node":
                color = (0, 255, 0)
            elif text.assigned_to == "edge":
                color = (255, 0, 0)
            else:
                color = (128, 128, 128)

            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        cv2.imwrite(str(debug_dir / "18_labels_overlay.png"), overlay)

    return nodes, edges, texts
