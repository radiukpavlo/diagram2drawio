"""Connector detection module for diagram2drawio."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from diagram2drawio.config import DiagramConfig
from diagram2drawio.ir import EdgeIR, NodeIR

logger = logging.getLogger(__name__)


def mask_nodes(
    binary_image: np.ndarray,
    nodes: List[NodeIR],
    margin: int = 5,
) -> np.ndarray:
    """
    Mask out node regions from a binary image.

    Args:
        binary_image: Binary input image.
        nodes: List of detected nodes to mask.
        margin: Additional margin around nodes.

    Returns:
        Binary image with node regions set to 0.
    """
    result = binary_image.copy()
    h, w = result.shape[:2]

    for node in nodes:
        x1 = max(0, int(node.x) - margin)
        y1 = max(0, int(node.y) - margin)
        x2 = min(w, int(node.x + node.w) + margin)
        y2 = min(h, int(node.y + node.h) + margin)
        result[y1:y2, x1:x2] = 0

    return result


def simplify_polyline(
    points: List[Tuple[float, float]], epsilon: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Simplify a polyline using Ramer-Douglas-Peucker algorithm.

    Args:
        points: List of (x, y) points.
        epsilon: Maximum distance for simplification.

    Returns:
        Simplified list of points.
    """
    if len(points) < 3:
        return points

    pts_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(pts_array, epsilon, False)

    return [(float(p[0][0]), float(p[0][1])) for p in simplified]


def trace_skeleton_paths(
    skeleton: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    """
    Trace continuous paths through a skeleton image.

    Args:
        skeleton: Binary skeleton image.

    Returns:
        List of paths, where each path is a list of (x, y) coordinates.
    """
    # Find all skeleton pixels
    points = np.column_stack(np.where(skeleton > 0))

    if len(points) == 0:
        return []

    # Build adjacency using 8-connectivity
    paths = []
    visited = np.zeros_like(skeleton, dtype=bool)

    def get_neighbors(y: int, x: int) -> List[Tuple[int, int]]:
        """Get unvisited 8-connected neighbors."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < skeleton.shape[0]
                    and 0 <= nx < skeleton.shape[1]
                    and skeleton[ny, nx] > 0
                    and not visited[ny, nx]
                ):
                    neighbors.append((ny, nx))
        return neighbors

    def count_neighbors(y: int, x: int) -> int:
        """Count 8-connected neighbors."""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < skeleton.shape[0]
                    and 0 <= nx < skeleton.shape[1]
                    and skeleton[ny, nx] > 0
                ):
                    count += 1
        return count

    # Find endpoints (degree 1) and junctions (degree > 2)
    endpoints = []
    for y, x in points:
        n = count_neighbors(y, x)
        if n == 1:
            endpoints.append((y, x))

    # Start tracing from endpoints
    start_points = endpoints if endpoints else [(points[0][0], points[0][1])]

    for start_y, start_x in start_points:
        if visited[start_y, start_x]:
            continue

        path = [(start_x, start_y)]
        visited[start_y, start_x] = True

        current = (start_y, start_x)
        while True:
            neighbors = get_neighbors(current[0], current[1])
            if not neighbors:
                break

            # Choose the first unvisited neighbor
            next_point = neighbors[0]
            visited[next_point[0], next_point[1]] = True
            path.append((next_point[1], next_point[0]))
            current = next_point

        if len(path) > 1:
            paths.append(path)

    return paths


def detect_connectors(
    binary_image: np.ndarray,
    nodes: List[NodeIR],
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> List[EdgeIR]:
    """
    Detect connectors/edges in a binary image.

    Args:
        binary_image: Binary input image (255 = ink).
        nodes: List of detected nodes (to mask them out).
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        List of detected EdgeIR objects.
    """
    # Mask out node regions
    connectors_only = mask_nodes(binary_image, nodes, margin=3)

    # Skeletonize to get 1-pixel lines
    binary_bool = connectors_only > 0
    skeleton = skeletonize(binary_bool).astype(np.uint8) * 255

    if debug_dir:
        cv2.imwrite(str(debug_dir / "11_skeleton.png"), skeleton)

    # Trace paths through skeleton
    paths = trace_skeleton_paths(skeleton)

    logger.info(f"Traced {len(paths)} raw connector paths")

    # Convert paths to EdgeIR objects
    edges = []
    for path in paths:
        # Calculate path length
        length = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            length += (dx * dx + dy * dy) ** 0.5

        # Filter short connectors
        if length < config.min_connector_length:
            continue

        # Simplify the path
        simplified = simplify_polyline([(float(p[0]), float(p[1])) for p in path])

        if len(simplified) >= 2:
            edge = EdgeIR(points=simplified)
            edges.append(edge)

    logger.info(f"Detected {len(edges)} connectors")

    # Save debug artifacts
    if debug_dir:
        # Save JSON
        edges_data = [
            {"id": e.id, "points": e.points, "num_points": len(e.points)} for e in edges
        ]
        with open(debug_dir / "12_connectors_raw.json", "w") as f:
            json.dump(edges_data, f, indent=2)

        # Save overlay
        overlay = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        for i, edge in enumerate(edges):
            color = (
                (i * 50) % 256,
                (i * 100 + 100) % 256,
                (i * 150 + 50) % 256,
            )
            pts = np.array([(int(p[0]), int(p[1])) for p in edge.points], dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, 2)

            # Mark endpoints
            if edge.points:
                cv2.circle(overlay, (int(edge.points[0][0]), int(edge.points[0][1])), 5, (0, 255, 0), -1)
                cv2.circle(overlay, (int(edge.points[-1][0]), int(edge.points[-1][1])), 5, (0, 0, 255), -1)

        cv2.imwrite(str(debug_dir / "13_connectors_overlay.png"), overlay)

    return edges
