"""Edge-to-node association module for diagram2drawio."""

import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from diagram2drawio.config import DiagramConfig
from diagram2drawio.ir import EdgeIR, NodeIR

logger = logging.getLogger(__name__)


def point_to_rect_distance(
    point: Tuple[float, float], node: NodeIR
) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate the distance from a point to the nearest edge of a rectangle.

    Args:
        point: (x, y) point.
        node: Node with bounding box.

    Returns:
        Tuple of (distance, closest_point_on_rect).
    """
    px, py = point
    x1, y1 = node.x, node.y
    x2, y2 = node.x + node.w, node.y + node.h

    # Find closest x coordinate
    if px < x1:
        cx = x1
    elif px > x2:
        cx = x2
    else:
        cx = px

    # Find closest y coordinate
    if py < y1:
        cy = y1
    elif py > y2:
        cy = y2
    else:
        cy = py

    # Calculate distance
    dx = px - cx
    dy = py - cy
    dist = math.sqrt(dx * dx + dy * dy)

    return dist, (cx, cy)


def find_nearest_node(
    point: Tuple[float, float],
    nodes: List[NodeIR],
    threshold: float,
) -> Optional[Tuple[NodeIR, Tuple[float, float]]]:
    """
    Find the nearest node to a point within a threshold.

    Args:
        point: (x, y) point.
        nodes: List of nodes.
        threshold: Maximum distance threshold.

    Returns:
        Tuple of (node, snap_point) if found, None otherwise.
    """
    best_node = None
    best_dist = threshold
    best_snap = None

    for node in nodes:
        dist, snap_point = point_to_rect_distance(point, node)
        if dist < best_dist:
            best_dist = dist
            best_node = node
            best_snap = snap_point

    if best_node:
        return best_node, best_snap
    return None


def associate_edges_to_nodes(
    edges: List[EdgeIR],
    nodes: List[NodeIR],
    config: DiagramConfig,
    debug_dir: Optional[Path] = None,
) -> List[EdgeIR]:
    """
    Associate edge endpoints with nearby nodes.

    Args:
        edges: List of detected edges.
        nodes: List of detected nodes.
        config: Pipeline configuration.
        debug_dir: Optional directory to save debug artifacts.

    Returns:
        Updated list of edges with source/target node IDs.
    """
    associations = []

    for edge in edges:
        if len(edge.points) < 2:
            continue

        # Check source (first point)
        start = edge.points[0]
        result = find_nearest_node(start, nodes, config.snap_distance_threshold)
        if result:
            node, snap_point = result
            edge.source_id = node.id
            # Optionally snap the endpoint to the node boundary
            edge.points[0] = snap_point
            associations.append({
                "edge_id": edge.id,
                "end": "source",
                "node_id": node.id,
                "snap_point": snap_point,
            })

        # Check target (last point)
        end = edge.points[-1]
        result = find_nearest_node(end, nodes, config.snap_distance_threshold)
        if result:
            node, snap_point = result
            edge.target_id = node.id
            # Optionally snap the endpoint
            edge.points[-1] = snap_point
            associations.append({
                "edge_id": edge.id,
                "end": "target",
                "node_id": node.id,
                "snap_point": snap_point,
            })

    connected = sum(1 for e in edges if e.source_id or e.target_id)
    logger.info(f"Associated {connected}/{len(edges)} edges to nodes")

    # Save debug artifacts
    if debug_dir:
        with open(debug_dir / "16_graph.json", "w") as f:
            json.dump(associations, f, indent=2)

    return edges
