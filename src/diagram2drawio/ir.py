"""Intermediate Representation (IR) models for diagram2drawio."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id() -> str:
    """Generate a unique ID for IR objects."""
    return str(uuid4())[:8]


class ShapeType(str, Enum):
    """Supported shape types for diagram nodes."""

    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    ELLIPSE = "ellipse"
    DIAMOND = "diamond"
    UNKNOWN = "unknown"


class TextIR(BaseModel):
    """Represents detected text in the diagram."""

    id: str = Field(default_factory=generate_id)
    text: str = Field(..., description="The recognized text content")
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="Bounding box as (x, y, width, height)"
    )
    polygon: Optional[List[Tuple[float, float]]] = Field(
        default=None, description="Optional polygon vertices"
    )
    confidence: float = Field(default=1.0, description="OCR confidence score")
    assigned_to: Optional[str] = Field(
        default=None, description="Type of element assigned to: 'node' or 'edge'"
    )
    assigned_id: Optional[str] = Field(
        default=None, description="ID of the assigned node or edge"
    )


class NodeIR(BaseModel):
    """Represents a shape/node in the diagram."""

    id: str = Field(default_factory=generate_id)
    shape: ShapeType = Field(default=ShapeType.RECTANGLE, description="Detected shape type")
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    w: float = Field(..., description="Width of the bounding box")
    h: float = Field(..., description="Height of the bounding box")
    polygon: Optional[List[Tuple[float, float]]] = Field(
        default=None, description="Optional contour polygon"
    )
    label: Optional[str] = Field(default=None, description="Text label for the node")
    confidence: float = Field(default=1.0, description="Detection confidence")
    style_props: Dict[str, str] = Field(
        default_factory=dict, description="Style properties (fill, stroke, etc.)"
    )


class EdgeIR(BaseModel):
    """Represents a connector/edge in the diagram."""

    id: str = Field(default_factory=generate_id)
    points: List[Tuple[float, float]] = Field(
        default_factory=list, description="Ordered list of polyline vertices"
    )
    source_id: Optional[str] = Field(default=None, description="ID of source node")
    target_id: Optional[str] = Field(default=None, description="ID of target node")
    has_arrowhead_source: bool = Field(
        default=False, description="Arrowhead at source end"
    )
    has_arrowhead_target: bool = Field(
        default=True, description="Arrowhead at target end"
    )
    label: Optional[str] = Field(default=None, description="Text label for the edge")
    confidence: float = Field(default=1.0, description="Detection confidence")


class DiagramIR(BaseModel):
    """Complete intermediate representation of a diagram."""

    width: int = Field(..., description="Diagram width in pixels")
    height: int = Field(..., description="Diagram height in pixels")
    nodes: List[NodeIR] = Field(default_factory=list, description="Detected nodes")
    edges: List[EdgeIR] = Field(default_factory=list, description="Detected edges")
    texts: List[TextIR] = Field(default_factory=list, description="Detected text regions")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Pipeline metadata"
    )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DiagramIR":
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)

    def get_node_by_id(self, node_id: str) -> Optional[NodeIR]:
        """Find a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edge_by_id(self, edge_id: str) -> Optional[EdgeIR]:
        """Find an edge by its ID."""
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        return None
