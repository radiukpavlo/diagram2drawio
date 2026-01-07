"""Tests for the IR module."""

import json

import pytest

from diagram2drawio.ir import DiagramIR, EdgeIR, NodeIR, ShapeType, TextIR


class TestNodeIR:
    """Tests for NodeIR model."""

    def test_create_node(self):
        """Test basic node creation."""
        node = NodeIR(x=10, y=20, w=100, h=50)
        assert node.x == 10
        assert node.y == 20
        assert node.w == 100
        assert node.h == 50
        assert node.shape == ShapeType.RECTANGLE
        assert node.label is None
        assert node.id is not None

    def test_node_with_shape(self):
        """Test node with specific shape type."""
        node = NodeIR(x=0, y=0, w=50, h=50, shape=ShapeType.ELLIPSE)
        assert node.shape == ShapeType.ELLIPSE

    def test_node_serialization(self):
        """Test node JSON serialization."""
        node = NodeIR(x=10, y=20, w=100, h=50, label="Test")
        data = node.model_dump()
        assert data["x"] == 10
        assert data["label"] == "Test"


class TestEdgeIR:
    """Tests for EdgeIR model."""

    def test_create_edge(self):
        """Test basic edge creation."""
        edge = EdgeIR(points=[(0, 0), (100, 100)])
        assert len(edge.points) == 2
        assert edge.source_id is None
        assert edge.target_id is None
        assert edge.has_arrowhead_target is True

    def test_edge_with_nodes(self):
        """Test edge connected to nodes."""
        edge = EdgeIR(
            points=[(0, 0), (100, 100)],
            source_id="node1",
            target_id="node2",
        )
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"


class TestDiagramIR:
    """Tests for DiagramIR model."""

    def test_create_empty_diagram(self):
        """Test empty diagram creation."""
        diagram = DiagramIR(width=800, height=600)
        assert diagram.width == 800
        assert diagram.height == 600
        assert len(diagram.nodes) == 0
        assert len(diagram.edges) == 0

    def test_diagram_with_elements(self):
        """Test diagram with nodes and edges."""
        node1 = NodeIR(x=0, y=0, w=100, h=50)
        node2 = NodeIR(x=200, y=0, w=100, h=50)
        edge = EdgeIR(
            points=[(100, 25), (200, 25)],
            source_id=node1.id,
            target_id=node2.id,
        )

        diagram = DiagramIR(
            width=400,
            height=100,
            nodes=[node1, node2],
            edges=[edge],
        )

        assert len(diagram.nodes) == 2
        assert len(diagram.edges) == 1

    def test_json_roundtrip(self):
        """Test JSON serialization and deserialization."""
        node = NodeIR(x=10, y=20, w=100, h=50, label="Test Node")
        original = DiagramIR(width=800, height=600, nodes=[node])

        json_str = original.to_json()
        restored = DiagramIR.from_json(json_str)

        assert restored.width == original.width
        assert restored.height == original.height
        assert len(restored.nodes) == 1
        assert restored.nodes[0].label == "Test Node"

    def test_get_node_by_id(self):
        """Test finding node by ID."""
        node = NodeIR(x=0, y=0, w=100, h=50)
        diagram = DiagramIR(width=800, height=600, nodes=[node])

        found = diagram.get_node_by_id(node.id)
        assert found is not None
        assert found.id == node.id

        not_found = diagram.get_node_by_id("nonexistent")
        assert not_found is None
