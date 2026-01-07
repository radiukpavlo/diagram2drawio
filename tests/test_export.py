"""Tests for the draw.io export module."""

import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from diagram2drawio.export.drawio import (
    compress_diagram,
    decompress_diagram,
    export_to_drawio,
    validate_drawio,
)
from diagram2drawio.ir import DiagramIR, EdgeIR, NodeIR, ShapeType


class TestCompression:
    """Tests for compression/decompression utilities."""

    def test_compress_decompress_roundtrip(self):
        """Test that compress and decompress are inverses."""
        original = "<mxGraphModel><root></root></mxGraphModel>"
        compressed = compress_diagram(original)
        decompressed = decompress_diagram(compressed)
        assert decompressed == original

    def test_compress_produces_shorter_output(self):
        """Test that compression reduces size for larger inputs."""
        original = "<mxGraphModel>" + "x" * 1000 + "</mxGraphModel>"
        compressed = compress_diagram(original)
        # Compressed should be shorter for repetitive content
        assert len(compressed) < len(original)


class TestExport:
    """Tests for draw.io export functionality."""

    def test_export_empty_diagram(self):
        """Test exporting an empty diagram."""
        diagram = DiagramIR(width=800, height=600)

        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_to_drawio(diagram, output_path)
            assert output_path.exists()

            # Validate XML structure
            tree = ET.parse(output_path)
            root = tree.getroot()
            assert root.tag == "mxfile"

            diagram_elem = root.find("diagram")
            assert diagram_elem is not None

        finally:
            output_path.unlink()

    def test_export_with_nodes(self):
        """Test exporting a diagram with nodes."""
        node1 = NodeIR(x=10, y=10, w=100, h=50, label="Node 1")
        node2 = NodeIR(
            x=200, y=10, w=100, h=50, label="Node 2", shape=ShapeType.ELLIPSE
        )

        diagram = DiagramIR(width=400, height=100, nodes=[node1, node2])

        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_to_drawio(diagram, output_path)

            # Find mxCell elements
            tree = ET.parse(output_path)
            cells = tree.findall(".//mxCell[@vertex='1']")

            # Should have 2 node cells
            assert len(cells) == 2

        finally:
            output_path.unlink()

    def test_export_with_edges(self):
        """Test exporting a diagram with connected edges."""
        node1 = NodeIR(x=10, y=10, w=100, h=50)
        node2 = NodeIR(x=200, y=10, w=100, h=50)
        edge = EdgeIR(
            points=[(110, 35), (200, 35)],
            source_id=node1.id,
            target_id=node2.id,
        )

        diagram = DiagramIR(
            width=400, height=100, nodes=[node1, node2], edges=[edge]
        )

        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_to_drawio(diagram, output_path)

            # Find edge cells
            tree = ET.parse(output_path)
            edge_cells = tree.findall(".//mxCell[@edge='1']")

            assert len(edge_cells) == 1
            assert edge_cells[0].get("source") == node1.id
            assert edge_cells[0].get("target") == node2.id

        finally:
            output_path.unlink()

    def test_export_compressed(self):
        """Test exporting with compression enabled."""
        diagram = DiagramIR(width=800, height=600)

        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_to_drawio(diagram, output_path, compressed=True)

            tree = ET.parse(output_path)
            root = tree.getroot()
            assert root.get("compressed") == "true"

            diagram_elem = root.find("diagram")
            # Compressed content should be base64 text
            assert diagram_elem.text is not None
            assert len(diagram_elem.text) > 0

        finally:
            output_path.unlink()


class TestValidation:
    """Tests for draw.io file validation."""

    def test_validate_valid_file(self):
        """Test validation of a valid draw.io file."""
        diagram = DiagramIR(width=800, height=600)

        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False) as f:
            output_path = Path(f.name)

        try:
            export_to_drawio(diagram, output_path)
            assert validate_drawio(output_path) is True
        finally:
            output_path.unlink()

    def test_validate_invalid_file(self):
        """Test validation of an invalid file."""
        with tempfile.NamedTemporaryFile(suffix=".drawio", delete=False, mode="w") as f:
            f.write("<invalid>content</invalid>")
            output_path = Path(f.name)

        try:
            assert validate_drawio(output_path) is False
        finally:
            output_path.unlink()
