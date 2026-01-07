"""draw.io XML export module for diagram2drawio."""

import base64
import logging
import urllib.parse
import zlib
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from diagram2drawio.ir import DiagramIR, EdgeIR, NodeIR, ShapeType

logger = logging.getLogger(__name__)


# Style mappings for different shape types
SHAPE_STYLES = {
    ShapeType.RECTANGLE: "rounded=0;whiteSpace=wrap;html=1;",
    ShapeType.ROUNDED_RECTANGLE: "rounded=1;whiteSpace=wrap;html=1;",
    ShapeType.ELLIPSE: "ellipse;whiteSpace=wrap;html=1;",
    ShapeType.DIAMOND: "rhombus;whiteSpace=wrap;html=1;",
    ShapeType.UNKNOWN: "rounded=0;whiteSpace=wrap;html=1;",
}


def compress_diagram(xml_content: str) -> str:
    """
    Compress diagram XML for draw.io format.

    Pipeline: URL-encode -> raw deflate -> base64 encode

    Args:
        xml_content: The XML content to compress.

    Returns:
        Compressed and encoded string.
    """
    # URL encode
    encoded = urllib.parse.quote(xml_content, safe="")

    # Compress with raw deflate (no header)
    compressed = zlib.compress(encoded.encode("utf-8"), level=9)
    # Remove zlib header (first 2 bytes) and checksum (last 4 bytes)
    raw_deflate = compressed[2:-4]

    # Base64 encode
    result = base64.b64encode(raw_deflate).decode("ascii")

    return result


def decompress_diagram(compressed: str) -> str:
    """
    Decompress diagram content from draw.io format.

    Pipeline: base64 decode -> raw inflate -> URL-decode

    Args:
        compressed: The compressed string.

    Returns:
        Decompressed XML content.
    """
    # Base64 decode
    raw_deflate = base64.b64decode(compressed)

    # Decompress with raw inflate
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    inflated = decompressor.decompress(raw_deflate)

    # URL decode
    result = urllib.parse.unquote(inflated.decode("utf-8"))

    return result


def create_mxcell_node(node: NodeIR, parent_id: str = "1") -> ET.Element:
    """
    Create an mxCell element for a node.

    Args:
        node: The node to convert.
        parent_id: Parent cell ID.

    Returns:
        mxCell XML element.
    """
    cell = ET.Element("mxCell")
    cell.set("id", node.id)
    cell.set("value", node.label or "")
    cell.set("style", SHAPE_STYLES.get(node.shape, SHAPE_STYLES[ShapeType.RECTANGLE]))
    cell.set("vertex", "1")
    cell.set("parent", parent_id)

    # Add geometry
    geometry = ET.SubElement(cell, "mxGeometry")
    geometry.set("x", str(int(node.x)))
    geometry.set("y", str(int(node.y)))
    geometry.set("width", str(int(node.w)))
    geometry.set("height", str(int(node.h)))
    geometry.set("as", "geometry")

    return cell


def create_mxcell_edge(edge: EdgeIR, parent_id: str = "1") -> ET.Element:
    """
    Create an mxCell element for an edge.

    Args:
        edge: The edge to convert.
        parent_id: Parent cell ID.

    Returns:
        mxCell XML element.
    """
    cell = ET.Element("mxCell")
    cell.set("id", edge.id)
    cell.set("value", edge.label or "")

    # Build style string
    style_parts = ["edgeStyle=orthogonalEdgeStyle", "html=1"]

    if edge.has_arrowhead_target:
        style_parts.append("endArrow=classic")
    else:
        style_parts.append("endArrow=none")

    if edge.has_arrowhead_source:
        style_parts.append("startArrow=classic")
    else:
        style_parts.append("startArrow=none")

    cell.set("style", ";".join(style_parts) + ";")
    cell.set("edge", "1")
    cell.set("parent", parent_id)

    if edge.source_id:
        cell.set("source", edge.source_id)
    if edge.target_id:
        cell.set("target", edge.target_id)

    # Add geometry with points
    geometry = ET.SubElement(cell, "mxGeometry")
    geometry.set("relative", "1")
    geometry.set("as", "geometry")

    # Add waypoints if there are intermediate points
    if len(edge.points) > 2:
        points_array = ET.SubElement(geometry, "Array")
        points_array.set("as", "points")

        for point in edge.points[1:-1]:
            pt = ET.SubElement(points_array, "mxPoint")
            pt.set("x", str(int(point[0])))
            pt.set("y", str(int(point[1])))

    # Add source and target points if not connected to nodes
    if not edge.source_id and edge.points:
        source_pt = ET.SubElement(geometry, "mxPoint")
        source_pt.set("x", str(int(edge.points[0][0])))
        source_pt.set("y", str(int(edge.points[0][1])))
        source_pt.set("as", "sourcePoint")

    if not edge.target_id and edge.points:
        target_pt = ET.SubElement(geometry, "mxPoint")
        target_pt.set("x", str(int(edge.points[-1][0])))
        target_pt.set("y", str(int(edge.points[-1][1])))
        target_pt.set("as", "targetPoint")

    return cell


def export_to_drawio(
    diagram: DiagramIR,
    output_path: Path,
    compressed: bool = False,
    debug_dir: Optional[Path] = None,
) -> None:
    """
    Export a diagram IR to draw.io XML format.

    Args:
        diagram: The diagram to export.
        output_path: Path for the output file.
        compressed: Whether to compress the diagram content.
        debug_dir: Optional directory to save debug artifacts.
    """
    # Create mxGraphModel
    graph_model = ET.Element("mxGraphModel")
    graph_model.set("dx", "0")
    graph_model.set("dy", "0")
    graph_model.set("grid", "1")
    graph_model.set("gridSize", "10")
    graph_model.set("guides", "1")
    graph_model.set("tooltips", "1")
    graph_model.set("connect", "1")
    graph_model.set("arrows", "1")
    graph_model.set("fold", "1")
    graph_model.set("page", "1")
    graph_model.set("pageScale", "1")
    graph_model.set("pageWidth", str(diagram.width))
    graph_model.set("pageHeight", str(diagram.height))

    # Create root
    root = ET.SubElement(graph_model, "root")

    # Add default parent cells
    cell0 = ET.SubElement(root, "mxCell")
    cell0.set("id", "0")

    cell1 = ET.SubElement(root, "mxCell")
    cell1.set("id", "1")
    cell1.set("parent", "0")

    # Add nodes
    for node in diagram.nodes:
        node_cell = create_mxcell_node(node)
        root.append(node_cell)

    # Add edges
    for edge in diagram.edges:
        edge_cell = create_mxcell_edge(edge)
        root.append(edge_cell)

    # Create diagram content XML
    graph_xml = ET.tostring(graph_model, encoding="unicode")

    # Create mxfile structure
    mxfile = ET.Element("mxfile")
    mxfile.set("host", "diagram2drawio")
    mxfile.set("version", "0.1.0")
    mxfile.set("compressed", "true" if compressed else "false")

    diagram_elem = ET.SubElement(mxfile, "diagram")
    diagram_elem.set("name", "Page-1")
    diagram_elem.set("id", "page1")

    if compressed:
        diagram_elem.text = compress_diagram(graph_xml)
    else:
        diagram_elem.append(graph_model)

    # Write to file
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")

    with open(output_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    logger.info(f"Exported diagram to {output_path}")

    # Save debug copy
    if debug_dir:
        debug_path = debug_dir / "20_export.xml"
        with open(debug_path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)


def validate_drawio(file_path: Path) -> bool:
    """
    Validate a draw.io XML file.

    Args:
        file_path: Path to the file to validate.

    Returns:
        True if valid, False otherwise.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        if root.tag != "mxfile":
            logger.error("Root element is not 'mxfile'")
            return False

        diagrams = root.findall("diagram")
        if not diagrams:
            logger.error("No 'diagram' elements found")
            return False

        # Check for mxGraphModel
        for diagram in diagrams:
            if diagram.text and diagram.text.strip():
                # Compressed format - try to decompress
                try:
                    content = decompress_diagram(diagram.text.strip())
                    ET.fromstring(content)
                except Exception as e:
                    logger.error(f"Failed to parse compressed diagram: {e}")
                    return False
            else:
                # Uncompressed format
                graph_model = diagram.find("mxGraphModel")
                if graph_model is None:
                    logger.error("No 'mxGraphModel' found in diagram")
                    return False

        logger.info(f"Validated draw.io file: {file_path}")
        return True

    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return False
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False
