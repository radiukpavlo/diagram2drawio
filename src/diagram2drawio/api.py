"""Main API module for diagram2drawio."""

import json
import logging
from pathlib import Path
from typing import Optional, Union

from diagram2drawio.config import DiagramConfig
from diagram2drawio.export.drawio import export_to_drawio
from diagram2drawio.ir import DiagramIR
from diagram2drawio.pipeline.arrowheads import detect_arrowheads
from diagram2drawio.pipeline.associate import associate_edges_to_nodes
from diagram2drawio.pipeline.connectors import detect_connectors
from diagram2drawio.pipeline.labels import assign_labels
from diagram2drawio.pipeline.ocr import run_ocr
from diagram2drawio.pipeline.preprocess import binarize, load_image
from diagram2drawio.pipeline.shapes import detect_shapes
from diagram2drawio.pipeline.text_mask import remove_text

logger = logging.getLogger(__name__)


def extract(
    input_path: Union[str, Path],
    output_ir_path: Optional[Union[str, Path]] = None,
    debug_dir: Optional[Union[str, Path]] = None,
    **config_kwargs,
) -> DiagramIR:
    """
    Extract diagram structure from an image.

    This runs the full CV pipeline and returns the Intermediate Representation.

    Args:
        input_path: Path to the input image.
        output_ir_path: Optional path to save the IR as JSON.
        debug_dir: Optional directory for debug artifacts.
        **config_kwargs: Additional configuration parameters.

    Returns:
        DiagramIR containing detected nodes, edges, and text.
    """
    input_path = Path(input_path)
    debug_path = Path(debug_dir) if debug_dir else None

    # Create config
    config = DiagramConfig(
        input_path=input_path,
        output_path=Path("output.drawio"),
        debug_dir=debug_path,
        **config_kwargs,
    )

    logger.info(f"Starting extraction from {input_path}")

    # Stage A: Load and normalize
    original, gray, scale = load_image(input_path, config, debug_path)
    h, w = gray.shape[:2]

    # Stage B: Binarize
    binary = binarize(gray, config, debug_path)

    # Stage C: OCR
    texts = run_ocr(gray, config, debug_path)

    # Stage D: Remove text
    binary_no_text = remove_text(binary, texts, config, debug_path)

    # Stage E: Detect shapes
    nodes = detect_shapes(binary_no_text, config, debug_path)

    # Stage F: Detect connectors
    edges = detect_connectors(binary_no_text, nodes, config, debug_path)

    # Stage G: Detect arrowheads
    edges = detect_arrowheads(binary, edges, config, debug_path)

    # Stage H: Associate edges to nodes
    edges = associate_edges_to_nodes(edges, nodes, config, debug_path)

    # Stage I: Assign labels
    nodes, edges, texts = assign_labels(texts, nodes, edges, config, debug_path, original)

    # Build IR
    diagram = DiagramIR(
        width=w,
        height=h,
        nodes=nodes,
        edges=edges,
        texts=texts,
        metadata={
            "source_file": str(input_path),
            "scale_factor": scale,
            "config": config.model_dump(mode="json"),
        },
    )

    # Save IR if requested
    if output_ir_path:
        output_ir_path = Path(output_ir_path)
        with open(output_ir_path, "w") as f:
            f.write(diagram.to_json())
        logger.info(f"Saved IR to {output_ir_path}")

    if debug_path:
        with open(debug_path / "19_diagram_ir.json", "w") as f:
            f.write(diagram.to_json())

    return diagram


def build(
    ir_path: Union[str, Path],
    output_path: Union[str, Path],
    compressed: bool = False,
) -> None:
    """
    Build a draw.io file from an IR JSON file.

    Args:
        ir_path: Path to the IR JSON file.
        output_path: Path for the output .drawio file.
        compressed: Whether to compress the output.
    """
    ir_path = Path(ir_path)
    output_path = Path(output_path)

    # Load IR
    with open(ir_path, "r") as f:
        diagram = DiagramIR.from_json(f.read())

    logger.info(f"Loaded IR from {ir_path}: {len(diagram.nodes)} nodes, {len(diagram.edges)} edges")

    # Export
    export_to_drawio(diagram, output_path, compressed=compressed)


def convert(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    debug_dir: Optional[Union[str, Path]] = None,
    compressed: bool = False,
    **config_kwargs,
) -> DiagramIR:
    """
    Convert an image to a draw.io file.

    This is the main entry point that runs extract + build.

    Args:
        input_path: Path to the input image.
        output_path: Path for the output .drawio file.
        debug_dir: Optional directory for debug artifacts.
        compressed: Whether to compress the output.
        **config_kwargs: Additional configuration parameters.

    Returns:
        DiagramIR containing the extracted diagram structure.

    Example:
        >>> from diagram2drawio import convert
        >>> convert("flowchart.png", "flowchart.drawio", debug_dir="debug/")
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    debug_path = Path(debug_dir) if debug_dir else None

    # Extract
    diagram = extract(input_path, debug_dir=debug_path, **config_kwargs)

    # Export
    export_to_drawio(diagram, output_path, compressed=compressed, debug_dir=debug_path)

    logger.info(f"Conversion complete: {input_path} -> {output_path}")

    return diagram
