"""CLI module for diagram2drawio."""

import logging
from pathlib import Path
from typing import Optional

import typer

from diagram2drawio import __version__
from diagram2drawio.api import build as api_build
from diagram2drawio.api import convert as api_convert
from diagram2drawio.api import extract as api_extract

app = typer.Typer(
    name="diagram2drawio",
    help="Convert raster diagram images to editable draw.io files.",
    add_completion=False,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"diagram2drawio {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """diagram2drawio - Convert raster diagrams to draw.io files."""
    pass


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Path to the input image file."),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Path for the output .drawio file. Defaults to input name with .drawio extension.",
    ),
    debug_dir: Optional[Path] = typer.Option(
        None,
        "--debug-dir",
        help="Directory to save debug artifacts.",
    ),
    compressed: bool = typer.Option(
        False,
        "--compressed",
        help="Compress the diagram content.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging.",
    ),
    # Pipeline configuration options
    min_shape_area: int = typer.Option(
        100,
        "--min-shape-area",
        help="Minimum contour area for shape detection.",
    ),
    snap_distance: int = typer.Option(
        15,
        "--snap-distance",
        help="Maximum distance for snapping edges to nodes.",
    ),
    ocr_confidence: float = typer.Option(
        0.4,
        "--ocr-confidence",
        help="Minimum OCR confidence threshold.",
    ),
    no_ocr: bool = typer.Option(
        False,
        "--no-ocr",
        help="Disable OCR text extraction.",
    ),
    no_arrowheads: bool = typer.Option(
        False,
        "--no-arrowheads",
        help="Disable arrowhead detection.",
    ),
) -> None:
    """
    Convert an image to a draw.io file.

    Example:
        diagram2drawio convert flowchart.png -o output.drawio --debug-dir debug/
    """
    setup_logging(verbose)

    # Default output path
    if output is None:
        output = input_path.with_suffix(".drawio")

    typer.echo(f"Converting {input_path} -> {output}")

    try:
        diagram = api_convert(
            input_path=input_path,
            output_path=output,
            debug_dir=debug_dir,
            compressed=compressed,
            min_shape_area=min_shape_area,
            snap_distance_threshold=snap_distance,
            ocr_confidence_threshold=ocr_confidence,
            enable_ocr=not no_ocr,
            enable_arrowheads=not no_arrowheads,
        )

        typer.echo(f"✓ Detected {len(diagram.nodes)} nodes and {len(diagram.edges)} edges")
        typer.echo(f"✓ Output saved to {output}")

        if debug_dir:
            typer.echo(f"✓ Debug artifacts saved to {debug_dir}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def extract(
    input_path: Path = typer.Argument(..., help="Path to the input image file."),
    output_ir: Path = typer.Option(
        None,
        "--out-ir",
        "-o",
        help="Path for the output IR JSON file.",
    ),
    debug_dir: Optional[Path] = typer.Option(
        None,
        "--debug-dir",
        help="Directory to save debug artifacts.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging.",
    ),
) -> None:
    """
    Extract diagram structure to an IR JSON file.

    This is the first step of the manual correction workflow.

    Example:
        diagram2drawio extract flowchart.png --out-ir diagram.json --debug-dir debug/
    """
    setup_logging(verbose)

    # Default output path
    if output_ir is None:
        output_ir = input_path.with_suffix(".json")

    typer.echo(f"Extracting {input_path} -> {output_ir}")

    try:
        diagram = api_extract(
            input_path=input_path,
            output_ir_path=output_ir,
            debug_dir=debug_dir,
        )

        typer.echo(f"✓ Detected {len(diagram.nodes)} nodes and {len(diagram.edges)} edges")
        typer.echo(f"✓ IR saved to {output_ir}")
        typer.echo("")
        typer.echo("You can now edit the JSON file to correct any detection errors,")
        typer.echo("then run 'diagram2drawio build' to generate the draw.io file.")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def build(
    ir_path: Path = typer.Argument(..., help="Path to the IR JSON file."),
    output: Path = typer.Option(
        None,
        "-o",
        "--output",
        help="Path for the output .drawio file.",
    ),
    compressed: bool = typer.Option(
        False,
        "--compressed",
        help="Compress the diagram content.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging.",
    ),
) -> None:
    """
    Build a draw.io file from an IR JSON file.

    This is the second step of the manual correction workflow.

    Example:
        diagram2drawio build diagram.json -o output.drawio
    """
    setup_logging(verbose)

    # Default output path
    if output is None:
        output = ir_path.with_suffix(".drawio")

    typer.echo(f"Building {ir_path} -> {output}")

    try:
        api_build(
            ir_path=ir_path,
            output_path=output,
            compressed=compressed,
        )

        typer.echo(f"✓ Output saved to {output}")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
