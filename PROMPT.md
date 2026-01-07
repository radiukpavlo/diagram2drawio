# AGENTS.md — Agent Specification for a CV+OCR “Diagram Understanding” Library (Raster → draw.io)

You are an **agentic large language model** operating like a senior **Computer Vision + OCR + Python packaging** engineer.

Your mission: implement a **mini diagram-understanding system** that converts a **raster flow-chart / block-diagram image** (PNG/JPG/etc.) into a **fully editable diagrams.net / draw.io** file (`.drawio` / `.xml` mxGraph).

This is *not* “vector tracing.” The output must be **semantic**: separate **nodes** (shapes), **connectors** (edges), and **labels** (text) as real draw.io objects.

---

## 0) Non-negotiable requirements

### Functional requirements
1. **Input**: raster image (PNG/JPG/TIFF/BMP) containing a flowchart or similar diagram.
2. **Output**: a draw.io file that opens cleanly and is **editable**:
   - default: `*.drawio` with **uncompressed** diagram XML (diff-friendly)
   - optional: compressed draw.io diagram payload (deflate/base64/url-encoding) for parity with draw.io defaults
3. Provide both:
   - **Python API**: `diagram2drawio.convert(image_path, out_path, config=...)`
   - **CLI**: `diagram2drawio convert input.png -o out.drawio --debug-dir debug/`
4. Provide **debug artifacts** (images + JSON) for every major stage.
5. Provide a **manual correction loop**:
   - export intermediate JSON (your IR)
   - allow user edits
   - rebuild draw.io output from edited JSON

### Packaging requirements (Windows + pip)
1. Must install via **`pip` on Windows** (no compiling, no “go build this C library”).
2. Prefer dependencies with **prebuilt wheels** on PyPI for Windows.
3. Use a modern **`pyproject.toml`** and `src/` layout.
4. Ship tests (`pytest`) and a minimal CI workflow (optional but strongly recommended).

### Operational requirements
1. Must run **offline** (no cloud OCR, no external APIs).
2. Deterministic given the same config.
3. Degrade gracefully:
   - if arrowheads cannot be detected → still output edges (undirected)
   - if OCR is poor → still output nodes/edges with empty labels
4. Provide clear warnings and a debug directory when quality is low.

---

## 1) Toolset (optimal for Windows + pip)

Your implementation must rely on a pip-installable toolchain:

### Core CV + preprocessing
- `opencv-python-headless` (preferred) or `opencv-python`  
- `numpy`

### Morphology + skeletonization
- `scikit-image`

### OCR (pluggable)
Default (pip-only):
- `easyocr`

Optional:
- `pytesseract` (requires the Tesseract binary; keep behind an extra)

### Geometry + graph reasoning
- `shapely`
- `networkx`

### Export + CLI
- `typer` (CLI)
- `pydantic` (or `dataclasses`) for configuration and IR validation
- stdlib `xml.etree.ElementTree` (or `lxml` if you want pretty XML)

### Dev tooling (extras)
- `pytest`
- `ruff`
- `mypy` (optional)
- `build`
- `twine` (optional)

---

## 2) The pip-installable library you must produce

### Project name
- PyPI package name: `diagram2drawio`
- Python import: `diagram2drawio`

### Installation UX
A user must be able to do:

```bash
pip install diagram2drawio
```

Optional extras:

```bash
pip install "diagram2drawio[dev]"
pip install "diagram2drawio[tesseract]"
```

### CLI entrypoint
Expose a `diagram2drawio` command:

```bash
diagram2drawio --help
diagram2drawio convert input.png -o output.drawio --debug-dir debug/
```

---

## 3) Repository + deliverables checklist

Your repo must include:

- `pyproject.toml`
- `README.md`
- `src/diagram2drawio/`
  - `__init__.py`
  - `api.py`
  - `cli.py`
  - `config.py`
  - `ir.py`
  - `pipeline/`
    - `preprocess.py`
    - `ocr.py`
    - `text_mask.py`
    - `shapes.py`
    - `connectors.py`
    - `arrowheads.py`
    - `associate.py`
    - `labels.py`
    - `validate.py`
  - `export/`
    - `drawio.py` (mxGraph writer + compression utilities)
- `tests/`
- `examples/` (tiny images + expected JSON/outputs)
- Optional: `CHANGELOG.md`, `LICENSE`

---

## 4) Intermediate Representation (IR) you must define

Define an internal **Diagram IR** that is independent of draw.io.

### Core IR objects
- `DiagramIR`
  - `width`, `height` (in “working pixels” after normalization)
  - `nodes: list[NodeIR]`
  - `edges: list[EdgeIR]`
  - `texts: list[TextIR]`
  - `meta: dict` (pipeline metadata, versions, config snapshot)

- `NodeIR`
  - `id: str`
  - `shape_type: enum`
  - `bbox: (x, y, w, h)` (float allowed; export rounds)
  - `polygon: list[(x, y)] | None`
  - `label: str`
  - `confidence: float`
  - `style_hint: dict` (optional: fill/stroke guessed from image)

- `EdgeIR`
  - `id: str`
  - `source_node_id: str | None`
  - `target_node_id: str | None`
  - `polyline: list[(x, y)]`
  - `directed: bool`
  - `label: str`
  - `confidence: float`

- `TextIR`
  - `id: str`
  - `text: str`
  - `bbox: (x, y, w, h)`
  - `polygon: list[(x, y)] | None`
  - `confidence: float`
  - `assigned_to: 'node'|'edge'|None`
  - `assigned_id: str|None`

### JSON round-trip
You must support:

- `DiagramIR → JSON` (for manual correction and debugging)
- `JSON → DiagramIR` (for rebuild)

---

## 5) Pipeline specification (complete CV + OCR)

Implement the conversion as a multi-stage pipeline. Each stage must:
- be unit-testable
- be configurable
- optionally emit debug artifacts

### Stage A — Ingest + normalization
1. Load image (OpenCV).
2. Normalize to a working resolution:
   - optional upscaling if image is tiny
   - optional downscaling if extremely large (keep aspect ratio)
3. Convert to grayscale, keep original BGR for optional style inference.

**Outputs**
- `00_input.png`
- `01_norm.png`

### Stage B — Preprocessing for segmentation
Goal: produce a clean binary mask of “ink” vs background.

1. Denoise (median/bilateral).
2. Contrast normalization (optional CLAHE).
3. Binarize:
   - adaptive threshold (default)
   - fallback to Otsu/global threshold
4. Morphological cleanup:
   - close gaps in lines
   - remove small specks

**Outputs**
- `02_gray.png`
- `03_binary.png`
- `04_binary_clean.png`

### Stage C — OCR (text boxes + transcription)
Implement a pluggable `OcrBackend` interface.

Default backend: EasyOCR.

Requirements:
- return bounding polygons or boxes
- return confidence per text region
- preserve reading order where possible

**Outputs**
- `05_ocr.json`
- `06_ocr_overlay.png`

### Stage D — Text masking (so text doesn’t become “shapes”)
1. Expand OCR boxes by padding.
2. Create a text mask.
3. Remove text pixels from the binary image:
   - simplest: set masked pixels to background
   - optional: inpaint if needed

**Outputs**
- `07_text_mask.png`
- `08_binary_no_text.png`

### Stage E — Node (shape) detection and classification
Detect diagram nodes as geometric primitives.

Core method (required):
- contour detection on a cleaned “filled” mask

Optional enhancements:
- separate detection for filled vs outlined shapes
- detect rounded rectangles using curvature/corner analysis

Classification heuristics (must implement):
- rectangle
- rounded rectangle
- diamond (rhombus)
- ellipse/circle
- fallback UNKNOWN

Required post-processing:
- non-max suppression / merge duplicates
- remove contours that are likely connector fragments

**Outputs**
- `09_nodes.json`
- `10_nodes_overlay.png`

### Stage F — Connector detection (two complementary methods)
You must implement **both** and fuse results.

#### F1) Skeleton-based topology extraction (primary)
1. Skeletonize `binary_no_text` to 1-pixel lines.
2. Build an 8-connected pixel graph.
3. Identify endpoints and junctions.
4. Trace stroke polylines.
5. Simplify polylines (RDP).

#### F2) Hough-based straight-line detection (fallback/assist)
1. Run edge detection (Canny) on the “no text” image.
2. Run probabilistic Hough transform for line segments.
3. Merge collinear segments.
4. Use these segments to:
   - repair broken skeleton strokes
   - improve orthogonal routing detection

Fusion requirement:
- unify skeleton strokes and Hough segments into a single set of candidate connector polylines.

**Outputs**
- `11_skeleton.png`
- `12_connectors_raw.json`
- `13_connectors_overlay.png`

### Stage G — Arrowhead detection + direction inference
Detect arrowheads and infer directed edges.

Minimum viable:
1. find small triangular contours
2. validate geometry + adjacency to a connector endpoint
3. infer tip vs base
4. mark edge direction

If arrowheads cannot be found:
- output undirected edge but keep geometry

**Outputs**
- `14_arrowheads.json`
- `15_arrowheads_overlay.png`

### Stage H — Associate connectors to nodes
Goal: attach edge endpoints to node boundaries.

1. For each connector polyline, compute terminal endpoints (or multiple endpoints if branching).
2. Snap endpoints to nearest node boundary within threshold using Shapely distance checks.
3. Assign `source_node_id` and `target_node_id`.
4. Handle branching:
   - split into multiple edges OR
   - create intermediate junction nodes (configurable)

**Outputs**
- `16_graph.json`
- `17_graph_overlay.png`

### Stage I — Assign labels (text → node/edge)
Rules:
1. Text inside a node bbox → node label.
2. Text near an edge polyline → edge label.
3. Merge multi-line labels inside the same node.

**Outputs**
- `18_labels_overlay.png`
- `19_diagram_ir.json` (final IR snapshot)

### Stage J — Export to draw.io XML (mxGraph)

#### J1) Uncompressed output (default)
Write:
- `<mxfile ... compressed="false">`
- `<diagram ...> <mxGraphModel> ... </mxGraphModel> </diagram>`

Within `<mxGraphModel>`:
- `<root>`
  - `<mxCell id="0" />`
  - `<mxCell id="1" parent="0" />`
  - one `<mxCell ... vertex="1">` per `NodeIR`
  - one `<mxCell ... edge="1" source="..." target="...">` per `EdgeIR`

#### J2) Compressed output (optional)
Implement utilities to encode/decode diagram payload:
- **Compression pipeline**: URL-encode XML → raw deflate → base64 encode
- **Decompression pipeline**: base64 decode → raw inflate → URL-decode

Expose these utilities publicly and test them.

#### Styles
Implement a minimal but valid style set:
- rectangles, rounded rectangles, rhombus, ellipse
- orthogonal edges with arrowheads when directed

Keep style inference optional (colors, fonts, etc.).

**Outputs**
- `output.drawio` (or `.xml`)
- `output_uncompressed.drawio`
- `20_export.xml` (optional, for debugging)

### Stage K — Validation
Implement validators:
- XML is well-formed
- every referenced cell ID exists
- geometry sanity checks
- warnings for low-confidence detection

---

## 6) Manual correction workflow (must implement)

Your tool must support this “human-in-the-loop” loop:

1. `diagram2drawio extract input.png --out-ir diagram.json --debug-dir debug/`
2. user edits `diagram.json` (fix missing labels, edges, node types)
3. `diagram2drawio build diagram.json --out out.drawio`

This is the safety valve that turns “90% automatic” into “100% usable.”

---

## 7) Testing strategy (must ship with package)

### Unit tests
- shape classification
- skeleton tracing and polyline simplification
- arrowhead detection on synthetic images
- XML export correctness (parse and count cells)

### Synthetic evaluation harness
Implement a small generator that draws simple diagrams:
- rectangles + diamonds + ellipses
- orthogonal arrows with arrowheads
- text labels with a common font
- optional noise/blur/compression

Generate ground-truth IR and validate recovered IR:
- node IoU
- edge attachment correctness
- label assignment accuracy

### Acceptance thresholds (initial target)
- Nodes: ≥90% detected with IoU ≥0.8 on synthetic set
- Edges: ≥85% correctly attached on synthetic set
- Labels: ≥85% correct assignment on synthetic set

---

## 8) Packaging + release instructions you must follow

1. Use `pyproject.toml` with a modern build backend.
2. Provide `diagram2drawio.__version__`.
3. Ensure `pip install .` works cleanly on Windows.
4. Provide a `python -m build` workflow and verify the wheel installs.
5. Avoid shipping gigantic model weights inside the wheel (EasyOCR downloads models at runtime).

---

## 9) Common failure modes + countermeasures (implement mitigations)

1. **Text becomes “shapes”**
   - ensure OCR masking happens before contour detection
2. **Node borders mistaken for connectors**
   - mask node bboxes (with margin) before skeletonization
3. **Broken connectors**
   - morphology closing
   - Hough fallback to bridge gaps
4. **Arrowheads missed**
   - search near endpoints; tolerate partial triangles
5. **Dense diagrams**
   - debug overlays must clearly show which element failed
   - provide config knobs for thresholds

---

## 10) Definition of “done”

You are done only when:

1. Windows user can `pip install diagram2drawio` and run `diagram2drawio convert input.png -o out.drawio`.
2. Output opens in diagrams.net and is **editable** (move boxes, reconnect edges).
3. `pytest` passes locally and in CI.
4. Debug outputs are clear, consistent, and helpful.
5. The library has a stable public API and versioned releases.

---

## 11) Boundaries

- Do not hardcode logic for a single example image.
- No cloud services.
- CPU must work; GPU is optional.
- The objective is a **maintainable, improvable system** with a reliable manual correction loop.
