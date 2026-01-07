# AGENTS.md — Detailed Implementation Specification for `diagram2drawio`

> **Role**: Senior Computer Vision Engineer & Python Architect
> **Objective**: Build a production-grade, offline-capable library to convert raster diagrams into editable `draw.io` files.

---

## 1. System Architecture & Design Patterns

### 1.1 Core Design Principles
*   **Pipeline Architecture**: The conversion process is a linear pipeline of isolated, testable transformations.
    *   `Image -> [Preprocess] -> [OCR] -> [Masking] -> [ShapeDetect] -> [ConnectorDetect] -> [Topology] -> [Export] -> XML`
*   **Immutability**: Intermediate representations (IR) should be treated as immutable snapshots where possible to enable "time-travel" debugging.
*   **Fail-Soft**: If a specific shape or edge fails detection, the rest of the diagram must still be generated.
*   **Pluggable Backends**: OCR and Export layers must be behind interfaces to allow future swapping (e.g., swapping EasyOCR for Tesseract).

### 1.2 Technology Stack (Strict)
*   **Image Processing**: `opencv-python-headless`, `scikit-image`, `numpy`
*   **OCR**: `easyocr` (default, offline)
*   **Geometry**: `shapely` (spatial predicates), `networkx` (graph topology)
*   **CLI/Config**: `typer`, `pydantic`
*   **Packaging**: `pyproject.toml`, `setuptools`/`hatch` (standard PEP 517)

---

## 2. Data Structures & Intermediate Representation (IR)

The `DiagramIR` is the source of truth. All pipeline stages mutate or enrich this structure.

### 2.1 Configuration (`src/diagram2drawio/config.py`)
```python
class DiagramConfig(BaseModel):
    # I/O
    input_path: Path
    output_path: Path
    debug_dir: Optional[Path] = None

    # Processing Toggles
    enable_ocr: bool = True
    enable_arrowheads: bool = True
    
    # Thresholds (Tunable kwargs)
    binarize_threshold_block_size: int = 21  # odd number for adaptive threshold
    binarize_c: int = 5
    min_shape_area: int = 100 # pixels
    epsilon_approx_poly_factor: float = 0.04
    ocr_confidence_threshold: float = 0.4
    snap_distance_threshold: int = 15 # pixels for edge snapping
```

### 2.2 Domain Models (`src/diagram2drawio/ir.py`)
```python
class ShapeType(str, Enum):
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    ELLIPSE = "ellipse"
    DIAMOND = "diamond"
    UNKNOWN = "unknown"

class NodeIR(BaseModel):
    id: str  # UUID
    shape: ShapeType
    # Geometry in "working space" coordinates
    x: float
    y: float
    w: float
    h: float
    label: Optional[str] = None
    style_props: Dict[str, str] = {} # e.g. fill color

class EdgeIR(BaseModel):
    id: str
    points: List[Tuple[float, float]] # Ordered list of vertices
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    has_arrowhead_source: bool = False
    has_arrowhead_target: bool = True # standard flow
    label: Optional[str] = None

class DiagramIR(BaseModel):
    width: int
    height: int
    nodes: List[NodeIR] = []
    edges: List[EdgeIR] = []
    metadata: Dict[str, Any] = {}
```

---

## 3. Detailed Pipeline Specification

Each stage represents a module in `src/diagram2drawio/pipeline/`.

### Stage A: Ingest & Normalization (`preprocess.py`)
*   **Input**: Raw image path.
*   **Logic**:
    1.  Read via `cv2.imread`.
    2.  Check resolution. If width > 4000px, downscale (maintain aspect ratio) to ~2000-3000px width for performance.
    3.  Convert to Grayscale (`cv2.COLOR_BGR2GRAY`).
*   **Output**: `(original_img, gray_img, scale_factor)`

### Stage B: Binarization & Cleaning (`preprocess.py`)
*   **Algorithm**:
    1.  Apply `cv2.fastNlMeansDenoising` or a simple Median Blur (`medianBlur(3)`).
    2.  **Adaptive Threshold**: `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ...)` is robust against lighting variations.
    3.  **Morphology**:
        *   `morphologyEx(OPEN)`: remove salt noise.
        *   `morphologyEx(CLOSE)`: close small gaps in lines.
*   **Output**: `binary_img` (0 for background, 255 for ink).

### Stage C: OCR (`ocr.py`)
*   **Backend**: `EasyOCR`. Initialize `Reader(['en'])`.
*   **Logic**:
    1.  Run `reader.readtext(gray_img)`.
    2.  Filter results where `confidence < config.ocr_confidence_threshold`.
    3.  Store results: `(bbox_poly, text, confidence)`.
*   **Optimization**: If CUDA is not available, ensure generic `cpu` mode is enforced to prevent crashes.

### Stage D: Text Removal (`text_mask.py`)
*   **Goal**: Prevent text characters from being detected as separate shapes.
*   **Logic**:
    1.  Create a blank mask.
    2.  For each OCR bbox: valid polygon? Draw `fillConvexPoly` on mask with white.
    3.  **Dilate** the mask by ~3-5px to cover anti-aliasing artifacts around text.
    4.  Apply `cv2.inpaint` (Radius=3) on the original image using this mask to "erase" text, OR simply set those pixels to background (white) in the binary map.
*   **Output**: `binary_no_text`

### Stage E: Shape Detection (`shapes.py`)
*   **Algorithm**:
    1.  `cv2.findContours(binary_no_text, RetrievalMode.EXTERNAL, ChainApproximation.SIMPLE)`.
    2.  Filter: Area < `config.min_shape_area` -> Ignore.
    3.  **Shape Classification**:
        *   `perimeter = cv2.arcLength(cnt, True)`
        *   `approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)`
        *   **If len(approx) == 4**: Check angles. If ~90 deg, it's `RECTANGLE`. If diagonals unequal but right angles, `RECTANGLE`. If diagonals bisect at right angles, `DIAMOND`.
        *   **If len(approx) > 8**: Calculate circularity `4 * pi * Area / (Perimeter^2)`. If > 0.85 -> `ELLIPSE/CIRCLE`.
        *   **Else**: Fallback to `RECTANGLE` bounding box or `ROUNDED_RECTANGLE` if corners are soft.
*   **Refinement**: Compute exact bounding rect `(x,y,w,h)` for the final IR.

### Stage F: Connector Extraction (`connectors.py`)
*   **Challenge**: Separating lines from shapes.
*   **Logic**:
    1.  Mask out (black out) all detected Node regions from `binary_no_text`.
    2.  Remaining pixels are potential arrows/lines.
    3.  **Skeletonize**: `skimage.morphology.skeletonize(binary_only_lines)`.
    4.  **Graph Building**: Convert skeleton pixels to a `networkx` graph. Nodes = pixels, Edges = 8-neighbor adjacency.
    5.  **Pruning**: Remove small disconnected components (noise).
    6.  **Path Tracing**:
        *   Identify junctions (nodes with degree > 2) and endpoints (degree == 1).
        *   Traverse paths between junctions/endpoints.
    7.  **Simplification**: Apply RDP (Ramer-Douglas-Peucker) to reduce pixel chains to vector polylines.

### Stage G: Arrowheads (`arrowheads.py`)
*   **Logic**:
    1.  Look at the "raw" contours again around the endpoints of the detected connector lines.
    2.  Triangular blobs near endpoints = Arrowheads.
    3.  **Direction**: Vector from arrow centroid to line endpoint determines flow (`Source -> Target`).

### Stage H: Association & Topology (`associate.py`)
*   **Logic**: "Snapping"
    1.  Build a spatial index (`shapely.STRtree`) of all Node polygons.
    2.  For each Edge endpoint:
        *   Query nearest Node within `config.snap_distance_threshold`.
        *   If found, strictly assign `edge.source = node.id` (or target).
        *   Adjust the edge coordinate to touch the node boundary exactly (ray casting from node center).

### Stage J: Export (`export/drawio.py`)
*   **Format**: XML structure for `draw.io`.
*   **Compression routine**:
    *   String -> UTF-8 Bytes -> `urllib.parse.quote` -> `zlib.compress` (no header, wbits=-15) -> `base64` -> XML attribute.
*   **Styling**:
    *   Map `NodeIR.shape` to standard mxGraph styles (e.g., `shape=ellipse`, `rounded=1`).
    *   Map `EdgeIR` to `edgeStyle=orthogonalEdgeStyle` if lines are rectified, or `straight` otherwise.

---

## 4. Development Workflow & Requirements

### 4.1 "Human in the Loop" Logic
The system **must** produce a standard JSON format that allows manual intervention.
*   `extract` command: Runs Stages A through I -> Dumps `diagram_ir.json`.
*   **User Action**: User opens JSON, fixes "shape_type": "unknown" or adds missing text.
*   `build` command: Reads `diagram_ir.json` -> Runs Stage J (Export).

### 4.2 Engineering Standards
*   **Type Hinting**: Strict `mypy` compliance.
*   **Error Handling**: Never crash on bad OCR. Log warning and produce an empty label.
*   **Paths**: Use `pathlib.Path` exclusively.
*   **Imports**: Absolute imports `from diagram2drawio.pipeline import ...`.

### 4.3 Testing Strategy
1.  **Unit Tests**:
    *   `test_geometry.py`: Test `approxPolyDP` wrapper on synthetic squares/circles.
    *   `test_export.py`: Verify generated XML opens in a mocked XML parser.
2.  **Integration**:
    *   Run full pipeline on `assets/examples/simple_flowchart.png`.
    *   Assert `len(ir.nodes) == expected_count`.

---

## 5. Implementation Roadmap (Task Decomposition)

### Phase 1: Skeleton & CLI
1.  Setup `pyproject.toml` with dependencies.
2.  Create `cli.py` with `extract`, `build`, `convert` stubs.
3.  Define `config.py` and `ir.py` data models.

### Phase 2: CV Pipeline (The Hard Part)
4.  Implement `preprocess.py` (load/binarize).
5.  Implement `ocr.py` (EasyOCR wrapper).
6.  Implement `text_mask.py`.
7.  Implement `shapes.py` (Contours -> IR).
8.  Implement `connectors.py` (Skeleton -> IR).

### Phase 3: Integration & Export
9.  Implement `associate.py` (Graph topology).
10. Implement `export/drawio.py` (XML Generation).
11. Wire everything into `api.py`.

### Phase 4: Polish
12. Add `arrowheads.py` logic.
13. Refine thresholds based on real-world test images.
14. Finalize documentation and packaging.
