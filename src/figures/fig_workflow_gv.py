"""The methodology flowchart, drawn by Graphviz and exported for draw.io.

Produces
    results/figures/fig_workflow.png        600 dpi
    results/figures/fig_workflow.pdf        vector, fonts embedded
    results/figures/fig_workflow.drawio.xml editable in app.diagrams.net

    python src/figures/fig_workflow_gv.py

One spec, two renderers. NODES and EDGES below are the single description of
the diagram; graphviz_source() turns them into DOT and drawio_xml() turns the
same lists into an mxGraphModel. Editing the diagram in draw.io and editing it
here therefore stay convertible in one direction - if the .xml is hand-edited,
those edits live only in the .xml, so treat the .xml as a starting point for
the copy-editor rather than as the master.

The counts inside the boxes come from facts() in fig_flowchart, which reads
them out of the pipeline outputs. A re-run of the analysis changes the diagram
without anyone having to remember to retype a number.
"""

from pathlib import Path
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import style as S  # noqa: E402

# Lane fills. Kept pale: the boxes sit on top of them and the box text has to
# stay readable in greyscale print.
LANES = [
    ("Input", "#dbe7f3", [0]),
    ("Pre-processing", "#e8eef5", [1]),
    ("Sampling", "#dff0e2", [2]),
    ("Modelling", "#f6ecd8", [3, 4]),
    ("Assessment", "#f7e3e8", [5, 6, 7]),
    ("Output", "#ececec", [8]),
]

ACCENT = "#8c6d1f"   # the one feedback edge
EDGE = "#444444"


def build(f):
    """Return (nodes, edges) with the pipeline's own numbers filled in.

    `f` is the dict from fig_flowchart.facts().
    """
    n_models = len(f["models"])
    models_a = ", ".join(f["models"][:3])
    models_b = ", ".join(f["models"][3:])

    # id, row, col, span, title, body lines
    nodes = [
        ("s2", 0, 0, 1, "Sentinel-2 MSI",
         [C.S2_COLLECTION.split("/")[-1],
          f"{C.DATE_START} to {C.DATE_END}",
          f"cloud cover < {C.MAX_CLOUD_PCT}%"]),
        ("roi", 0, 1, 1, "Training polygons",
         [f"{f['n_poly'] or 110} single-part polygons",
          f"{len(C.CLASS_NAMES)} classes, field/visual labels",
          f"in the '{C.CLASS_PROPERTY}' property"]),
        ("bnd", 0, 2, 1, "City boundary",
         ["Taipei City administrative", "polygon"]),

        ("comp", 1, 0, 1, "Median composite",
         [f"bands {'/'.join(C.BANDS_PUBLISHED)}",
          f"{C.SCALE} m, {C.EXPORT_CRS}", "clipped to the boundary"]),
        ("parts", 1, 1, 1, "Explode to single parts",
         ["MultiPolygon parts split so that",
          "one polygon is one sampling cluster"]),

        ("samp", 2, 0, 1, "sampleRegions",
         [f"on the native {C.EXPORT_CRS} {C.SCALE} m grid",
          f"{f['n_px'] or 'n'} labelled pixels"]),
        ("split", 2, 1, 2, "Polygon-level split",
         [f"train {f['poly_train'] or ''} polygons / {f['n_train'] or ''} px",
          f"val {f['poly_val'] or ''} polygons / {f['n_val'] or ''} px",
          "stratified by class"]),

        ("cv", 3, 0, 1, "Grouped CV tuning",
         [f"{C.CV_FOLDS}-fold, groups = polygon",
          "RandomizedSearchCV",
          f"{C.N_ITER_SEARCH} draws per model"]),
        ("clf", 3, 1, 2, f"{n_models} classifiers", [models_a, models_b]),
        ("predict", 4, 0, 3, "Classify every city pixel",
         [f"{n_models} wall-to-wall maps at {C.SCALE} m"]),

        ("hold", 5, 0, 1, "Hold-out accuracy",
         ["OA, kappa, per-class UA/PA",
          f"{C.N_BOOTSTRAP} bootstrap CIs"]),
        ("mcn", 5, 1, 2, "Pairwise significance",
         ["McNemar, naive and", "polygon-cluster corrected"]),
        ("diag", 6, 0, 1, "Diagnostics",
         ["Jeffries-Matusita separability",
          "error concentration, design effect"]),
        ("olof", 6, 1, 2, "Area-weighted accuracy",
         ["Olofsson et al. (2014)",
          "strata weights from each model's own map"]),
        ("ref", 7, 0, 3, "Blind stratified reference sample",
         [f"{f['n_ref']} points" if f["n_ref"] else "interpreted independently"]),

        ("out", 8, 0, 3, "Outputs",
         [f"{n_models} classified maps  |  per-model class areas  |  "
          "accuracy and area CIs"]),
    ]

    edges = [
        ("s2", "comp", "", False),
        ("roi", "parts", "", False),
        ("bnd", "comp", "", False),
        ("comp", "samp", "", False),
        ("parts", "samp", "", False),
        ("samp", "split", "", False),
        ("split", "cv", "", False),
        ("split", "clf", "", False),
        ("cv", "clf", "best settings", False),
        ("clf", "predict", "", False),
        ("predict", "hold", "", False),
        ("predict", "mcn", "", False),
        ("hold", "diag", "", False),
        ("mcn", "olof", "", False),
        ("diag", "ref", "", False),
        ("olof", "ref", "", False),
        ("ref", "out", "", False),
        # The one feedback edge worth drawing: the map areas are the strata
        # weights, so the area-weighted estimate is not independent of the map
        # it is assessing.
        ("predict", "olof", "areas = weights", True),
    ]
    return nodes, edges


# ---------------------------------------------------------------- graphviz


def _lane_of(row):
    for name, fill, rows in LANES:
        if row in rows:
            return name, fill
    return "", "#ffffff"


def _lane_start(row):
    """The first row of the lane `row` belongs to."""
    for _, _, rows in LANES:
        if row in rows:
            return rows[0]
    return row


def _html_label(title, lines, width_pt):
    """A two-tier label: bold title over small body lines."""
    head = f'<B>{escape(title)}</B>'
    if not lines:
        body = ""
    else:
        body = ('<BR/><FONT POINT-SIZE="7.0" COLOR="#333333">'
                + "<BR/>".join(escape(t) for t in lines if t) + "</FONT>")
    return f'<<TABLE BORDER="0" CELLPADDING="3"><TR><TD WIDTH="{width_pt}">' \
           f'{head}{body}</TD></TR></TABLE>>'


def graphviz_source(nodes, edges):
    """DOT source for the flowchart.

    No cluster subgraphs. Inside a cluster dot ignores the flat (same-rank)
    edges that would otherwise pin the left-to-right order of a row, so the
    boxes came out in whatever order minimised crossings - "City boundary"
    landed between "Sentinel-2" and "Training polygons", and the assessment
    row read right-to-left. Without clusters the flat edges are honoured, so
    the stage is carried by the box fill colour and by a stage name in a
    left-hand column instead of by a coloured band. That also removes the
    staircase: cluster margins were pushing each lane further right than the
    one above it.
    """
    by_row = {}
    for nid, row, col, span, title, lines in nodes:
        by_row.setdefault(row, []).append((col, nid, span, title, lines))

    L = []
    L.append("digraph workflow {")
    # splines=ortho is tempting for a flowchart but Graphviz cannot place edge
    # labels on orthogonal edges - it warns and drops them. polyline keeps the
    # square look without losing the two labelled edges.
    L.append('  graph [rankdir=TB, splines=polyline, nodesep=0.26, '
             'ranksep=0.40, fontname="Helvetica", bgcolor="white"];')
    L.append('  node  [shape=box, style="filled,rounded", '
             'color="#7a7a7a", penwidth=0.7, fontname="Helvetica", '
             'fontsize=9, margin="0.07,0.05"];')
    L.append(f'  edge  [color="{EDGE}", penwidth=0.7, arrowsize=0.5, '
             'fontname="Helvetica", fontsize=7.5];')

    for row in sorted(by_row):
        items = sorted(by_row[row])
        lane, fill = _lane_of(row)
        tag = f"stage{row}"
        # One stage node per row keeps the left-hand column the same width on
        # every rank; only the row that opens a lane carries the text.
        if row == _lane_start(row):
            L.append(f'  {tag} [label="{lane}", shape=box, style="filled", '
                     f'fillcolor="{fill}", color="{fill}", fontsize=8.5, '
                     f'fontcolor="#333333", width=0.95, height=0.30];')
        else:
            # Invisible, not a blank coloured chip: an unlabelled swatch in
            # the margin reads as a legend entry that lost its text. It still
            # takes up its width in the layout, which is all it is for.
            L.append(f'  {tag} [label="", style=invis, width=0.95, '
                     f'height=0.16];')
        L.append(f'  {{ rank=same; {tag};')
        for col, nid, span, title, lines in items:
            w = 155 if span == 1 else 250
            L.append(f'    {nid} [label={_html_label(title, lines, w)}, '
                     f'fillcolor="{fill}"];')
        chain = [tag] + [it[1] for it in items]
        for a, b in zip(chain, chain[1:]):
            L.append(f'    {a} -> {b} [style=invis];')
        L.append("  }")

    # Anchor the left-hand column: the stage nodes form a straight vertical
    # spine, so the rows cannot slide sideways relative to one another.
    stages = [f"stage{r}" for r in sorted(by_row)]
    for a, b in zip(stages, stages[1:]):
        L.append(f'  {a} -> {b} [style=invis, weight=200];')

    for a, b, label, dashed in edges:
        attrs = []
        if label:
            attrs.append(f'label=" {label} "')
        if dashed:
            attrs.append(f'style=dashed, color="{ACCENT}", '
                         f'fontcolor="{ACCENT}", constraint=false')
        L.append(f'  {a} -> {b}' + (f' [{", ".join(attrs)}]' if attrs else "")
                 + ";")
    L.append("}")
    return "\n".join(L)


def render(dot_src, stem, dpi=600, width_in=5.51):
    """Write PNG and PDF next to the other figures.

    width_in defaults to 140 mm, the Elsevier 1.5-column width; `size` only
    ever shrinks a drawing in Graphviz, so an over-wide layout is scaled to
    fit and a narrower one is left alone.
    """
    out = C.FIGURES
    out.mkdir(parents=True, exist_ok=True)
    dot_path = out / f"{stem}.gv"
    dot_path.write_text(dot_src, encoding="utf-8")
    if shutil.which("dot") is None:
        raise SystemExit("graphviz 'dot' is not on PATH "
                         "(apt-get install graphviz / brew install graphviz)")
    for fmt, extra in (("pdf", []), ("png", [f"-Gdpi={dpi}"])):
        subprocess.run(["dot", f"-T{fmt}", f"-Gsize={width_in},11",
                        *extra, str(dot_path), "-o", str(out / f"{stem}.{fmt}")],
                       check=True)
    for fmt in ("png", "pdf"):
        p = out / f"{stem}.{fmt}"
        print(f"  wrote {p.name} ({p.stat().st_size / 1024:.0f} KB)")


# ------------------------------------------------------------------ draw.io

COL_W, NODE_W, GAP = 250, 220, 30
ROW_H, NODE_H = 104, 76
PAD_X, PAD_Y = 28, 34
LANE_W = 3 * COL_W - (COL_W - NODE_W) + 2 * PAD_X


def _geom(row, col, span, lane_top):
    x = PAD_X + col * COL_W
    w = NODE_W + (span - 1) * COL_W
    y = lane_top + (row - _first_row(row)) * ROW_H + (ROW_H - NODE_H) / 2
    return x, y, w, NODE_H


def _first_row(row):
    for _, _, rows in LANES:
        if row in rows:
            return rows[0]
    return row


def drawio_xml(nodes, edges):
    """An mxGraphModel of the same diagram, laid out on the row/column grid.

    Graphviz places the boxes itself; draw.io needs explicit coordinates, so
    the grid the spec is written on is used directly. Lanes are plain
    background rectangles rather than swimlane containers: a swimlane makes
    every box a child of the lane, and moving a box between lanes in the
    editor then silently reparents it.
    """
    lane_top, tops = PAD_Y, {}
    lane_cells = []
    for lane, fill, rows in LANES:
        h = len(rows) * ROW_H
        for r in rows:
            tops[r] = lane_top
        lane_cells.append((lane, fill, lane_top, h))
        lane_top += h + 10

    x = ['<mxfile host="app.diagrams.net" type="device">',
         '  <diagram id="workflow" name="Workflow">',
         f'    <mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" '
         f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
         f'pageScale="1" pageWidth="850" pageHeight="1100" math="0" '
         f'shadow="0">',
         '      <root>',
         '        <mxCell id="0" />',
         '        <mxCell id="1" parent="0" />']

    def cell(cid, value, style, geo, edge=False, src=None, tgt=None):
        attrs = (f'id="{cid}" value="{escape(value, {chr(34): "&quot;"})}" '
                 f'style="{style}" parent="1"')
        attrs += ' edge="1"' if edge else ' vertex="1"'
        if src:
            attrs += f' source="{src}"'
        if tgt:
            attrs += f' target="{tgt}"'
        g = ("" if geo is None else
             f'<mxGeometry x="{geo[0]:.0f}" y="{geo[1]:.0f}" '
             f'width="{geo[2]:.0f}" height="{geo[3]:.0f}" as="geometry" />')
        if geo is None:
            g = '<mxGeometry relative="1" as="geometry" />'
        x.append(f'        <mxCell {attrs}>')
        x.append(f'          {g}')
        x.append('        </mxCell>')

    # Lanes first so they sit behind the boxes (draw.io z-order is document
    # order).
    for i, (lane, fill, top, h) in enumerate(lane_cells):
        cell(f"lane{i}", lane,
             f"rounded=1;arcSize=4;fillColor={fill};strokeColor=none;"
             "verticalAlign=top;align=left;spacingLeft=8;spacingTop=2;"
             "fontSize=10;fontColor=#444444;html=1;",
             (0, top, LANE_W, h))

    for nid, row, col, span, title, lines in nodes:
        gx, gy, gw, gh = _geom(row, col, span, tops[row])
        body = "".join("<br/><font style='font-size:9px' color='#333333'>"
                       f"{t}</font>" for t in lines if t)
        # draw.io stores an html=1 label as ordinary XML attribute text, so
        # the whole markup string is escaped once here; the editor unescapes
        # it and then renders the tags.
        value = escape(f"<b>{title}</b>{body}", {'"': "&quot;"})
        x.append(
            f'        <mxCell id="{nid}" value="{value}" '
            f'style="rounded=1;arcSize=8;fillColor=#ffffff;'
            f'strokeColor=#666666;strokeWidth=0.7;html=1;whiteSpace=wrap;'
            f'align=center;verticalAlign=middle;fontSize=11;spacing=4;" '
            f'vertex="1" parent="1">')
        x.append(f'          <mxGeometry x="{gx:.0f}" y="{gy:.0f}" '
                 f'width="{gw:.0f}" height="{gh:.0f}" as="geometry" />')
        x.append('        </mxCell>')

    for i, (a, b, label, dashed) in enumerate(edges):
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
                 "endArrow=block;endFill=1;endSize=5;")
        style += (f"dashed=1;strokeColor={ACCENT};fontColor={ACCENT};"
                  if dashed else f"strokeColor={EDGE};")
        style += "fontSize=9;"
        cell(f"e{i}", label, style, None, edge=True, src=a, tgt=b)

    x += ['      </root>', '    </mxGraphModel>', '  </diagram>', '</mxfile>']
    return "\n".join(x)


def main():
    S.use()
    import fig_flowchart as FF
    f = FF.facts()
    nodes, edges = build(f)

    print("fig_workflow (graphviz)")
    render(graphviz_source(nodes, edges), "fig_workflow")

    p = C.FIGURES / "fig_workflow.drawio.xml"
    p.write_text(drawio_xml(nodes, edges), encoding="utf-8")
    print(f"  wrote {p.name} ({p.stat().st_size / 1024:.0f} KB) "
          "- open at app.diagrams.net")


if __name__ == "__main__":
    main()
