"""
PDF EXPORT: turns a conversation into a printable report, charts included.

    build_pdf(turns, meta) -> bytes

`turns` is the list of answers exactly as the API / chat screen holds them
(the dicts returned by Answer.to_dict(), which carry asked_at / answered_at).

How the pieces fit
------------------
  * ReportLab (platypus) lays out the document: headings, paragraphs, tables,
    images, page numbers. It is pure Python, so it works the same on Windows.
  * Each chart is still a Plotly figure (JSON). Kaleido, Plotly's official image
    exporter, renders it to a PNG by drawing it in a headless Chrome browser.
    If no Chrome is found, Plotly can download one once (pio.get_chrome()).
  * Maps need country outlines (TopoJSON). We serve the bundled copy in
    frontend/static/topojson from a tiny local web server, so rendering a map
    never depends on Plotly's website.
  * If a chart cannot be rendered (no browser, no internet for the one-time
    download), the PDF is still produced, with a short note in place of the image.
"""
from __future__ import annotations

import functools
import http.server
import io
import re
import logging
import socketserver
import threading
from datetime import datetime
from xml.sax.saxutils import escape

import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from backend import config

log = logging.getLogger(__name__)
for _noisy in ("kaleido", "choreographer", "logistro"):   # the image exporter chats a lot at INFO
    logging.getLogger(_noisy).setLevel(logging.WARNING)

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN
IMG_PX_W = 1000                      # render width in pixels (scale 2 for sharp print)
ENGINE_NAMES = {"SQL": "SQL (DuckDB)", "GRAPH": "Knowledge graph ({g})",
                "HYBRID": "Hybrid ({g} + DuckDB)", "NONE": "Assistant"}
INK = colors.HexColor("#1f1f1d")
MUTED = colors.HexColor("#6b6a65")
RULE = colors.HexColor("#d9d8d2")
ACCENT = colors.HexColor("#2a78d6")

# The built-in PDF fonts (Helvetica) cover Western European text (ä, ö, ü, ß ...)
# but not every symbol. Replace the few symbols our answers use.
_SUBST = {"≥": ">=", "≤": "<=", "→": "->", "↪": "->", "✅": "", "×": "x", "−": "-",
          "\u2009": " ", "\u202f": " ", "\u2007": " ", "\u200b": "", "\u2248": "~",
          "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2043": "-"}     # the LLM likes unbreakable hyphens


def engine_name(a: dict) -> str:
    g = "Neo4j" if a.get("graph_backend") == "neo4j" else "NetworkX"
    return ENGINE_NAMES.get(a.get("engine", "NONE"), str(a.get("engine", ""))).format(g=g)


def clean_text(text) -> str:
    s = "" if text is None else str(text)
    for a, b in _SUBST.items():
        s = s.replace(a, b)
    return s.encode("cp1252", errors="replace").decode("cp1252")


def _p(text) -> str:
    """Text made safe for a ReportLab Paragraph (which reads a little XML)."""
    return escape(clean_text(text))


def _rich(text) -> str:
    """Like _p, but keeps the chat's **bold** and line breaks."""
    s = _p(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")


def fmt_time(iso: str | None, with_date: bool = True) -> str:
    if not iso:
        return ""
    try:
        t = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return t.strftime("%d %b %Y, %H:%M:%S" if with_date else "%H:%M:%S")


def seconds_between(a: str | None, b: str | None) -> float | None:
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# chart -> PNG
# ---------------------------------------------------------------------------
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):          # keep the console clean
        pass


@functools.lru_cache(maxsize=1)
def _topojson_url() -> str:
    """Start (once) a tiny local web server for the bundled map outlines."""
    folder = config.ROOT / "frontend" / "static" / "topojson"
    handler = functools.partial(_Quiet, directory=str(folder))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}/"


_chrome_tried = False


def chart_png(chart_json: str) -> tuple[bytes | None, str | None]:
    """Render one Plotly chart (JSON) to PNG bytes. Returns (png, error_message)."""
    global _chrome_tried
    try:
        fig = pio.from_json(chart_json)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable chart ({exc.__class__.__name__})"
    # printed on white paper: solid white background, dark text
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1f1f1d"))
    fig.update_xaxes(automargin=True)            # never cut off long district names on the axes
    fig.update_yaxes(automargin=True)
    height = int(fig.layout.height or 460)
    if any(t.type == "scattergeo" for t in fig.data):
        pio.defaults.topojson = _topojson_url()
    for attempt in range(2):
        try:
            return fig.to_image(format="png", width=IMG_PX_W, height=height, scale=2), None
        except Exception as exc:  # noqa: BLE001
            msg = f"{exc.__class__.__name__}: {exc}"
            needs_chrome = "chrome" in msg.lower() or "browser" in msg.lower()
            if attempt == 0 and needs_chrome and not _chrome_tried:
                _chrome_tried = True
                try:                                      # one-time download of a headless Chrome
                    log.warning("No Chrome found for chart images; downloading one (one time only)...")
                    pio.get_chrome()
                    continue
                except Exception as exc2:  # noqa: BLE001
                    msg = f"could not obtain Chrome ({exc2.__class__.__name__})"
            log.warning("Chart image failed: %s", msg)
            return None, msg[:200]
    return None, "chart image failed"


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20,
                                leading=24, textColor=INK, alignment=TA_LEFT, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                             leading=16, textColor=INK, spaceBefore=4, spaceAfter=3),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=10,
                               leading=14, textColor=INK, spaceAfter=6),
        "meta": ParagraphStyle("m", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.5,
                               leading=11, textColor=MUTED, spaceAfter=4),
        "note": ParagraphStyle("n", parent=ss["BodyText"], fontName="Helvetica-Oblique", fontSize=9,
                               leading=12, textColor=MUTED, spaceAfter=6),
        "cell": ParagraphStyle("c", parent=ss["BodyText"], fontName="Helvetica", fontSize=8,
                               leading=10, textColor=INK),
        "cellh": ParagraphStyle("ch", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=8,
                                leading=10, textColor=INK),
    }


def _table(rows: list[list], st, col_widths=None, header=True) -> Table:
    data = [[Paragraph(_p(c), st["cellh"] if (header and i == 0) else st["cell"]) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK if header else RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return "" if v is None else str(v)


def _data_table(rows: list[dict], st, max_rows: int = 12, max_cols: int = 6):
    if not rows:
        return None
    cols = [c for c in rows[0].keys() if c not in ("lat", "lon")][:max_cols]
    body = [[_cell(r.get(c)) for c in cols] for r in rows[:max_rows]]
    out = [_table([cols] + body, st)]
    if len(rows) > max_rows:
        out.append(Paragraph(_p(f"... {len(rows) - max_rows} more rows (see the chat for all rows)"), st["meta"]))
    return out


def _on_page(canvas, doc, title: str):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 1.1 * cm, clean_text(title))
    canvas.drawRightString(PAGE_W - MARGIN, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(turns: list[dict], meta: dict | None = None, render_charts: bool = True) -> tuple[bytes, dict]:
    """Build the report. Returns (pdf_bytes, stats) where stats counts charts rendered/failed."""
    meta = meta or {}
    st = _styles()
    title = meta.get("title", "VA Chatbot - conversation report")
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    stats = {"turns": len(turns), "charts": 0, "images": 0, "image_errors": []}
    story: list = []

    # ---- cover: what this is, and a one-line index of every question ----
    story.append(Paragraph(_p(title), st["title"]))
    story.append(Paragraph(_p(
        f"COVID-19 in Germany's 411 districts, 2020 · generated {fmt_time(generated)} · "
        f"{len(turns)} question{'s' if len(turns) != 1 else ''}"
        + (f" · graph: {meta['graph']}" if meta.get("graph") else "")
        + (f" · language model: {meta['llm']}" if meta.get("llm") else "")), st["meta"]))
    story.append(Spacer(1, 8))
    if turns:
        index = [["#", "Asked at", "Question", "Answered by"]]
        for i, a in enumerate(turns, 1):
            index.append([str(i), fmt_time(a.get("asked_at")), a.get("question", ""),
                          engine_name(a)])
        story.append(_table(index, st, [0.8 * cm, 3.6 * cm, CONTENT_W - 9.4 * cm, 5.0 * cm]))
    else:
        story.append(Paragraph("No questions were asked in this conversation.", st["body"]))

    # ---- one section per question ----
    for i, a in enumerate(turns, 1):
        story.append(PageBreak())
        head = [Paragraph(_p(f"{i}. {a.get('question', '')}"), st["h2"])]
        took = seconds_between(a.get("asked_at"), a.get("answered_at"))
        meta_line = (f"Asked {fmt_time(a.get('asked_at'))}"
                     + (f" · answered in {took:.1f} s" if took is not None else "")
                     + f" · {engine_name(a)}"
                     + (f" · question type: {a['intent']}" if a.get("intent") else ""))
        head.append(Paragraph(_p(meta_line), st["meta"]))
        if a.get("followup"):
            head.append(Paragraph(_p("Follow-up: " + a["followup"].replace("Follow-up to ", "to ", 1)), st["note"]))
        head.append(Paragraph(_rich(a.get("text", "")), st["body"]))
        story.append(KeepTogether(head))

        if a.get("kpis"):
            story.append(_table([[k["label"] for k in a["kpis"]], [k["value"] for k in a["kpis"]]], st))
            story.append(Spacer(1, 6))

        if a.get("chart") and render_charts:
            stats["charts"] += 1
            png, err = chart_png(a["chart"])
            if png:
                img = Image(io.BytesIO(png))
                ratio = img.imageHeight / img.imageWidth
                w = CONTENT_W
                h = w * ratio
                max_h = PAGE_H - 2 * MARGIN - 4 * cm
                if h > max_h:                               # tall maps: shrink to fit the page
                    h, w = max_h, max_h / ratio
                img.drawWidth, img.drawHeight = w, h
                story.append(img)
                stats["images"] += 1
            else:
                stats["image_errors"].append(err)
                story.append(Paragraph(_p(f"[Chart image could not be created here: {err}. "
                                          "The interactive chart is in the chat.]"), st["note"]))
            story.append(Spacer(1, 6))
        elif a.get("chart"):
            stats["charts"] += 1

        tbl = _data_table(a.get("table") or [], st)
        if tbl:
            story.append(Paragraph("Data behind the chart", st["cellh"]))
            story.append(Spacer(1, 2))
            story.extend(tbl)
            story.append(Spacer(1, 6))


    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN, title=clean_text(title),
                            author="VA Chatbot", subject="Conversation report")
    doc.build(story, onFirstPage=lambda c, d: _on_page(c, d, title),
              onLaterPages=lambda c, d: _on_page(c, d, title))
    return buf.getvalue(), stats
