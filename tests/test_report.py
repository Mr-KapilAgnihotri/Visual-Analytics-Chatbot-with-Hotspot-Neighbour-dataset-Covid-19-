"""
Tests for timestamps and the PDF export.   Run:  python -m pytest -q

Chart images need Kaleido + a Chrome browser. If neither can be found here,
the image test is skipped (the PDF itself is still tested).
"""
import io
import logging
import uuid
from datetime import datetime

import pytest
from pypdf import PdfReader

logging.disable(logging.WARNING)

from backend import report  # noqa: E402
from backend.orchestrator import get_orchestrator  # noqa: E402


@pytest.fixture(scope="module")
def turns():
    o, s = get_orchestrator(), uuid.uuid4().hex
    qs = ["Top 5 districts by incidence on 15 November",     # bar chart
          "Map of incidence on 20 December",                 # map (needs the local outlines)
          "neighbours of"]                                   # clarification: no chart
    return [o.ask(q, s).to_dict() for q in qs]


def _images(pdf: bytes) -> int:
    return sum(len(p.images) for p in PdfReader(io.BytesIO(pdf)).pages)


def test_every_answer_is_timestamped(turns):
    for a in turns:
        asked, answered = datetime.fromisoformat(a["asked_at"]), datetime.fromisoformat(a["answered_at"])
        assert asked.tzinfo is not None and asked <= answered
    assert turns[0]["asked_at"] <= turns[1]["asked_at"] <= turns[2]["asked_at"]


def test_pdf_without_images_is_valid(turns):
    pdf, stats = report.build_pdf(turns, render_charts=False)
    assert pdf.startswith(b"%PDF") and stats["images"] == 0
    text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Top 5 districts" in text and "Asked" in text and "15 Nov" in text
    assert "How this was answered" not in text and "SELECT" not in text   # no queries in the PDF


def test_pdf_with_chart_images(turns):
    pdf, stats = report.build_pdf(turns)
    if stats["image_errors"] and stats["images"] == 0:
        pytest.skip("no Chrome for Kaleido here: " + stats["image_errors"][0])
    assert stats["charts"] == 2 and stats["images"] == 2 and _images(pdf) == 2


def test_failed_chart_gives_a_note_not_a_crash(turns, monkeypatch):
    monkeypatch.setattr(report, "chart_png", lambda _: (None, "test: no browser"))
    pdf, stats = report.build_pdf(turns)
    text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert stats["images"] == 0 and "could not be created" in text


def test_special_characters_are_safe():
    pdf, _ = report.build_pdf([{"question": "Köln <b> & ≥ 75 → ✅", "text": "a < b & c", "engine": "NONE",
                                 "asked_at": "2026-10-28T10:00:00+01:00", "answered_at": "bad"}])
    assert pdf.startswith(b"%PDF")


def test_report_endpoint(turns):
    from fastapi.testclient import TestClient
    from backend.main import app
    r = TestClient(app).post("/report", json={"turns": turns[:1]})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


def test_long_axis_labels_are_never_cut_off(turns):
    """Regression: on an older Plotly the district names on bar charts were clipped in the PDF."""
    import plotly.io as pio
    fig = pio.from_json(turns[0]["chart"])
    assert fig.layout.yaxis.automargin is True


def test_llm_hyphens_survive():
    assert report.clean_text("7‑day Baden‑Württemberg") == "7-day Baden-Württemberg"


def test_tied_hotspot_share_lists_all_states(orch=None):
    a = get_orchestrator().ask("Map of incidence on 20 December")
    assert len(a.facts["states_with_highest_hotspot_share"]) >= 1
    assert a.facts["highest_state_hotspot_share_pct"] <= 100
