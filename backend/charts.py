"""
VISUALISATION: turns query results into Plotly figures (sent to the UI as JSON).

Chart choice follows the job the data does:
    ranking                 -> horizontal bar (sorted, longest on top)
    parts of a whole        -> donut (top 6 + "Other"; more slices are unreadable)
    change over time        -> line (one line per place, max 8)
    two measures            -> scatter + least-squares trend line
    where (geography)       -> map of district points, colour = magnitude
    relationships / borders -> network drawn on the map (nodes + border edges)

Colour rules (from a colour-blind-validated palette):
    * categorical hues are used in a FIXED order, never cycled
    * magnitude uses ONE hue, light -> dark (never a rainbow)
    * red is reserved for "hotspot" and always comes with a text label in the legend
    * one y-axis per chart, recessive grid lines, thin marks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.config import HOTSPOT_THRESHOLD

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_SCALE = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]
HOTSPOT = "#d03b3b"          # status "critical"
FOCUS = SERIES[0]
CONTEXT = "#a9a8a1"          # neutral gray for context marks
MUTED = "#898781"
GRID = "rgba(137,135,129,0.22)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
MAX_LINES = 8


def _layout(fig: go.Figure, title: str, x_title: str = "", y_title: str = "", height: int = 460,
            legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yref="container", yanchor="top", font=dict(size=16)),
        font=dict(family=FONT, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=95 if legend else 60, b=40), height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font=dict(family=FONT)),
        bargap=0.25,
    )
    fig.update_xaxes(title_text=x_title, gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED),
                     automargin=True)
    fig.update_yaxes(title_text=y_title, gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED),
                     automargin=True)
    return fig


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


# ---------------------------------------------------------------------------
def bar_ranking(df: pd.DataFrame, title: str, value_label: str, color_mode: str = "focus",
                focus_names: list | None = None, hotspot_col: str | None = None) -> go.Figure:
    """Horizontal ranking bars. color_mode: 'focus' (one hue), 'hotspot' (status), 'highlight'."""
    d = df.sort_values("value", ascending=True)
    names, values = d["name"].tolist(), d["value"].tolist()
    if color_mode == "hotspot":
        hot = d[hotspot_col] if hotspot_col else d["value"] >= HOTSPOT_THRESHOLD
        fig = go.Figure()
        for flag, color, label in ((True, HOTSPOT, f"Hotspot (≥ {HOTSPOT_THRESHOLD})"),
                                   (False, CONTEXT, "Not a hotspot")):
            sub = d[hot == flag]
            if len(sub):
                fig.add_bar(y=sub["name"], x=sub["value"], orientation="h", name=label,
                            marker=dict(color=color, cornerradius=4),
                            text=[fmt(v) for v in sub["value"]], textposition="outside",
                            hovertemplate="%{y}<br>" + value_label + ": %{x:,.1f}<extra></extra>")
        fig.update_yaxes(categoryorder="array", categoryarray=names)
        if focus_names:
            fig.update_yaxes(ticktext=[f"<b>{n}</b>" if n in focus_names else n for n in names], tickvals=names)
    else:
        colors = [FOCUS if (not focus_names or n in focus_names) else CONTEXT for n in names]
        fig = go.Figure(go.Bar(y=names, x=values, orientation="h", marker=dict(color=colors, cornerradius=4),
                               text=[fmt(v) for v in values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>" + value_label + ": %{x:,.1f}<extra></extra>"))
    h = max(320, 34 * len(d) + 120)
    return _layout(fig, title, value_label, "", height=h, legend=color_mode == "hotspot")


def donut(df: pd.DataFrame, title: str, value_label: str, top: int = 6) -> go.Figure:
    d = df.sort_values("value", ascending=False)
    if len(d) > top + 1:
        other = pd.DataFrame([{"name": f"Other ({len(d) - top} states)", "value": d["value"].iloc[top:].sum()}])
        d = pd.concat([d.head(top), other])
    colors = SERIES[: len(d) - 1] + [CONTEXT] if len(d) > top else SERIES[: len(d)]
    fig = go.Figure(go.Pie(labels=d["name"], values=d["value"], hole=0.55, sort=False, direction="clockwise",
                           marker=dict(colors=colors, line=dict(color="rgba(255,255,255,0.9)", width=2)),
                           textinfo="percent", textposition="outside",
                           hovertemplate="%{label}<br>" + value_label + ": %{value:,.0f}<br>%{percent}<extra></extra>"))
    fig = _layout(fig, title, height=460)
    fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"))
    return fig


def lines(df: pd.DataFrame, title: str, value_label: str, threshold: float | None = None) -> go.Figure:
    """df: period, series, value."""
    fig = go.Figure()
    order = (df.groupby("series")["value"].max().sort_values(ascending=False).index.tolist())[:MAX_LINES]
    for i, s in enumerate(order):
        sub = df[df["series"] == s].sort_values("period")
        fig.add_scatter(x=sub["period"], y=sub["value"], mode="lines", name=s,
                        line=dict(color=SERIES[i], width=2),
                        hovertemplate="%{x|%d %b %Y}<br>" + s + ": %{y:,.1f}<extra></extra>")
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=HOTSPOT, width=1, dash="dot"),
                      annotation_text=f"Hotspot threshold ({threshold:g})", annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
    fig = _layout(fig, title, "", value_label, legend=len(order) > 1)
    fig.update_layout(hovermode="x unified")
    return fig


def scatter(df: pd.DataFrame, title: str, x_label: str, y_label: str,
            point_label: str = "District") -> tuple[go.Figure, dict]:
    d = df.dropna(subset=["x", "y"])
    fig = go.Figure(go.Scatter(x=d["x"], y=d["y"], mode="markers+text" if len(d) <= 20 else "markers",
                               name=point_label, textposition="top center", textfont=dict(size=10, color=MUTED),
                               marker=dict(size=9, color=FOCUS, opacity=0.7,
                                           line=dict(color="rgba(255,255,255,0.9)", width=1)),
                               text=d["name"], customdata=d["state"],
                               hovertemplate="%{text} (%{customdata})<br>" + x_label + ": %{x:,.1f}<br>"
                                             + y_label + ": %{y:,.1f}<extra></extra>"))
    stats = {"n": int(len(d)), "r": None, "slope": None}
    if len(d) > 2 and d["x"].std() > 0 and d["y"].std() > 0:
        r = float(np.corrcoef(d["x"], d["y"])[0, 1])
        slope, intercept = np.polyfit(d["x"], d["y"], 1)
        xs = np.linspace(d["x"].min(), d["x"].max(), 50)
        fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines", name=f"Trend (r = {r:.2f})",
                        line=dict(color=SERIES[1], width=2), hoverinfo="skip")
        stats.update(r=r, slope=float(slope))
    return _layout(fig, title, x_label, y_label), stats


# ---------------------------------------------------------------------------
# maps (Scattergeo: works offline, no map tiles needed)
# ---------------------------------------------------------------------------
def _geo(fig: go.Figure, title: str, height: int = 620, legend: bool = True) -> go.Figure:
    fig.update_geos(scope="europe", fitbounds="locations", projection_type="mercator", showcountries=True,
                    countrycolor=MUTED, showcoastlines=True, coastlinecolor=MUTED,
                    showland=True, landcolor="rgba(137,135,129,0.07)", showocean=False, showlakes=False,
                    bgcolor="rgba(0,0,0,0)", resolution=50, showframe=False)
    fig = _layout(fig, title, height=height, legend=legend)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.01, xanchor="left", x=0))
    return fig


def value_map(df: pd.DataFrame, title: str, value_label: str) -> go.Figure:
    """df: lat, lon, name, value, population."""
    d = df.copy()
    size = 6 + 16 * np.sqrt(d["population"] / d["population"].max())
    fig = go.Figure(go.Scattergeo(
        lat=d["lat"], lon=d["lon"], text=d["name"], customdata=np.stack([d["value"], d["state"]], axis=1),
        mode="markers", name=value_label,
        marker=dict(size=size, color=d["value"], colorscale=SEQ_SCALE, cmin=0,
                    line=dict(color="rgba(255,255,255,0.8)", width=0.8),
                    colorbar=dict(title=dict(text=value_label, side="right"), thickness=12, outlinewidth=0)),
        hovertemplate="%{text} (%{customdata[1]})<br>" + value_label + ": %{customdata[0]:,.1f}<extra></extra>"))
    return _geo(fig, title, legend=False)


def network_map(nodes: pd.DataFrame, edges: list, title: str, color_by: str = "role",
                value_label: str = "", roles: dict | None = None, background_edges: list | None = None,
                height: int = 620) -> go.Figure:
    """
    nodes: district_id, name, state, lat, lon, [role], [value]
    edges: [(a, b), ...] drawn as border links
    color_by: 'role'  -> discrete groups given by nodes['role'] and roles={role: (color, label)}
              'value' -> sequential colour by nodes['value']
    """
    pos = nodes.set_index("district_id")[["lat", "lon"]].to_dict("index")
    fig = go.Figure()

    def edge_trace(pairs, color, width, name, show):
        lat, lon = [], []
        for a, b in pairs:
            if a in pos and b in pos:
                lat += [pos[a]["lat"], pos[b]["lat"], None]
                lon += [pos[a]["lon"], pos[b]["lon"], None]
        if lat:
            fig.add_trace(go.Scattergeo(lat=lat, lon=lon, mode="lines", line=dict(color=color, width=width),
                                        hoverinfo="skip", name=name, showlegend=show))

    if background_edges:
        edge_trace(background_edges, "rgba(137,135,129,0.25)", 0.6, "All borders", False)
    edge_trace(edges, "rgba(137,135,129,0.85)", 1.2, "Border", True)

    if color_by == "value":
        fig.add_trace(go.Scattergeo(
            lat=nodes["lat"], lon=nodes["lon"], text=nodes["name"],
            mode="markers+text" if len(nodes) <= 30 else "markers", name=value_label,
            textposition="top center", textfont=dict(size=10, color=MUTED),
            customdata=np.stack([nodes["value"], nodes["state"]], axis=1), showlegend=False,
            marker=dict(size=11, color=nodes["value"], colorscale=SEQ_SCALE, cmin=0,
                        line=dict(color="rgba(255,255,255,0.9)", width=1.5),
                        colorbar=dict(title=dict(text=value_label, side="right"), thickness=12, outlinewidth=0)),
            hovertemplate="%{text} (%{customdata[1]})<br>" + value_label + ": %{customdata[0]:,.1f}<extra></extra>"))
    else:
        for role, (color, label) in (roles or {}).items():
            sub = nodes[nodes["role"] == role]
            if not len(sub):
                continue
            extra = "" if "value" not in sub else "<br>" + value_label + ": %{customdata[1]:,.1f}"
            cd = np.stack([sub["state"], sub["value"] if "value" in sub else sub["state"]], axis=1)
            labelled = len(nodes) <= 30
            fig.add_trace(go.Scattergeo(
                lat=sub["lat"], lon=sub["lon"], text=sub["name"], customdata=cd,
                mode="markers+text" if labelled else "markers", name=label,
                textposition="top center", textfont=dict(size=10, color=MUTED),
                marker=dict(size=13 if role in ("focus", "start", "end") else 9, color=color,
                            line=dict(color="rgba(255,255,255,0.9)", width=1.5)),
                hovertemplate="%{text} (%{customdata[0]})" + extra + "<extra></extra>"))
    return _geo(fig, title, height=height)


def to_json(fig: go.Figure | None) -> str | None:
    return fig.to_json() if fig is not None else None
