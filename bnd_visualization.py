"""Plotly-Abbildungen: Netz mit Master-Entwurf, Fluss und Kapazitätspreisen, Schranken je Iteration, Varianten, Größe und Verteilung.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Kanten haben über unsichtbare Marker einen Hover-Text
(Plotly-Linien reagieren nur an ihren Stützpunkten)."""

from math import atan2, degrees, hypot

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bnd_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _layout(fig, net, height, skip=()):
    xs = [p[0] for v, p in enumerate(net.pos) if v not in skip]
    ys = [p[1] for v, p in enumerate(net.pos) if v not in skip]
    pad = 9
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    return _base(fig, height)


def _curve(p0, p1, bulge, steps=8):
    """Punkte von p0 nach p1; mit `bulge` > 0 als flacher Bogen nach rechts (so trennen sich Vorwärts- und Rückkante). Dazu der Pfeilwinkel bei 65 %."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = hypot(dx, dy) or 1.0
    cx, cy = (x0 + x1) / 2 + bulge * length * dy / length, (y0 + y1) / 2 - bulge * length * dx / length
    ts = [k / steps for k in range(steps + 1)]
    xs = [(1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1 for t in ts]
    ys = [(1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1 for t in ts]
    t = 0.65
    tx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
    ty = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)
    ax = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1
    ay = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1
    return xs, ys, (ax, ay, degrees(atan2(tx, ty))), (xs[steps // 2], ys[steps // 2])


def _segments(curves):
    x, y = [], []
    for xs, ys, _, _ in curves:
        x += xs + [None]
        y += ys + [None]
    return x, y


def _lines(fig, curves, color, width, name, dash=None, showlegend=True):
    if not curves:
        return
    x, y = _segments(curves)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", name=name, showlegend=showlegend))


def _arrows(fig, curves, color, size=9):
    if not curves:
        return
    fig.add_trace(go.Scatter(x=[c[2][0] for c in curves], y=[c[2][1] for c in curves], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="arrow", size=size, color=color, angle=[c[2][2] for c in curves])))


def _hover_points(fig, net, entries):
    """Unsichtbare Marker entlang jeder Kante, damit der Hover-Text überall auf der Kante erscheint. entries: [(Kurve, Text)]"""
    x, y, text = [], [], []
    for curve, label in entries:
        xs, ys = curve[0], curve[1]
        for k in range(1, len(xs) - 1):
            x.append(xs[k]); y.append(ys[k]); text.append(label)
    if x:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=9, opacity=0), hovertext=text, hoverinfo="text", showlegend=False))


def _labels(fig, points):
    """points: [(x, y, Text)] - als Annotationen mit heller Hinterlegung, damit sie Kanten, Pfeile und Knotenbeschriftungen nicht unlesbar machen."""
    for x, y, text in points:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", font=dict(size=11, color="#111"), bgcolor="rgba(255,255,255,0.88)", borderpad=1)


def _arc_name(net, i):
    u, v = net.arcs[i][0], net.arcs[i][1]
    return f"{net.names[u]} → {net.names[v]}"


def _wscale(net):
    return max(c for _, _, c, _, _ in net.arcs)


def _width(amount, top, lo=1.0, hi=6.0):
    return lo + (hi - lo) * amount / top if top else lo


def _node_text_positions(net, idx):
    if net.logistic:
        return ["top center" if (v == 0 or net.names[v].startswith("Werk")) else "bottom center" if (v == 1 or net.names[v].startswith("Filiale")) else "middle left" for v in idx]
    return ["top center" if v == net.s else "bottom center" if v == net.t else "middle left" for v in idx]


TOL = 1e-6


def _shift(curve, dx, dy):
    xs, ys, (ax, ay, ang), (mx, my) = curve
    return [x + dx for x in xs], [y + dy for y in ys], (ax + dx, ay + dy, ang), (mx + dx, my + dy)


def _fmt(x):
    return f"{x:.1f}".replace(".", ",") if abs(x - round(x)) > TOL else f"{round(x)}"


def _is_terminal(net, e):
    return net.arcs[e][0] == net.s or net.arcs[e][1] == net.t


def _num(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_master_map(design, frame, opt_open=None, height=480):
    """Netz in einer Iteration: Blau = der Master öffnet die Kante, grau = geschlossen; orange die Flussmenge des Teilproblems (Dicke ~ Fluss / Kapazität); violette Unterlage = Kante mit Kapazitätspreis > 0 im
    Schnitt dieser Iteration (Dicke ~ Preis); rote Unterlage = im ganzzahligen Optimum offen. Im Streckennetz sind die Kanten von S und zu T ausgeblendet."""
    net = design.net
    grid = design.mcf.layout == "grid"
    fig = go.Figure()
    bulge = 0.0 if net.logistic else 0.12
    y, slope, x = frame.y, frame.cut.slope, frame.x
    K, m = design.K, design.m
    flows = np.asarray(x, dtype=float).reshape(K, m).sum(axis=0) if len(x) == K * m else np.zeros(m)
    top_price = float(np.abs(slope).max()) or 1.0
    curves = {"open": [], "closed": [], "opt": [], "fixed": []}
    price, flow, hover, labels = [], [], [], []
    seen, drawn = set(), {}
    for e, (u, v, cap, cost, kind) in enumerate(net.arcs):
        if grid and _is_terminal(net, e):
            continue
        g = design.group[e]
        if grid and g in seen:
            continue
        par = drawn.get((u, v), 0)
        drawn[(u, v)] = par + 1
        base = _curve(net.pos[u], net.pos[v], 0.0 if grid else bulge + 0.2 * par)
        load = float(flows[e])
        if g < 0:
            curves["fixed"].append(base)
            hover.append((base, f"{_arc_name(net, e)} (Kapazität {cap}, immer offen): Fluss {_fmt(load)}"))
            continue
        seen.add(g)
        is_open = y[g] > 0.5
        curves["open" if is_open else "closed"].append(base)
        if opt_open is not None and g in opt_open:
            curves["opt"].append(base)
        p = -float(slope[g])
        if p > 1e-9:
            price.append((base, 3 + 9 * p / top_price))
        if is_open and load > TOL:
            flow.append((base, 1.5 + 4 * min(1.0, load / cap)))
        hover.append((base, f"{_arc_name(net, e)}: {'offen' if is_open else 'geschlossen'}, Fixkosten {design.fixed[g]}, Kapazität {cap}, Fluss {_fmt(load)}, Preis im Schnitt {_num(p, 1)}"))
        if is_open and load > TOL and design.m <= 60:
            labels.append((base[0][3] + 1.5, base[1][3], f"{_fmt(load)}/{cap}"))
    _lines(fig, curves["opt"], "rgba(214,39,40,0.35)", 12, "Optimum (ganzzahlig)")
    for c, w in price:
        _lines(fig, [c], "rgba(148,103,189,0.5)", w, "Kapazitätspreis im Schnitt", showlegend=False)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="rgba(148,103,189,0.5)", width=9), name="Preis im Schnitt"))
    _lines(fig, curves["fixed"], C.COLORS["closed"], 1.0, "Werks-/Nachfragekanten", showlegend=False)
    _lines(fig, curves["closed"], C.COLORS["closed"], 1.0, "geschlossen")
    _lines(fig, curves["open"], C.COLORS["open"], 3.5, "offen")
    for c, w in flow:
        _lines(fig, [c], C.COLORS["flow"], w, "Fluss", showlegend=False)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C.COLORS["flow"], width=3), name="Fluss"))
    if net.m <= 80:
        _arrows(fig, curves["open"], "rgba(60,60,60,0.55)", 7)
    _hover_points(fig, net, hover)
    _labels(fig, labels)
    idx = [v for v in range(net.n) if not (grid and v in (net.s, net.t))]
    fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in idx], y=[net.pos[v][1] for v in idx], mode="markers+text", showlegend=False, text=[net.labels[v] for v in idx],
                             textposition=_node_text_positions(net, idx), hovertext=[net.names[v] for v in idx], hoverinfo="text",
                             marker=dict(symbol=["square" if v in (net.s, net.t) else "circle" for v in idx], size=[13 if v in (net.s, net.t) else 10 for v in idx], color=C.COLORS["node"], line=dict(width=1.5, color="#333"))))
    return _layout(fig, net, height, skip=(net.s, net.t) if grid else ())


def build_bounds(frames, opt, i, height=340):
    """Untere Schranke (Master) und obere Schranke (bester Entwurf mit Fluss) je Iteration gegen das Optimum; Marken nach Schnittart (Kreuz = Zulässigkeitsschnitt, Punkt = Optimalitätsschnitt), Ring = gezeigte Iteration."""
    xs = [f.iteration for f in frames]
    fig = go.Figure()
    fig.add_hline(y=opt, line=dict(color=C.COLORS["opt"], width=2))
    fig.add_trace(go.Scatter(x=xs, y=[f.lb for f in frames], mode="lines", line=dict(color="#7f7f7f", width=2), name="untere Schranke (Master)", hoverinfo="skip"))
    for kind, sym, name, color in (("opt", "circle", "Optimalitätsschnitt", C.COLORS["optcut"]), ("feas", "x", "Zulässigkeitsschnitt", C.COLORS["feas"])):
        sel = [f for f in frames if f.kind == kind]
        fig.add_trace(go.Scatter(x=[f.iteration for f in sel], y=[f.lb for f in sel], mode="markers", marker=dict(symbol=sym, size=7, color=color), name=name,
                                 hovertext=[f"Iteration {f.iteration}: LB {_num(f.lb, 1)}" for f in sel], hoverinfo="text"))
    fin = [f for f in frames if f.ub < float("inf")]
    if fin:
        fig.add_trace(go.Scatter(x=[f.iteration for f in fin], y=[f.ub for f in fin], mode="lines", line=dict(color="#2ca02c", width=2, dash="dot"), name="obere Schranke", hoverinfo="skip"))
    f = frames[i]
    fig.add_trace(go.Scatter(x=[f.iteration], y=[f.lb], mode="markers", marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(width=3, color=C.COLORS["price"])), name="gezeigte Iteration", hoverinfo="skip"))
    lo = min(f.lb for f in frames)
    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title="Kosten (rot: Optimum)", range=[max(0, lo - 0.05 * opt), opt * 1.12])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_variants(table, height=340):
    """Iterationen je Variante (Median, logarithmisch); Beschriftung: Anteil der Zulässigkeitsschnitte."""
    rows = table["rows"]
    fig = go.Figure(go.Bar(x=[r["name"] for r in rows], y=[r["iterations"] for r in rows], marker_color="#1f77b4", text=[f"{r['iterations']:.0f}" for r in rows], textposition="outside",
                           hovertext=[f"{r['name']}: {r['iterations']:.0f} Iterationen, {100 * r['feas_share']:.0f} % Zulässigkeitsschnitte, {r['converged']} von {r['n']} konvergiert" for r in rows], hoverinfo="text"))
    fig.update_yaxes(type="log", title="Iterationen (Median, log.)")
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return lock_axes(fig)


def build_sizes(rows, height=320):
    """Sekunden je Netzgröße: Benders (Pareto + Cut-Set) gegen HiGHS, logarithmisch."""
    xs = [f"{P}/{D}/{S}" for P, D, S in (r["size"] for r in rows)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["seconds"] for r in rows], name="Benders", marker_color="#9467bd", text=[f"{r['seconds']:.2f}" for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(x=xs, y=[r["highs"] for r in rows], name="HiGHS (direktes MIP)", marker_color="#2ca02c", text=[f"{r['highs']:.3f}" for r in rows], textposition="outside"))
    fig.update_xaxes(title="Werke / Verteilzentren / Filialen", type="category")
    fig.update_yaxes(type="log", title="Sekunden je Netz (log.)")
    fig.update_layout(height=height, barmode="group", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_dist(dist, mine, height=340):
    """Verteilung der Iterationen über die festen Netze (Box mit Punkten); Raute = die gezeigte Ziehung."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=dist["cols"]["iterations"], name="Iterationen", marker_color="#1f77b4", boxpoints="all", jitter=0.4, pointpos=0, marker_size=4, showlegend=False))
    if mine is not None:
        fig.add_trace(go.Scatter(x=["Iterationen"], y=[mine], mode="markers", name="Ihre Ziehung", marker=dict(symbol="diamond", size=13, color=C.COLORS["opt"], line=dict(width=1.5, color="#111"))))
    fig.update_yaxes(title="Iterationen bis LB = UB")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)
