"""Auswertung: Netzbau, Benders-Lauf mit den gewählten Optionen, Optimum (HiGHS), Urteil, Verteilung und Experiment-Tabellen."""

import statistics as st
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import bnd_benders as bd
import bnd_constants as C
import bnd_formulation as fm
import bnd_model as md


class NetParams(NamedTuple):
    net: str
    k: int
    p: int
    d: int
    s: int
    density: int
    spread: int
    load: int
    fix: int
    seed: int
    gw: int
    gh: int
    gdensity: int
    gcap: int
    gdem: int
    gseed: int


class Opts(NamedTuple):
    sub: str
    cut: str
    start: str
    iters: int
    control: str


DEFAULT_PARAMS = NetParams(C.DEFAULT_NET, C.DEFAULT_K, C.DEFAULT_P, C.DEFAULT_D, C.DEFAULT_S, C.DEFAULT_DENSITY, C.DEFAULT_SPREAD, C.DEFAULT_LOAD, C.DEFAULT_FIX, C.DEFAULT_SEED,
                           C.DEFAULT_GW, C.DEFAULT_GH, C.DEFAULT_GDENSITY, C.DEFAULT_GCAP, C.DEFAULT_GDEM, C.DEFAULT_GSEED)
DEFAULT_OPTS = Opts(C.DEFAULT_SUB, C.DEFAULT_CUT, C.DEFAULT_START, C.DEFAULT_ITER, C.DEFAULT_CONTROL)


def normalise(p):
    """Feste Netze ignorieren alle Regler der Zufallsnetze; das Streckennetz die des Distributionsnetzes und umgekehrt (gleicher Schlüssel für den Zwischenspeicher)."""
    if p.net in C.FIXED_NETS:
        return DEFAULT_PARAMS._replace(net=p.net, k={"bigm": 1, "rounding": 1, "bundle": 2}[p.net])
    if p.net == "grid":
        return p._replace(p=C.DEFAULT_P, d=C.DEFAULT_D, s=C.DEFAULT_S, density=C.DEFAULT_DENSITY, spread=C.DEFAULT_SPREAD, load=C.DEFAULT_LOAD, seed=C.DEFAULT_SEED)
    return p._replace(gw=C.DEFAULT_GW, gh=C.DEFAULT_GH, gdensity=C.DEFAULT_GDENSITY, gcap=C.DEFAULT_GCAP, gdem=C.DEFAULT_GDEM, gseed=C.DEFAULT_GSEED)


def build(p):
    if p.net == "bigm":
        return md.bigm_net()
    if p.net == "rounding":
        return md.rounding_net()
    if p.net == "bundle":
        return md.bundle_net()
    if p.net == "grid":
        return md.generate_design_grid(p.gw, p.gh, p.gdensity, p.gcap, p.gdem, p.k, p.gseed, p.fix)
    return md.generate_design(p.p, p.d, p.s, p.density, p.spread, p.load, p.seed, p.k, p.fix)


def with_seed(p, seed):
    return p._replace(gseed=seed) if p.net == "grid" else p._replace(seed=seed)


def run(design, sub="strong", cut="pareto", start="cutset", iters=C.DEFAULT_ITER, control="normal"):
    return bd.benders(design, level=C.LEVELS[sub], pareto=cut == "pareto", start=start, max_iter=iters, feas_cuts=control != "nofeas", wrong=control == "wrong")


@dataclass(frozen=True)
class Analysis:
    params: NetParams
    opts: Opts
    design: object
    feasible: bool
    opt: object = None              # Solution des exakten MIP (HiGHS)
    result: object = None           # BendersResult
    weak_lp: float = None           # schwache LP-Schranke der Formulierung (= LP-Relaxation des Masters)
    problem: object = None


def analyse(params, opts=DEFAULT_OPTS):
    params = normalise(params)
    design = build(params)
    try:
        weak, opt = fm.solve_lp(design, fm.WEAK), fm.solve_mip(design)
    except fm.Infeasible:
        return Analysis(params, opts, design, False)
    result = run(design, *opts[:3], iters=opts.iters, control=opts.control)
    return Analysis(params, opts, design, True, opt, result, weak.objective)


def verdict(a):
    """(Stufe, Code, Daten): ok / limit (Iterationsgrenze, LB < UB) / stuck (ohne Zulässigkeitsschnitte) / wrong (ungültige Schnitte) / nothing (nicht lieferbar)."""
    if not a.feasible:
        return "warning", "nothing", {}
    r, opt = a.result, a.opt.objective
    d = {"opt": opt, "lb": r.lb, "ub": r.ub, "iterations": r.iterations, "total": r.total_iterations, "n_opt": r.n_opt, "n_feas": r.n_feas, "converged": r.converged, "stuck": r.stuck, "crossed": r.crossed,
         "seconds": r.seconds, "master_share": r.master_seconds / r.seconds if r.seconds else 0.0, "highs_seconds": a.opt.seconds, "groups": a.design.G, "y_cuts": len(r.y_cuts),
         "weak": a.weak_lp, "gap": (r.ub - r.lb) / r.ub if r.ub not in (0, float("inf")) else float("nan"), "n_cuts": len(r.cuts), "pre": r.pre_iterations,
         "feas_share": r.n_feas / max(1, r.iterations)}
    if r.stuck:
        return "error", "stuck", d
    if r.crossed or r.lb > opt + 1e-6 * max(1.0, opt) or (r.converged and r.ub > opt + 1e-6 * max(1.0, opt)):
        return "error", "wrong", d
    if not r.converged:
        return "warning", "limit", d
    return "success", "ok", d


# --- Verteilung über feste Netze --------------------------------------------------------------------------------------------------------

def distribution(params, opts=DEFAULT_OPTS, seeds=C.QUALITY_SEEDS):
    """Iterationen, Sekunden und Anteil der Zulässigkeitsschnitte über feste Netze derselben Art (nur die Seeds ändern sich); nicht lieferbare Netze werden gezählt, nicht gemittelt."""
    rows, infeasible = [], 0
    for seed in seeds:
        a = analyse(with_seed(params, seed), opts)
        if not a.feasible:
            infeasible += 1
            continue
        r = a.result
        rows.append(dict(seed=seed, iterations=r.total_iterations, seconds=r.seconds, highs=a.opt.seconds, feas_share=r.n_feas / max(1, r.iterations), converged=r.converged, groups=a.design.G))
    col = lambda key: [r[key] for r in rows]
    return dict(n_seeds=len(seeds), n_feasible=len(rows), n_infeasible=infeasible, cols={k: col(k) for k in ("iterations", "seconds", "highs", "feas_share", "converged")},
                iterations_median=st.median(col("iterations")) if rows else float("nan"), iterations_mean=st.mean(col("iterations")) if rows else float("nan"),
                seconds_mean=st.mean(col("seconds")) if rows else float("nan"), highs_mean=st.mean(col("highs")) if rows else float("nan"),
                feas_share_mean=st.mean(col("feas_share")) if rows else float("nan"), converged_share=(sum(col("converged")) / len(rows)) if rows else float("nan"))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------

VARIANTS = (
    ("Reines Benders (schwach)", "weak", "normal", "none"),
    ("Starkes Teilproblem", "strong", "normal", "none"),
    ("+ Pareto-Schnitte", "strong", "pareto", "none"),
    ("+ LP-Wurzelschnitte", "strong", "pareto", "lp"),
    ("Cut-Set im Master", "strong", "normal", "cutset"),
    ("Pareto + Cut-Set im Master", "strong", "pareto", "cutset"),
)


def variant_table(params, size=(3, 3, 6), seeds=C.VARIANT_SEEDS, iters=200):
    """Aufwand je Variante (Iterationen einschließlich LP-Start, Anteil Zulässigkeitsschnitte, Sekunden) auf festen Netzen der Größe `size`; dazu HiGHS."""
    P, D, S = size
    nets = []
    for seed in seeds:
        d = md.generate_design(P, D, S, params.density, params.spread, params.load, seed, params.k, params.fix)
        try:
            fm.solve_lp(d, fm.WEAK)
        except fm.Infeasible:
            continue
        nets.append(d)
    highs = [fm.solve_mip(d).seconds for d in nets]
    rows = []
    for name, sub, cut, start in VARIANTS:
        rs = [run(d, sub, cut, start, iters=iters) for d in nets]
        rows.append(dict(name=name, sub=sub, cut=cut, start=start, converged=sum(r.converged for r in rs), n=len(rs), iterations=st.median(r.total_iterations for r in rs),
                         iterations_mean=st.mean(r.total_iterations for r in rs), feas_share=st.mean(r.n_feas / max(1, r.iterations) for r in rs), seconds=st.mean(r.seconds for r in rs),
                         master_share=st.mean(r.master_seconds / r.seconds for r in rs)))
    return dict(n=len(nets), size=size, rows=rows, highs=st.mean(highs) if highs else float("nan"), groups=st.mean(d.G for d in nets) if nets else float("nan"))


def size_table(params, sizes=((2, 2, 4), (3, 3, 8), (4, 4, 12)), seeds=C.SIZE_SEEDS, iters=100):
    """Benders (Pareto + Cut-Set, stark) gegen HiGHS über wachsende Netze: Gruppen, Iterationen, Anteil konvergiert, Sekunden."""
    out = []
    for P, D, S in sizes:
        rows = []
        for seed in seeds:
            d = md.generate_design(P, D, S, params.density, params.spread, params.load, seed, params.k, params.fix)
            try:
                opt = fm.solve_mip(d)
            except fm.Infeasible:
                continue
            r = run(d, "strong", "pareto", "cutset", iters=iters)
            rows.append((d.G, r.total_iterations, r.converged, r.seconds, opt.seconds))
        out.append(dict(size=(P, D, S), n=len(rows), groups=st.mean(x[0] for x in rows) if rows else float("nan"), iterations=st.mean(x[1] for x in rows) if rows else float("nan"),
                        converged=sum(x[2] for x in rows), seconds=st.mean(x[3] for x in rows) if rows else float("nan"), highs=st.mean(x[4] for x in rows) if rows else float("nan")))
    return out


def control_table(params):
    """Negativkontrollen auf dem gezeigten Netz: wie beschrieben, ohne Zulässigkeitsschnitte, ungültige Schnitte."""
    design = build(normalise(params))
    opt = fm.solve_mip(design).objective
    rows = []
    for key in ("normal", "nofeas", "wrong"):
        r = run(design, control=key, iters=100)
        rows.append(dict(control=key, iterations=r.iterations, converged=r.converged, stuck=r.stuck, crossed=r.crossed, lb=r.lb, ub=r.ub, opt=opt))
    return rows


def lp_master_table(params, seeds=C.VARIANT_SEEDS):
    """Die LP-Relaxation des Masters (mit allen Schnitten) gegen die schwache LP-Schranke der Formulierung (Anteil am Optimum): dieselbe Zahl."""
    rows = []
    for seed in seeds:
        design = build(normalise(with_seed(params, seed)))
        try:
            weak, opt = fm.solve_lp(design, fm.WEAK), fm.solve_mip(design)
        except fm.Infeasible:
            continue
        r = bd.benders(design, level=fm.WEAK, integer=False)
        rows.append(dict(seed=seed, weak=weak.objective / opt.objective, master=r.lb / opt.objective, iterations=r.iterations, same=abs(r.lb - weak.objective) <= 1e-6 * max(1.0, weak.objective)))
    return rows


# --- Anzeige-Helfer ------------------------------------------------------------------------------------------------------------------

def arc_name(design, e):
    net = design.net
    return f"{net.names[net.arcs[e][0]]} → {net.names[net.arcs[e][1]]}"


def group_name(design, g):
    return arc_name(design, design.arcs_of(g)[0])


def cut_words(design, cut, top=3):
    """Der Schnitt in Worten: bei Optimalitätsschnitten die Kanten mit den größten Preisen, bei Zulässigkeitsschnitten die Kanten, die er verlangt."""
    order = np.argsort(cut.slope)[:top] if cut.kind == "opt" else np.argsort(-np.abs(cut.slope))[:top]
    parts = [f"{group_name(design, int(g))} (Preis {-cut.slope[g]:.1f})" for g in order if abs(cut.slope[g]) > 1e-9]
    if cut.kind == "opt":
        return f"θ ≥ {cut.const:.1f} − Σ Preis · y: jede geöffnete Kante spart dem Fluss ihren Preis; die teuersten Engpässe: " + (", ".join(parts) if parts else "keine")
    return "Der Entwurf lässt " + f"{cut.value:.1f} Einheiten unbedient; verlangt wird mehr Kapazität an: " + (", ".join(parts) if parts else "–")
