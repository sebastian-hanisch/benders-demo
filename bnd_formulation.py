"""Formulierungen des Entwurfsproblems: drei LP-Stufen, das exakte MIP (HiGHS) und ein Brute-Force-Referenzlöser.

Variablen: x[k][e] (Fluss je Gut und Kante, K·m Stück) und y[g] (Entwurfsentscheidung je Gruppe, G Stück), in dieser Reihenfolge.
Die volle Nachfrage muss geliefert werden (Nachfragekanten haben Unter- gleich Obergrenze), das Ziel ist Stückkosten + Fixkosten.

Stufen der Kopplung zwischen Fluss und Entscheidung:
- **0 schwach:** sum_k x[k][e] <= u_e · y_g (Big-M mit M = Kapazität). Im LP genügt y = Fluss / Kapazität: die Fixkosten werden nur anteilig gezahlt.
- **1 stark:**  zusätzlich x[k][e] <= min(u_e, d_k) · y_g je Gut (d_k = Nachfrage des Guts, mehr kann nie fließen).
- **2 stark + Schnitte:** die Schnittungleichungen aus `bnd_cuts.py` kommen hinzu (Zeilen über y allein).
"""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

import bnd_model as md

WEAK, STRONG = 0, 1
LEVEL_LABELS = {0: "schwach", 1: "stark", 2: "stark + Schnitte"}
TOL = 1e-6


@dataclass(frozen=True)
class Solution:
    objective: float          # Stückkosten + Fixkosten (bei LP: Schranke)
    x: np.ndarray             # (K, m)
    y: np.ndarray             # (G,)
    flow_cost: float
    fixed_cost: float
    nodes: int = 0            # Knoten des Branch-and-Bound (nur MIP)
    seconds: float = 0.0
    n_rows: int = 0
    n_cols: int = 0

    @property
    def n_fractional(self):
        return int(np.sum((self.y > TOL) & (self.y < 1 - TOL)))

    @property
    def n_open(self):
        return int(np.sum(self.y > 1 - TOL))


class Infeasible(Exception):
    """Selbst mit allen Kanten offen lässt sich die Nachfrage nicht decken."""


def demand_of(design, k):
    return design.mcf.demand(k)


def _build(design, level, cuts=()):
    """Zielvektor, Gleichungen, Ungleichungen und Grenzen des Problems (Stufe `level`, dazu die Schnitte `cuts`)."""
    mcf, net, K, m, G = design.mcf, design.net, design.K, design.m, design.G
    nx = K * m
    c = np.zeros(nx + G)
    for k in range(K):
        for e in range(m):
            if not mcf.reward[e]:
                c[k * m + e] = mcf.cost(k, e)
    c[nx:] = design.fixed
    inner = [v for v in range(net.n) if v not in (net.s, net.t)]
    row_of = {v: i for i, v in enumerate(inner)}
    rows, cols, vals = [], [], []
    for k in range(K):
        for e, (u, v, _, _, _) in enumerate(net.arcs):
            if v in row_of:
                rows.append(k * len(inner) + row_of[v]); cols.append(k * m + e); vals.append(1.0)
            if u in row_of:
                rows.append(k * len(inner) + row_of[u]); cols.append(k * m + e); vals.append(-1.0)
    n_eq = K * len(inner)
    a_eq = coo_matrix((vals, (rows, cols)), shape=(n_eq, nx + G)).tocsr()

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for e in range(m):
        if not mcf.joint[e]:
            continue
        cap = net.arcs[e][2]
        g = design.group[e]
        for k in range(K):
            rows.append(r); cols.append(k * m + e); vals.append(1.0)
        if g >= 0:
            rows.append(r); cols.append(nx + g); vals.append(-float(cap))      # sum_k x <= u · y
            rhs.append(0.0)
        else:
            rhs.append(float(cap))                                             # feste Kante: sum_k x <= u
        r += 1
    if level >= STRONG:
        for e in range(m):
            g = design.group[e]
            if g < 0:
                continue
            for k in range(K):
                bound = min(net.arcs[e][2], demand_of(design, k))
                if mcf.ub[k][e] < md.BIG:
                    bound = min(bound, mcf.ub[k][e])
                rows.append(r); cols.append(k * m + e); vals.append(1.0)
                rows.append(r); cols.append(nx + g); vals.append(-float(bound))
                rhs.append(0.0)
                r += 1
    for cut in cuts:
        for g, a in cut.coef.items():
            rows.append(r); cols.append(nx + g); vals.append(-float(a))
        rhs.append(-float(cut.rhs))
        r += 1
    a_ub = coo_matrix((vals, (rows, cols)), shape=(r, nx + G)).tocsr()
    lo = np.zeros(nx + G)
    hi = np.full(nx + G, np.inf)
    for k in range(K):
        for e in range(m):
            if mcf.ub[k][e] < md.BIG:
                hi[k * m + e] = mcf.ub[k][e]
            if mcf.reward[e]:
                lo[k * m + e] = hi[k * m + e]                                   # volle Nachfrage
    hi[nx:] = 1.0
    return c, a_eq, np.zeros(n_eq), a_ub, np.array(rhs), lo, hi


def _solution(design, res_x, objective, seconds=0.0, nodes=0, shape=(0, 0)):
    K, m, G = design.K, design.m, design.G
    x = np.asarray(res_x[:K * m], dtype=float).reshape(K, m)
    y = np.asarray(res_x[K * m:], dtype=float)
    x[np.abs(x) < 1e-9] = 0.0
    y[np.abs(y) < 1e-9] = 0.0
    y[np.abs(y - 1) < 1e-9] = 1.0
    flow_cost = float(sum(design.mcf.cost(k, e) * x[k, e] for k in range(K) for e in range(m) if not design.mcf.reward[e]))
    return Solution(float(objective), x, y, flow_cost, float(np.dot(design.fixed, y)) if G else 0.0, nodes, seconds, *shape)


def solve_lp(design, level=STRONG, cuts=()):
    """LP-Relaxation der Stufe `level` (mit Schnitten `cuts`); Infeasible, wenn es keinen Fluss gibt."""
    c, a_eq, b_eq, a_ub, b_ub, lo, hi = _build(design, level, cuts)
    t0 = perf_counter()
    res = linprog(c, A_ub=a_ub if a_ub.shape[0] else None, b_ub=b_ub if a_ub.shape[0] else None, A_eq=a_eq, b_eq=b_eq, bounds=list(zip(lo, np.where(np.isinf(hi), None, hi))), method="highs")
    if res.status == 2:
        raise Infeasible("keine Lieferung möglich")
    if res.status != 0:
        raise RuntimeError(res.message)
    return _solution(design, res.x, res.fun, perf_counter() - t0, 0, (a_ub.shape[0] + a_eq.shape[0], len(c)))


def solve_mip(design, level=STRONG, cuts=(), time_limit=120.0):
    """Exaktes Optimum: y ganzzahlig (HiGHS Branch-and-Cut); `level` und `cuts` bestimmen nur die Formulierung, das Optimum ist dasselbe."""
    c, a_eq, b_eq, a_ub, b_ub, lo, hi = _build(design, level, cuts)
    cons = [LinearConstraint(a_eq, b_eq, b_eq)]
    if a_ub.shape[0]:
        cons.append(LinearConstraint(a_ub, -np.inf, b_ub))
    integrality = np.zeros(len(c))
    integrality[design.K * design.m:] = 1
    t0 = perf_counter()
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lo, hi), options={"time_limit": time_limit})
    seconds = perf_counter() - t0
    if res.status == 2:
        raise Infeasible("keine Lieferung möglich")
    if res.x is None:
        raise RuntimeError(res.message)
    return _solution(design, res.x, res.fun, seconds, int(getattr(res, "mip_node_count", 0) or 0), (a_ub.shape[0] + a_eq.shape[0], len(c)))


def solve_fixed_design(design, opened):
    """Bester Fluss bei festem Entwurf: `opened` = Menge geöffneter Gruppen. None, wenn die Nachfrage nicht zu decken ist."""
    c, a_eq, b_eq, a_ub, b_ub, lo, hi = _build(design, WEAK)
    nx = design.K * design.m
    lo, hi = lo.copy(), hi.copy()
    for g in range(design.G):
        v = 1.0 if g in opened else 0.0
        lo[nx + g] = hi[nx + g] = v
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=list(zip(lo, hi)), method="highs")
    if res.status != 0:
        return None
    return _solution(design, res.x, res.fun)


def brute_force(design):
    """Optimum durch Aufzählen aller 2^G Entwürfe (nur für Kleinstnetze); Vergleich für das MIP und Prüfstein für jeden Schnitt."""
    best = None
    feasible = []
    for bits in product((0, 1), repeat=design.G):
        opened = {g for g, b in enumerate(bits) if b}
        sol = solve_fixed_design(design, opened)
        if sol is None:
            continue
        feasible.append(bits)
        if best is None or sol.objective < best[0].objective - 1e-9:
            best = (sol, bits)
    if best is None:
        raise Infeasible("keine Lieferung möglich")
    return best[0], best[1], feasible


def round_up(design, lp):
    """Obere Schranke durch Aufrunden: jede Gruppe mit y > 0 im LP öffnen, danach den besten Fluss suchen und ungenutzte Gruppen wieder schließen."""
    opened = {g for g in range(design.G) if lp.y[g] > TOL}
    sol = solve_fixed_design(design, opened)
    if sol is None:
        opened = set(range(design.G))
        sol = solve_fixed_design(design, opened)
    used = {design.group[e] for k in range(design.K) for e in range(design.m) if design.group[e] >= 0 and sol.x[k, e] > TOL}
    opened &= used
    sol = solve_fixed_design(design, opened)
    return sol.flow_cost + float(sum(design.fixed[g] for g in opened)), opened


def gap_closed(bound, weak, opt):
    """Anteil der Lücke zwischen der schwachen Schranke und dem Optimum, den `bound` schließt (0 = nichts, 1 = alles)."""
    return 1.0 if opt - weak < 1e-9 else (bound - weak) / (opt - weak)
