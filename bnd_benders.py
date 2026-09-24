"""Benders-Zerlegung für das Fixkosten-Netzdesign: Entwurf y im ganzzahligen Master, Mehrgüterfluss als LP im Teilproblem, Dualpreise als Schnitte.

Für einen festen Entwurf y ist der Fluss ein LP:  v(y) = min c·x  u.d.N.  A_eq x = 0,  A_x x <= b - A_y y,  lo <= x <= hi
(Zeilen der Kapazität u_e·y_g, bei starker Kopplung auch je Gut). v ist in y konvex und stückweise linear. Jede dual zulässige Lösung (λ, μ, ρ, ω) des Teilproblems liefert nach schwacher Dualität eine untere Schranke
    v(y) >= D(y) = b_eq·λ - (b - A_y y)·μ + lo·ρ - hi·ω  =  const + slope·y      mit  A_eq^T λ - A_x^T μ + ρ - ω = c,  μ, ρ, ω >= 0.
- **Optimalitätsschnitt:** θ >= const + slope·y (für die Duale, die im Punkt ȳ optimal sind, gilt D(ȳ) = v(ȳ)). slope_g <= 0 ist, was eine geöffnete Kante an Flusskosten spart.
- **Zulässigkeitsschnitt:** ist das Teilproblem bei ȳ unlösbar, minimiert ein Phase-1-LP die Fehlmenge; sein Dual D1 gibt const1 + slope1·y <= -D (D = Gesamtnachfrage): nur Entwürfe mit Fehlmenge 0 erfüllen ihn.
- **Pareto-optimale Schnitte (Magnanti-Wong):** unter allen im Punkt ȳ optimalen Dualen das mit dem größten Wert an einem Kernpunkt y0 - ein stärkerer Schnitt, dieselbe Gültigkeit.
Der Master ist min f·y + θ mit den gesammelten Schnitten (HiGHS `milp`); untere Schranke LB = Master-Wert, obere UB = beste f·y + v(y).
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

import bnd_cuts as cu
import bnd_formulation as fm

TOL = 1e-6


@dataclass(frozen=True)
class Cut:
    kind: str              # "opt" (θ >= const + slope·y) oder "feas" (slope·y <= -const)
    const: float
    slope: np.ndarray      # (G,)
    y: np.ndarray          # der Entwurf, an dem der Schnitt entstand
    value: float           # v(ȳ) bei "opt", Fehlmenge bei "feas"
    pareto: bool = False

    def at(self, y):
        """Wert der rechten Seite an y: bei "opt" die untere Schranke für θ, bei "feas" die linke Seite (muss <= 0 sein)."""
        return self.const + float(self.slope @ np.asarray(y, dtype=float))

    def holds(self, y, theta=None, tol=1e-6):
        if self.kind == "feas":
            return self.at(y) <= tol
        return theta is None or theta >= self.at(y) - tol


@dataclass(frozen=True)
class Frame:
    iteration: int
    y: np.ndarray          # Entwurf des Masters
    theta: float
    lb: float
    ub: float
    kind: str              # "opt" (Optimalitätsschnitt) oder "feas" (Zulässigkeitsschnitt)
    cut: object            # Cut oder None
    x: np.ndarray          # Fluss des Teilproblems (bei "feas": der beste Teilfluss der Phase 1)
    shortfall: float       # Fehlmenge bei unzulässigem Entwurf (sonst 0)
    flow_cost: float
    fixed_cost: float


@dataclass(frozen=True)
class BendersResult:
    frames: tuple
    lb: float
    ub: float
    y_best: np.ndarray
    converged: bool
    stuck: bool                 # True: der Master lieferte einen Entwurf, für den es keinen Schnitt gab (nur ohne Zulässigkeitsschnitte)
    crossed: bool               # True: LB > UB (nur mit ungültigen Schnitten)
    cuts: tuple
    y_cuts: tuple               # zusätzliche Ungleichungen über y (Cut-Set-Schnitte), falls verwendet
    seconds: float
    master_seconds: float
    sub_seconds: float
    pre_iterations: int = 0     # Iterationen des LP-Starts ("lp"), sie zählen zum Aufwand

    @property
    def iterations(self):
        return len(self.frames)

    @property
    def total_iterations(self):
        return len(self.frames) + self.pre_iterations

    @property
    def n_opt(self):
        return sum(1 for f in self.frames if f.kind == "opt")

    @property
    def n_feas(self):
        return sum(1 for f in self.frames if f.kind == "feas")


class Problem:
    """Das Teilproblem eines Netzes: Matrizen einmal aufgebaut, für jeden Entwurf nur die rechte Seite ändern."""

    def __init__(self, design, level=fm.STRONG):
        self.design, self.level = design, level
        c, a_eq, b_eq, a_ub, b_ub, lo, hi = fm._build(design, level)
        nx = design.K * design.m
        self.nx = nx
        self.c = c[:nx].copy()
        self.fixed = np.asarray(design.fixed, dtype=float)
        self.a_eq = a_eq[:, :nx].toarray()
        self.b_eq = b_eq
        self.a_x = a_ub[:, :nx].toarray()
        self.a_y = a_ub[:, nx:].toarray()
        self.b = np.asarray(b_ub, dtype=float)
        self.lo = lo[:nx].copy()
        self.hi = hi[:nx].copy()
        self.demand = float(sum(design.mcf.demand(k) for k in range(design.K)))
        dem = [k * design.m + e for k in range(design.K) for e in design.demand_arcs()]
        self.dem_cols = np.array(dem, dtype=int)
        self.c1 = np.zeros(nx)
        self.c1[self.dem_cols] = -1.0                     # Phase 1: Lieferung maximieren
        self.lo1 = self.lo.copy()
        self.lo1[self.dem_cols] = 0.0                     # Phase 1: Nachfragekanten dürfen unterliefern

    def rhs(self, y):
        return self.b - self.a_y @ np.asarray(y, dtype=float)

    def _linprog(self, c, y, lo):
        bounds = list(zip(lo, np.where(np.isinf(self.hi), None, self.hi)))
        return linprog(c, A_ub=self.a_x, b_ub=self.rhs(y), A_eq=self.a_eq, b_eq=self.b_eq, bounds=bounds, method="highs")

    def _dual_cut(self, res, c, lo, kind, y, value, pareto=False):
        """Dualwerte des Primal-LP als Schnitt: μ = -Marginale der Ungleichungen, λ = Marginale der Gleichungen, ρ, ω aus den Grenzen."""
        mu = -np.asarray(res.ineqlin.marginals)
        lam = np.asarray(res.eqlin.marginals)
        rho = np.asarray(res.lower.marginals)
        omega = -np.asarray(res.upper.marginals)
        return self._cut_from_dual(kind, lam, mu, rho, omega, lo, y, value, pareto)

    def _cut_from_dual(self, kind, lam, mu, rho, omega, lo, y, value, pareto):
        hi_fin = np.where(np.isinf(self.hi), 0.0, self.hi)
        const = float(self.b_eq @ lam - self.b @ mu + lo @ rho - hi_fin @ omega)
        slope = self.a_y.T @ mu
        if kind == "feas":
            const = const + self.demand                      # D1(y) <= -D  ->  const1 + D + slope1·y <= 0
        # Rauschen entfernen, ohne die Gültigkeit zu verlieren (y in [0, 1]): ein winziger negativer Koeffizient wird zur Konstante, ein positiver entfällt (beides macht den Schnitt nur schwächer)
        tiny = np.abs(slope) < 1e-7
        const += float(np.minimum(slope[tiny], 0.0).sum())
        slope = np.where(tiny, 0.0, slope)
        if kind == "feas":                                    # Zulässigkeitsschnitte auf Koeffizienten der Größe 1 skalieren (HiGHS verträgt große Spannen schlecht)
            scale = max(1.0, float(np.abs(slope).max()))
            const, slope = const / scale, slope / scale
        return Cut(kind, const, slope, np.asarray(y, dtype=float).copy(), value, pareto)

    def subproblem(self, y):
        """Fluss bei festem Entwurf. Rückgabe (Cut, x, Fehlmenge): bei Fehlmenge > 0 ein Zulässigkeitsschnitt aus Phase 1, sonst ein Optimalitätsschnitt."""
        y = np.asarray(y, dtype=float)
        p1 = self._linprog(self.c1, y, self.lo1)
        if p1.status != 0:
            raise RuntimeError(p1.message)
        shortfall = self.demand + p1.fun
        if shortfall > TOL:
            return self._dual_cut(p1, self.c1, self.lo1, "feas", y, shortfall), p1.x, shortfall
        res = self._linprog(self.c, y, self.lo)
        if res.status != 0:                                   # Fehlmenge unter der Toleranz, aber das LP ist numerisch unlösbar: als unzulässig behandeln
            return self._dual_cut(p1, self.c1, self.lo1, "feas", y, max(shortfall, TOL)), p1.x, max(shortfall, TOL)
        return self._dual_cut(res, self.c, self.lo, "opt", y, float(res.fun)), res.x, 0.0

    def value(self, y):
        """v(y) oder None, wenn der Entwurf die Nachfrage nicht decken kann."""
        res = self._linprog(self.c, y, self.lo)
        return float(res.fun) if res.status == 0 else None

    def pareto_cut(self, y, value, core):
        """Magnanti-Wong: unter den im Punkt y optimalen Dualen das mit dem größten Wert im Kernpunkt `core`. None, wenn das Hilfs-LP scheitert (dann gilt der normale Schnitt)."""
        nx, ne, nu = self.nx, self.a_eq.shape[0], self.a_x.shape[0]
        y = np.asarray(y, dtype=float)
        # Variablen: λ (frei) | μ >= 0 | ρ >= 0 | ω >= 0 (nur bei endlicher Obergrenze)
        n = ne + nu + 2 * nx
        a_dual = np.hstack([self.a_eq.T, -self.a_x.T, np.eye(nx), -np.eye(nx)])
        hi_fin = np.where(np.isinf(self.hi), 0.0, self.hi)

        def dual_obj(rhs_vec):
            v = np.zeros(n)
            v[:ne] = self.b_eq
            v[ne:ne + nu] = -rhs_vec
            v[ne + nu:ne + nu + nx] = self.lo
            v[ne + nu + nx:] = -hi_fin
            return v

        eps = 1e-7 * max(1.0, abs(value))
        bounds = [(None, None)] * ne + [(0, None)] * nu + [(0, None)] * nx + [(0, None) if np.isfinite(h) else (0, 0) for h in self.hi]
        res = linprog(-dual_obj(self.rhs(core)), A_ub=-dual_obj(self.rhs(y))[None, :], b_ub=[-(value - eps)], A_eq=a_dual, b_eq=self.c, bounds=bounds, method="highs")
        if res.status != 0:
            return None
        z = res.x
        cut = self._cut_from_dual("opt", z[:ne], z[ne:ne + nu], z[ne + nu:ne + nu + nx], z[ne + nu + nx:], self.lo, y, value, True)
        return cut if cut.at(y) >= value - 1e-5 * max(1.0, abs(value)) else None


def _master(problem, opt_cuts, feas_cuts, y_cuts, integer, upper=None):
    """Master: min f·y + θ mit den Schnitten. Rückgabe (y, θ, Zielwert, Sekunden) oder None, wenn der Master unlösbar ist."""
    G = problem.design.G
    rows, rhs = [], []
    for c in opt_cuts:
        rows.append(np.concatenate([c.slope, [-1.0]])); rhs.append(-c.const)
    for c in feas_cuts:
        rows.append(np.concatenate([c.slope, [0.0]])); rhs.append(-c.const)
    for c in y_cuts:
        row = np.zeros(G + 1)
        for g, a in c.coef.items():
            row[g] = -a
        rows.append(row); rhs.append(-c.rhs)
    cost = np.concatenate([problem.fixed, [1.0]])
    t0 = perf_counter()
    lo = np.concatenate([np.zeros(G), [0.0]])
    hi = np.concatenate([np.ones(G), [np.inf]])
    if rows:
        a = np.array(rows)
        b = np.array(rhs)
        if integer:
            integrality = np.concatenate([np.ones(G), [0]])
            res = milp(cost, constraints=[LinearConstraint(a, -np.inf, b)], integrality=integrality, bounds=Bounds(lo, hi))
        else:
            res = linprog(cost, A_ub=a, b_ub=b, bounds=list(zip(lo, [1.0] * G + [None])), method="highs")
    else:
        res = milp(cost, integrality=np.concatenate([np.ones(G), [0]]), bounds=Bounds(lo, hi)) if integer else linprog(cost, bounds=list(zip(lo, [1.0] * G + [None])), method="highs")
    seconds = perf_counter() - t0
    if res.x is None:
        return None
    y = np.round(res.x[:G]) if integer else np.asarray(res.x[:G], dtype=float)
    return y, float(res.x[G]), float(cost @ np.concatenate([y, [res.x[G]]])), seconds


def core_point(design, problem):
    """Kernpunkt für die Pareto-Schnitte: die Mitte zwischen der LP-Lösung der Formulierung und dem Entwurf mit allen Kanten offen (echt im Innern, der Fluss ist dort zulässig)."""
    lp = fm.solve_lp(design, fm.WEAK)
    return 0.5 * (np.asarray(lp.y) + 1.0)


def benders(design, level=fm.STRONG, pareto=False, start="none", max_iter=200, tol=TOL, feas_cuts=True, wrong=False, integer=True, problem=None):
    """Die Schleife. `start`: "none", "lp" (erst am LP-Master bis zur Konvergenz, deren Schnitte kommen in den ganzzahligen Master) oder "cutset" (Cut-Set-Ungleichungen aus fixkosten-netzdesign in den Master).
    `feas_cuts=False` ist die Negativkontrolle (ohne Zulässigkeitsschnitte bleibt der Master bei unzulässigen Entwürfen hängen), `wrong=True` verschiebt jeden Optimalitätsschnitt nach oben (ungültig).
    `integer=False` löst den Master als LP (nur für den LP-Start)."""
    problem = problem or Problem(design, level)
    core = core_point(design, problem) if pareto else None
    opt_cuts, feas_list, y_cuts, frames = [], [], [], []
    if start == "cutset":
        y_cuts = list(cu.cut_and_solve(design, level=level).cuts)
    t_start = perf_counter()
    pre_it, pre_master, pre_sub = 0, 0.0, 0.0
    if start == "lp" and integer:
        pre = benders(design, level, pareto, "none", max_iter, tol, feas_cuts, wrong, integer=False, problem=problem)
        opt_cuts = [c for c in pre.cuts if c.kind == "opt"]
        feas_list = [c for c in pre.cuts if c.kind == "feas"]
        pre_it, pre_master, pre_sub = pre.iterations, pre.master_seconds, pre.sub_seconds
    lb, ub, y_best = -np.inf, np.inf, None
    t_master, t_sub = pre_master, pre_sub
    converged = stuck = crossed = False
    seen = set()
    for it in range(1, max_iter + 1):
        sol = _master(problem, opt_cuts, feas_list, y_cuts, integer)
        if sol is None:
            break
        y, theta, obj, sec = sol
        t_master += sec
        lb = max(lb, obj)
        key = tuple(np.round(y, 6))
        ts = perf_counter()
        cut, x, shortfall = problem.subproblem(y)
        if cut.kind == "opt" and pareto:
            better = problem.pareto_cut(y, cut.value, core)
            cut = better or cut
        t_sub += perf_counter() - ts
        fixed_cost = float(problem.fixed @ y)
        if cut.kind == "opt":
            if integer:
                ub_new = fixed_cost + cut.value
                if ub_new < ub - 1e-9:
                    ub, y_best = ub_new, y.copy()
            if wrong:
                cut = Cut(cut.kind, cut.const + 0.05 * abs(cut.value) + 1.0, cut.slope, cut.y, cut.value, cut.pareto)
        frames.append(Frame(it, y.copy(), theta, lb, ub, cut.kind, cut, x, shortfall, cut.value if cut.kind == "opt" else 0.0, fixed_cost))
        if integer and lb > ub + 1e-6 * max(1.0, abs(ub)):
            crossed = True
            break
        if cut.kind == "opt":
            done = (ub - lb <= tol * max(1.0, abs(ub))) if integer else (theta >= cut.value - 1e-7 * max(1.0, abs(cut.value)))       # LP-Master: kein verletzter Schnitt mehr
            if done:
                converged = True
                break
        if cut.kind == "feas" and not feas_cuts:
            if key in seen:
                stuck = True
                break
            seen.add(key)
            continue
        (opt_cuts if cut.kind == "opt" else feas_list).append(cut)
    return BendersResult(tuple(frames), float(lb), float(ub), y_best if y_best is not None else np.zeros(design.G), converged, stuck, crossed, tuple(opt_cuts + feas_list), tuple(y_cuts),
                         perf_counter() - t_start, t_master, t_sub, pre_it)
