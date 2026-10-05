"""Unabhängiges Orakel zu Benders: ein eigenes Knoten-Kanten-Modell aus den Rohdaten des Netzes (andere Variablenreihenfolge, ohne `bnd_formulation`).

1. Optimalwert: Benders (obere = untere Schranke) = Aufzählen aller Entwürfe mit dem eigenen Fluss-LP = eigenes volles MILP.
2. Jeder Schnitt (Optimalität, Zulässigkeit, Pareto, Cut-Set) ist für jeden Entwurf gültig, den das eigene Modell als lieferbar kennt; Optimalitätsschnitte sind
   am Entstehungspunkt genau v(y), Zulässigkeitsschnitte dort verletzt.
3. Die Fehlmenge der Zulässigkeitsiterationen und die Flusskosten der Optimalitätsiterationen stimmen mit dem eigenen LP überein.
4. Die LP-Relaxation des Masters mit allen Schnitten ist die schwache LP-Schranke (eigenes LP)."""

import itertools

import numpy as np
import pytest

import bnd_benders as bd
import bnd_formulation as fm
import bnd_model as md
import bnd_scenario as sc

pytest.importorskip("scipy")
from scipy.optimize import LinearConstraint, Bounds, linprog, milp  # noqa: E402


class Own:
    def __init__(self, d):
        self.d, mcf, net = d, d.mcf, d.net
        self.m, self.K, self.G = len(net.arcs), mcf.K, d.G
        self.nx = self.m * self.K
        self.cost = np.zeros(self.nx)
        for e, (u, v, cap, c, kind) in enumerate(net.arcs):
            for k in range(self.K):
                if not mcf.reward[e]:
                    self.cost[e * self.K + k] = mcf.factors[k] * c if kind in (sc.K_LANE_IN, sc.K_LANE_OUT) else c
        self.demand = sum(mcf.ub[k][e] for e in range(self.m) if mcf.reward[e] for k in range(self.K))
        self.reward_cols = [e * self.K + k for e in range(self.m) if mcf.reward[e] for k in range(self.K)]
        rows = []
        for k in range(self.K):
            for v in range(net.n):
                if v in (net.s, net.t):
                    continue
                r = np.zeros(self.nx)
                for e, (a, b, *_rest) in enumerate(net.arcs):
                    r[e * self.K + k] += (b == v) - (a == v)
                rows.append(r)
        self.a_eq = np.array(rows)
        ax, ay, b = [], [], []
        for e, (a, bb, cap, c, kind) in enumerate(net.arcs):
            if not mcf.joint[e]:
                continue
            rx, ry = np.zeros(self.nx), np.zeros(self.G)
            rx[e * self.K:(e + 1) * self.K] = 1
            if d.group[e] >= 0:
                ry[d.group[e]] = -cap
                b.append(0.0)
            else:
                b.append(float(cap))
            ax.append(rx)
            ay.append(ry)
        self.ax, self.ay, self.b = np.array(ax), np.array(ay), np.array(b)

    def _bounds(self, full):
        lo, hi = np.zeros(self.nx), np.full(self.nx, np.inf)
        for e in range(self.m):
            for k in range(self.K):
                if self.d.mcf.ub[k][e] < md.BIG:
                    hi[e * self.K + k] = self.d.mcf.ub[k][e]
                    if full and self.d.mcf.reward[e]:
                        lo[e * self.K + k] = hi[e * self.K + k]
        return lo, hi

    def _lp(self, c, y, full):
        lo, hi = self._bounds(full)
        return linprog(c, A_ub=self.ax, b_ub=self.b - self.ay @ np.asarray(y, float), A_eq=self.a_eq, b_eq=np.zeros(len(self.a_eq)),
                       bounds=list(zip(lo, np.where(np.isinf(hi), None, hi))), method="highs")

    def v(self, y):
        res = self._lp(self.cost, y, True)
        return None if res.status != 0 else float(res.fun)

    def shortfall(self, y):
        c = np.zeros(self.nx)
        c[self.reward_cols] = -1
        return self.demand + self._lp(c, y, False).fun

    def mip(self, relax=False):
        lo, hi = self._bounds(True)
        c = np.concatenate([self.cost, np.array(self.d.fixed, float)])
        a_ub = np.hstack([self.ax, self.ay])
        a_eq = np.hstack([self.a_eq, np.zeros((len(self.a_eq), self.G))])
        integ = np.zeros(self.nx + self.G) if relax else np.concatenate([np.zeros(self.nx), np.ones(self.G)])
        res = milp(c, constraints=[LinearConstraint(a_ub, -np.inf, self.b), LinearConstraint(a_eq, 0, 0)], integrality=integ,
                   bounds=Bounds(np.concatenate([lo, np.zeros(self.G)]), np.concatenate([hi, np.ones(self.G)])))
        return None if res.x is None else float(res.fun)

    def brute_force(self):
        best = None
        for y in itertools.product((0, 1), repeat=self.G):
            v = self.v(y)
            if v is not None:
                best = v + float(np.dot(self.d.fixed, y)) if best is None else min(best, v + float(np.dot(self.d.fixed, y)))
        return best


def _nets():
    out = [md.bigm_net(), md.rounding_net(), md.bundle_net()]
    for seed in range(100000, 100200):
        d = md.generate_design(2, 2, 2, 60, 50, 40, seed, 2, 30)
        if 6 <= d.G <= 8 and Own(d).mip() is not None:
            out.append(d)
            if len(out) == 5:
                break
    return out


NETS = _nets()


def test_benders_optimum_equals_independent_brute_force_and_milp():
    assert len(NETS) == 5
    for d in NETS:
        own = Own(d)
        opt = own.mip()
        assert own.brute_force() == pytest.approx(opt, abs=1e-6)
        for level, pareto, start in ((fm.WEAK, False, "none"), (fm.STRONG, True, "cutset")):
            r = bd.benders(d, level=level, pareto=pareto, start=start)
            assert r.converged and r.ub == pytest.approx(opt, abs=1e-6) and r.lb == pytest.approx(opt, abs=1e-4)


def test_every_cut_valid_for_every_deliverable_design_and_tight_at_its_origin():
    for d in NETS[:4]:
        own = Own(d)
        designs = list(itertools.product((0, 1), repeat=d.G))
        value = {y: own.v(y) for y in designs}
        for kwargs in (dict(level=fm.WEAK), dict(level=fm.STRONG, pareto=True, start="cutset")):
            r = bd.benders(d, **kwargs)
            for cut in r.cuts:
                for y in designs:
                    v = value[y]
                    if v is None:
                        continue
                    assert (cut.at(y) <= v + 1e-5 * max(1.0, abs(v))) if cut.kind == "opt" else (cut.at(y) <= 1e-6)
                v0 = own.v(cut.y)
                if cut.kind == "opt":
                    assert v0 is not None and cut.at(cut.y) == pytest.approx(v0, abs=1e-4 * max(1.0, abs(v0)))
                else:
                    assert v0 is None and cut.at(cut.y) > 1e-9
            for yc in r.y_cuts:
                assert all(sum(a * y[g] for g, a in yc.coef.items()) >= yc.rhs - 1e-9 for y in designs if value[y] is not None)


def test_frame_numbers_match_the_independent_flow_lp():
    for d in NETS[:4]:
        own = Own(d)
        for f in bd.benders(d, level=fm.STRONG, pareto=True).frames:
            if f.kind == "opt":
                assert f.flow_cost == pytest.approx(own.v(f.y), abs=1e-5 * max(1.0, f.flow_cost))
            else:
                assert own.v(f.y) is None and f.shortfall == pytest.approx(own.shortfall(f.y), abs=1e-5 * max(1.0, own.demand))


def test_lp_master_bound_equals_independent_weak_lp_relaxation():
    for d in NETS[:4]:
        r = bd.benders(d, level=fm.WEAK, integer=False)
        assert r.converged and r.lb == pytest.approx(Own(d).mip(relax=True), rel=1e-6, abs=1e-6)
