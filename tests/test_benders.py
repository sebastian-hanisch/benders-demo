"""Benders: Dualität, Gültigkeit jedes Schnitts (Aufzählen aller Entwürfe), Konvergenz gegen Brute Force und HiGHS, Schranken, Optionen, Negativkontrollen."""

import itertools

import numpy as np
import pytest

import bnd_benders as bd
import bnd_formulation as fm
import bnd_model as md


def tiny_nets(count=2, groups=(6, 11)):
    """Kleine lieferbare Distributionsnetze (wenige Entwurfsgruppen), damit das Aufzählen aller 2^G Entwürfe reicht."""
    out = []
    for seed in range(100000, 100200):
        d = md.generate_design(2, 2, 2, 60, 50, 40, seed, 2, 30)
        if groups[0] <= d.G <= groups[1]:
            try:
                fm.solve_lp(d, fm.WEAK)
            except fm.Infeasible:
                continue
            out.append(d)
            if len(out) == count:
                return out
    raise AssertionError("keine passenden Kleinstnetze")


TEACHING = (md.bigm_net, md.rounding_net, md.bundle_net)
NETS = [f() for f in TEACHING] + tiny_nets()


def _all_designs(d):
    return [np.array(bits, dtype=float) for bits in itertools.product((0, 1), repeat=d.G)]


def test_strong_duality_at_the_generating_point():
    """Am Entwurf, an dem ein Optimalitätsschnitt entsteht, ist seine rechte Seite genau v(y) (Vorzeichen der HiGHS-Marginalen stimmen)."""
    for d in NETS:
        for level in (fm.WEAK, fm.STRONG):
            p = bd.Problem(d, level)
            for y in _all_designs(d):
                cut, x, short = p.subproblem(y)
                if cut.kind == "opt":
                    assert cut.at(y) == pytest.approx(cut.value, abs=1e-6) and cut.value == pytest.approx(p.value(y), abs=1e-9) and short == 0.0
                    assert all(s <= 1e-9 for s in cut.slope)                          # eine geöffnete Kante spart Flusskosten
                else:
                    assert cut.at(y) > 1e-9 and short > 0 and p.value(y) is None      # der Zulässigkeitsschnitt verletzt seinen eigenen Entwurf


@pytest.mark.parametrize("level", [fm.WEAK, fm.STRONG])
@pytest.mark.parametrize("pareto", [False, True])
def test_every_cut_is_valid_for_every_design(level, pareto):
    """Kernargument: jeder Optimalitätsschnitt liegt für jeden zulässigen Entwurf unter v(y), jeder Zulässigkeitsschnitt wird von jedem zulässigen Entwurf erfüllt - aus Schnitten der Läufe und aus Schnitten an ALLEN Entwürfen."""
    for d in NETS:
        p = bd.Problem(d, level)
        designs = _all_designs(d)
        value = {tuple(y): p.value(y) for y in designs}
        cuts = list(bd.benders(d, level=level, pareto=pareto).cuts)
        core = bd.core_point(d, p)
        for y in designs:
            cut, x, short = p.subproblem(y)
            cuts.append(cut)
            if cut.kind == "opt" and pareto:
                better = p.pareto_cut(y, cut.value, core)
                if better is not None:
                    cuts.append(better)
        for cut in cuts:
            for y in designs:
                v = value[tuple(y)]
                if cut.kind == "opt":
                    if v is not None:
                        assert cut.at(y) <= v + 1e-5 * max(1.0, abs(v)), (cut.pareto, cut.y, y)
                elif v is not None:
                    assert cut.at(y) <= 1e-6, (cut.y, y)


def test_cuts_are_valid_at_fractional_designs_too():
    """Der LP-Wert v(y) ist konvex: die Schnitte gelten auch an gebrochenen Entwürfen (Grundlage der LP-Wurzelschnitte)."""
    rng = np.random.default_rng(3)
    for d in NETS[:4]:
        p = bd.Problem(d, fm.STRONG)
        cuts = list(bd.benders(d, level=fm.STRONG, pareto=True).cuts)
        for _ in range(30):
            y = rng.uniform(0.3, 1.0, d.G)
            v = p.value(y)
            for cut in cuts:
                if cut.kind == "opt" and v is not None:
                    assert cut.at(y) <= v + 1e-5 * max(1.0, abs(v))
                if cut.kind == "feas" and v is not None:
                    assert cut.at(y) <= 1e-6


def test_pareto_cut_dominates_at_the_core_point():
    for d in NETS:
        p = bd.Problem(d, fm.STRONG)
        core = bd.core_point(d, p)
        for y in _all_designs(d)[1:40]:
            cut, _, short = p.subproblem(y)
            if cut.kind != "opt":
                continue
            better = p.pareto_cut(y, cut.value, core)
            assert better is not None and better.pareto and better.at(y) >= cut.value - 1e-5 * max(1.0, abs(cut.value)) and better.at(core) >= cut.at(core) - 1e-6


@pytest.mark.parametrize("level,pareto,start", list(itertools.product((fm.WEAK, fm.STRONG), (False, True), ("none", "lp", "cutset"))))
def test_every_option_converges_to_the_optimum(level, pareto, start):
    for d in NETS[:4]:
        opt = fm.solve_mip(d).objective
        r = bd.benders(d, level=level, pareto=pareto, start=start)
        assert r.converged and not r.stuck and not r.crossed and r.ub == pytest.approx(opt, abs=1e-6) and r.lb == pytest.approx(opt, abs=1e-4)
        assert r.iterations == len(r.frames) and r.n_opt + r.n_feas == r.iterations and r.n_opt >= 1 and r.total_iterations == r.iterations + (r.pre_iterations if start == "lp" else 0) and (start == "lp") == (r.pre_iterations > 0)


def test_optimum_equals_brute_force_and_highs():
    for d in NETS:
        best = fm.brute_force(d)[0].objective
        assert fm.solve_mip(d).objective == pytest.approx(best, abs=1e-6)
        r = bd.benders(d)
        assert r.ub == pytest.approx(best, abs=1e-6) and r.lb == pytest.approx(best, abs=1e-4)
        y = r.y_best
        assert d.G == len(y) and fm.solve_fixed_design(d, {g for g in range(d.G) if y[g] > 0.5}).objective == pytest.approx(best, abs=1e-6)


def test_bounds_are_monotone_and_ordered():
    d = md.generate_design(3, 3, 8, 60, 50, 50, 155, 3, 30)
    r = bd.benders(d, pareto=True, start="cutset")
    lbs = [f.lb for f in r.frames]
    ubs = [f.ub for f in r.frames]
    assert lbs == sorted(lbs) and ubs == sorted(ubs, reverse=True) and all(lb <= ub + 1e-6 for lb, ub in zip(lbs, ubs))
    assert all(f.kind in ("opt", "feas") for f in r.frames) and r.frames[-1].kind == "opt" and r.converged


def test_teaching_nets_by_hand():
    """Big-M-Falle: drei Iterationen (eine Zulässigkeit, zwei Optimalität), Optimum 15; Rundungs-Falle 23; Bündelung 11."""
    big = bd.benders(md.bigm_net())
    assert (big.iterations, big.n_feas, big.n_opt) == (3, 1, 2) and big.ub == pytest.approx(15.0)
    assert bd.benders(md.rounding_net()).ub == pytest.approx(23.0) and bd.benders(md.bundle_net()).ub == pytest.approx(11.0)


def test_without_feasibility_cuts_the_master_gets_stuck():
    """Negativkontrolle: ohne Zulässigkeitsschnitte schlägt der Master immer wieder denselben unzulässigen Entwurf vor (y = 0 ist am billigsten)."""
    r = bd.benders(md.bigm_net(), feas_cuts=False)
    assert r.stuck and not r.converged and r.n_feas >= 1 and r.n_opt == 0


def test_invalid_cuts_push_the_lower_bound_over_the_optimum():
    """Negativkontrolle: um 5 % + 1 nach oben verschobene Optimalitätsschnitte sind ungültig - die untere Schranke überschreitet das Optimum."""
    for d in (md.rounding_net(), md.bundle_net(), tiny_nets()[0]):
        opt = fm.solve_mip(d).objective
        r = bd.benders(d, wrong=True)
        assert r.crossed or r.lb > opt + 1e-6 or r.ub > opt + 1e-6 or not r.converged


def test_lp_master_bound_equals_the_weak_lp_relaxation():
    """Die LP-Relaxation des Masters mit allen Schnitten ist die schwache LP-Schranke der Formulierung (die Schnitte beschreiben v(y) vollständig)."""
    for d in (md.bigm_net(), md.rounding_net(), md.bundle_net(), md.generate_design(3, 3, 8, 60, 50, 50, 155, 3, 30)):
        r = bd.benders(d, level=fm.WEAK, integer=False)
        assert r.converged and r.lb == pytest.approx(fm.solve_lp(d, fm.WEAK).objective, rel=1e-6, abs=1e-6)


def test_cutset_start_adds_y_cuts():
    d = md.generate_design(3, 3, 8, 60, 50, 50, 155, 3, 30)
    r = bd.benders(d, start="cutset")
    assert len(r.y_cuts) > 5 and r.converged
    assert bd.benders(d, start="none", max_iter=5).y_cuts == ()
