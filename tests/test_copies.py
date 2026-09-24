"""Die aus dem Vorgänger kopierten Bausteine (Netze, Formulierung, Schnitte) sind bewacht: dieselben Zahlen wie in fixkosten-netzdesign-demo und garg-koenemann-demo."""

import pytest

import bnd_cuts as cu
import bnd_formulation as fm
import bnd_model as md
import bnd_scenario as sc


def test_splitmix64_stream_is_the_portfolio_standard():
    rng = sc.SplitMix64(1)
    assert [rng.next() for _ in range(2)] == [10451216379200822465, 13757245211066428519]


def test_distribution_net_and_grid_of_the_predecessors():
    m = md.generate_mcf(3, 3, 8, 60, 50, 90, 155, 3)
    assert (m.total_demand(), m.net.m, m.net.n, m.K) == (76, 38, 19, 3)
    g = md.generate_grid(5, 4, 70, 2, 4, 4, 7)
    assert (g.net.m, g.net.n, g.K, g.total_demand()) == (64, 22, 4, 12)


def test_teaching_nets_of_the_predecessor():
    """Big-M: schwach 4, Optimum 15. Rundung: schwach 18, Optimum 23. Bündelung: schwach 6,5, stark und Optimum 11."""
    big, rnd, bun = md.bigm_net(), md.rounding_net(), md.bundle_net()
    assert fm.solve_lp(big, fm.WEAK).objective == pytest.approx(4.0) and fm.solve_mip(big).objective == pytest.approx(15.0)
    assert fm.solve_lp(rnd, fm.WEAK).objective == pytest.approx(18.0) and fm.solve_mip(rnd).objective == pytest.approx(23.0)
    assert fm.solve_lp(bun, fm.WEAK).objective == pytest.approx(6.5) and fm.solve_lp(bun, fm.STRONG).objective == pytest.approx(11.0) and fm.solve_mip(bun).objective == pytest.approx(11.0)


def test_standard_design_and_cut_loop_of_the_predecessor():
    """Standardnetz Seed 155: Optimum 1 784, 27 Entwurfskanten, schwach 69,8 %, mit Schnitten 93,4 %."""
    d = md.generate_design(3, 3, 8, 60, 50, 50, 155, 3, 30)
    opt = fm.solve_mip(d).objective
    assert opt == 1784.0 and d.G == 27 and fm.solve_lp(d, fm.WEAK).objective / opt == pytest.approx(0.698, abs=0.003)
    assert 0.92 <= cu.cut_and_solve(d).bound / opt <= 0.945
