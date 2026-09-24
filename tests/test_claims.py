"""Jede Zahl, die README und Hilfetexte nennen, ist hier belegt (Standardeinstellungen, feste Netze, Seeds ab 100000).
Das Optimum kommt aus HiGHS (MIP); Iterationen des Verfahrens hängen von der Wahl unter gleichwertigen Dualpreisen und Master-Lösungen ab und werden deshalb mit Bändern geprüft, Sekunden nur als grobes Verhältnis."""

import pytest

import bnd_constants as C
import bnd_evaluation as ev

P = ev.DEFAULT_PARAMS
GRID = P._replace(net="grid", k=3, fix=10)
PLAIN = ev.Opts("weak", "normal", "none", 200, "normal")
BEST = ev.Opts("strong", "pareto", "cutset", 200, "normal")


def test_teaching_nets_by_hand():
    """Big-M: 3 Iterationen (1 Zulässigkeits-, 2 Optimalitätsschnitte), Optimum 15. Rundungs-Falle: 2 Iterationen, Optimum 23 (Cut-Set y1 + y2 >= 2 vorab). Bündelung: etwa 8 Iterationen, davon 6 Zulässigkeit, Optimum 11."""
    big = ev.verdict(ev.analyse(P._replace(net="bigm")))[2]
    assert (big["total"], big["n_feas"], big["n_opt"], big["opt"]) == (3, 1, 2, 15.0)
    rnd = ev.verdict(ev.analyse(P._replace(net="rounding")))[2]
    assert (rnd["total"], rnd["opt"], rnd["y_cuts"]) == (2, 23.0, 1)
    bun = ev.verdict(ev.analyse(P._replace(net="bundle")))[2]
    assert bun["opt"] == 11.0 and 6 <= bun["total"] <= 10 and bun["n_feas"] >= 4 and bun["n_opt"] >= 2


def test_seed_155_numbers():
    """Zufallsnetz Seed 155 (starkes Teilproblem, Pareto, Cut-Set): Optimum 1 784 nach etwa 40 Iterationen (etwa 19 Optimalitäts-, 21 Zulässigkeitsschnitte), etwa 26 Cut-Set-Ungleichungen; HiGHS mindestens fünfmal schneller."""
    d = ev.verdict(ev.analyse(P))[2]
    assert d["opt"] == 1784.0 and d["converged"] and d["lb"] == pytest.approx(1784.0, abs=1e-3) and d["groups"] == 27
    assert 30 <= d["total"] <= 52 and 14 <= d["n_opt"] <= 26 and 15 <= d["n_feas"] <= 28 and 20 <= d["y_cuts"] <= 32
    assert d["seconds"] > 5 * d["highs_seconds"]


def test_plain_benders_preset():
    """Reines Benders (schwach, normal, ohne Start; 3/3/6, Seed 7): Optimum 1 348 nach etwa 71 Iterationen, davon etwa 63 (89 %) Zulässigkeitsschnitte; HiGHS mindestens zehnmal schneller."""
    d = ev.verdict(ev.analyse(P._replace(s=6, seed=7), PLAIN))[2]
    assert d["opt"] == 1348.0 and d["groups"] == 18 and d["converged"]
    assert 55 <= d["total"] <= 90 and 0.80 <= d["feas_share"] <= 0.95 and d["n_feas"] >= 45 and d["seconds"] > 10 * d["highs_seconds"]


def test_grid_preset():
    """Streckennetz (4 x 3, Seed 6, Fixkosten 10): Optimum 245 nach etwa 27 Iterationen, davon etwa 25 Zulässigkeitsschnitte."""
    d = ev.verdict(ev.analyse(GRID))[2]
    assert d["opt"] == 245.0 and d["converged"] and 20 <= d["total"] <= 36 and d["n_feas"] >= 18 and d["feas_share"] >= 0.8


def test_negative_controls():
    """Ohne Zulässigkeitsschnitte hängt der Master im Big-M-Netz nach 2 Iterationen (untere Schranke 0); ungültige Schnitte in der Rundungs-Falle: untere Schranke 24,15 über dem Optimum 23 nach 2 Iterationen."""
    s = ev.verdict(ev.analyse(P._replace(net="bigm"), ev.Opts("strong", "pareto", "cutset", 200, "nofeas")))
    assert s[1] == "stuck" and s[2]["iterations"] == 2 and s[2]["n_opt"] == 0 and s[2]["lb"] == pytest.approx(0.0, abs=1e-6)
    w = ev.verdict(ev.analyse(P._replace(net="rounding"), ev.Opts("strong", "pareto", "cutset", 200, "wrong")))
    assert w[1] == "wrong" and w[2]["iterations"] == 2 and 24.0 <= w[2]["lb"] <= 24.3 and w[2]["opt"] == 23.0


def test_distribution_default_net():
    """20 feste Netze (18 lieferbar, Standardvariante): Median etwa 18 Iterationen, etwa 49 % Zulässigkeitsschnitte, alle konvergiert; Benders im Mittel mindestens fünfmal langsamer als HiGHS."""
    d = ev.distribution(P)
    assert d["n_seeds"] == 20 and 17 <= d["n_feasible"] <= 19 and d["converged_share"] == 1.0
    assert 13 <= d["iterations_median"] <= 24 and 0.40 <= d["feas_share_mean"] <= 0.58 and d["seconds_mean"] > 5 * d["highs_mean"]


@pytest.fixture(scope="module")
def variants():
    return ev.variant_table(P)


def test_variant_matrix(variants):
    """3/3/6, 6 lieferbare Netze: Iterationen (Median) reines Benders 69, starkes Teilproblem 69, Pareto 50, LP-Wurzelschnitte 76 (einschließlich der LP-Iterationen), Cut-Set 14, Pareto + Cut-Set 12; Zulässigkeitsschnitte im reinen Verfahren
    86 % der Iterationen; alle konvergieren; HiGHS löst ein Netz in etwa 0,02 s, jede Variante braucht mindestens fünfmal länger."""
    by = {r["name"]: r for r in variants["rows"]}
    assert variants["n"] == 6 and all(r["converged"] == r["n"] for r in variants["rows"])
    plain, strong, pareto, lp, cutset, both = (by[n] for n in ("Reines Benders (schwach)", "Starkes Teilproblem", "+ Pareto-Schnitte", "+ LP-Wurzelschnitte", "Cut-Set im Master", "Pareto + Cut-Set im Master"))
    assert 55 <= plain["iterations"] <= 85 and abs(strong["iterations"] - plain["iterations"]) <= 0.2 * plain["iterations"] and 35 <= pareto["iterations"] <= 65 and pareto["iterations"] < plain["iterations"]
    assert 60 <= lp["iterations"] <= 100 and lp["iterations"] > 0.9 * strong["iterations"] and 8 <= cutset["iterations"] <= 22 and 7 <= both["iterations"] <= 18 and both["iterations"] <= cutset["iterations"]
    assert 0.78 <= plain["feas_share"] <= 0.93 and 0.50 <= lp["feas_share"] <= 0.75 and 0.38 <= cutset["feas_share"] <= 0.60
    assert all(r["seconds"] > 5 * variants["highs"] for r in variants["rows"]) and both["seconds"] < plain["seconds"] / 3


def test_sizes():
    """Benders (Pareto + Cut-Set) gegen HiGHS: 23 Entwurfskanten etwa 14 Iterationen (alle konvergiert), 42 Kanten etwa 66 Iterationen (3 von 5 konvergiert, Grenze 100), dort etwa hundertmal langsamer als HiGHS."""
    rows = ev.size_table(P)
    small, mid, big = rows
    assert small["groups"] < mid["groups"] < big["groups"] and 20 <= mid["groups"] <= 26 and 38 <= big["groups"] <= 46
    assert mid["converged"] == mid["n"] and 9 <= mid["iterations"] <= 22 and big["iterations"] >= 40 and big["converged"] < big["n"] and big["seconds"] > 20 * big["highs"]


def test_lp_master_bound_is_the_weak_lp_relaxation():
    """Die LP-Relaxation des Masters mit allen Schnitten ist in allen 8 Netzen genau die schwache LP-Schranke (66 bis 78 % des Optimums)."""
    rows = ev.lp_master_table(P)
    assert len(rows) == 8 and all(r["same"] for r in rows) and all(0.62 <= r["master"] <= 0.80 for r in rows) and all(r["master"] == pytest.approx(r["weak"], abs=1e-6) for r in rows)
