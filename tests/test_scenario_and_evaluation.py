"""Auswertung: Netzbau, Urteil, Verteilung, Experiment-Tabellen, Anzeige-Helfer."""

import pytest

import bnd_constants as C
import bnd_evaluation as ev
import bnd_model as md

P = ev.DEFAULT_PARAMS
GRID = P._replace(net="grid", k=3, fix=10)
PLAIN = ev.Opts("weak", "normal", "none", 200, "normal")


def test_build_dispatches_on_the_net_and_ignores_the_random_settings_for_fixed_nets():
    assert ev.build(P) == md.generate_design(3, 3, 8, 60, 50, 50, 155, 3, 30)
    assert ev.build(GRID) == md.generate_design_grid(4, 3, 60, 4, 2, 3, 6, 10)
    assert ev.build(P._replace(net="bigm")) == md.bigm_net() and ev.build(P._replace(net="rounding")) == md.rounding_net() and ev.build(P._replace(net="bundle")) == md.bundle_net()
    assert ev.normalise(P._replace(net="rounding", k=5, seed=9, fix=77)) == ev.normalise(P._replace(net="rounding", fix=5))
    assert ev.normalise(P._replace(net="grid", p=6, seed=9)) == ev.normalise(P._replace(net="grid", p=2, seed=77))
    assert ev.normalise(P._replace(gw=6, gseed=9)) == ev.normalise(P._replace(gw=3, gseed=1))


def test_with_seed_changes_only_the_seed_of_the_shown_net():
    assert ev.with_seed(P, 5).seed == 5 and ev.with_seed(P, 5).gseed == P.gseed
    assert ev.with_seed(GRID, 5).gseed == 5 and ev.with_seed(GRID, 5).seed == P.seed


def test_verdict_codes_and_data():
    lvl, code, d = ev.verdict(ev.analyse(P))
    assert (lvl, code) == ("success", "ok") and d["opt"] == 1784.0 and d["lb"] == pytest.approx(1784.0, abs=1e-3) and d["ub"] == pytest.approx(1784.0, abs=1e-6) and d["converged"]
    assert d["n_opt"] + d["n_feas"] == d["iterations"] == d["total"] and d["y_cuts"] > 0 and 0 < d["feas_share"] < 1 and d["highs_seconds"] < d["seconds"]
    assert ev.verdict(ev.analyse(P, ev.Opts("strong", "pareto", "cutset", 200, "nofeas")))[:2] == ("error", "stuck")
    assert ev.verdict(ev.analyse(P._replace(net="rounding"), ev.Opts("strong", "pareto", "cutset", 200, "wrong")))[:2] == ("error", "wrong")
    lvl, code, d = ev.verdict(ev.analyse(P, ev.Opts("strong", "normal", "none", 50, "normal")))
    assert (lvl, code) == ("warning", "limit") and not d["converged"] and d["lb"] < d["opt"] < d["ub"] and d["iterations"] == 50
    assert ev.verdict(ev.analyse(P._replace(seed=13)))[:2] == ("warning", "nothing")


def test_lp_start_counts_its_iterations():
    a = ev.analyse(P._replace(net="bundle"), ev.Opts("strong", "pareto", "lp", 200, "normal"))
    d = ev.verdict(a)[2]
    assert d["pre"] > 0 and d["total"] == d["iterations"] + d["pre"]


def test_distribution_is_consistent_with_single_runs():
    seeds = tuple(range(100000, 100004))
    dist = ev.distribution(P, seeds=seeds)
    feasible = [s for s in seeds if ev.analyse(ev.with_seed(P, s)).feasible]
    assert dist["n_seeds"] == 4 and dist["n_feasible"] == len(feasible) and dist["n_infeasible"] == 4 - len(feasible)
    for i, seed in enumerate(feasible):
        d = ev.verdict(ev.analyse(ev.with_seed(P, seed)))[2]
        assert dist["cols"]["iterations"][i] == d["total"] and dist["cols"]["feas_share"][i] == pytest.approx(d["feas_share"])
    assert dist["converged_share"] == 1.0 and ev.distribution(GRID, seeds=seeds)["n_seeds"] == 4


def test_variant_table_shape_and_ordering():
    t = ev.variant_table(P, size=(3, 3, 6), seeds=C.VARIANT_SEEDS[:3])
    assert [r["name"] for r in t["rows"]] == [v[0] for v in ev.VARIANTS] and t["n"] >= 2
    by = {r["name"]: r for r in t["rows"]}
    assert all(r["converged"] == r["n"] for r in t["rows"]) and by["Pareto + Cut-Set im Master"]["iterations"] <= by["Reines Benders (schwach)"]["iterations"]
    assert all(r["seconds"] > t["highs"] for r in t["rows"])


def test_size_table_shape():
    rows = ev.size_table(P, sizes=((2, 2, 4), (3, 3, 6)), seeds=C.SIZE_SEEDS, iters=30)
    assert [r["size"] for r in rows] == [(2, 2, 4), (3, 3, 6)] and rows[0]["groups"] < rows[1]["groups"] and all(r["seconds"] > r["highs"] for r in rows)


def test_control_and_lp_master_tables():
    rows = {r["control"]: r for r in ev.control_table(P._replace(net="rounding"))}
    assert rows["normal"]["converged"] and rows["nofeas"]["stuck"] or rows["nofeas"]["iterations"] >= 1
    assert rows["wrong"]["crossed"] or rows["wrong"]["lb"] > rows["wrong"]["opt"] + 1e-6 or not rows["wrong"]["converged"]
    lt = ev.lp_master_table(P, seeds=C.VARIANT_SEEDS[:4])
    assert lt and all(r["same"] and r["master"] == pytest.approx(r["weak"], abs=1e-6) and r["master"] < 1 for r in lt)


def test_cut_words_and_names():
    a = ev.analyse(P, ev.Opts("strong", "pareto", "cutset", 200, "normal"))
    frames = a.result.frames
    text = ev.cut_words(a.design, next(f.cut for f in frames if f.kind == "opt" and abs(f.cut.slope).max() > 0))
    assert text.startswith("θ ≥") and "→" in text and "Preis" in text
    feas = ev.cut_words(a.design, next(f.cut for f in frames if f.kind == "feas"))
    assert feas.startswith("Der Entwurf lässt") and "→" in feas
    r = ev.analyse(P._replace(net="rounding"), PLAIN)
    assert ev.arc_name(r.design, 0) == "A → B" and ev.group_name(r.design, 1) == "A → B"
