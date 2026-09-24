"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Optionen, Randgrößen, Iterations-Regler, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import itertools
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bnd_constants as C
import bnd_evaluation as ev
from bnd_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# Anfang der Meldung zum gezeigten Netz (Streamlit legt das führende Emoji in `icon`, nicht in `value`)
EXPECTED = {
    "🚚 Zufallsnetz": "Optimum 1784,0 nach **",
    "🗺️ Streckennetz": "Optimum 245,0 nach **",
    "💸 Big-M-Falle": "Optimum 15,0 nach **3 Iterationen**",
    "🔁 Rundungs-Falle": "Optimum 23,0 nach **2 Iterationen**",
    "📦 Bündelung": "Optimum 11,0 nach **",
    "🧱 Reines Benders": "Optimum 1348,0 nach **",
    "🚫 Ohne Zulässigkeitsschnitte": "Ohne Zulässigkeitsschnitte bleibt der Master hängen",
    "❌ Ungültige Schnitte": "Die verschobenen Optimalitätsschnitte sind ungültig:",
}
OPTION_LABELS = {"Teilproblem", "Schnitte", "Start des Masters", "Iterationsgrenze", "Kontrolle"}


def _run(setup=None, timeout=900):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.radio)}


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info) + list(at.error)]


def _has(at, prefix):
    return any(t.startswith(prefix) for t in _texts(at))


def _metric(at, label):
    return [m.value for m in at.metric if m.label == label]


def _frame_slider(at):
    found = [s for s in at.slider if s.key == "bnd_frame"]
    return found[0] if found else None


def test_default_renders_without_exception():
    at = _run()
    assert any("Iteration für Iteration" in m.value for m in at.markdown)
    assert _has(at, EXPECTED["🚚 Zufallsnetz"]) and not at.error
    assert _metric(at, "Optimum (HiGHS)")[0] == "1784,0" and _metric(at, "Iterationen")[0].isdigit() and _metric(at, "Sekunden")
    assert _frame_slider(at).value == _frame_slider(at).max >= 15


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdicts(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert _has(at, EXPECTED[name]), _texts(at)
    if C.PRESETS[name]["net"] in C.FIXED_NETS:
        assert any(t.startswith("Festes Netz") for t in _texts(at))


@pytest.mark.parametrize("K", range(C.K_MIN, C.K_MAX + 1))
@pytest.mark.parametrize("net", ["random", "grid"])
def test_every_number_of_goods_renders(net, K):
    def setup(at):
        at.session_state["net_select"] = net
        at.session_state["k_slider"] = K
        at.session_state["iter_radio"] = 50
    at = _run(setup)
    assert not at.error and (_has(at, "Optimum") or _has(at, "Nach ") or _has(at, "Dieses Netz kann"))


@pytest.mark.parametrize("sub,cut,start", list(itertools.product(C.SUBS, C.CUTS, C.STARTS)))
def test_every_option_combination_renders_on_a_small_net(sub, cut, start):
    def setup(at):
        at.session_state["net_select"] = "bundle"
        at.session_state["sub_radio"] = sub
        at.session_state["cut_radio"] = cut
        at.session_state["start_radio"] = start
    at = _run(setup)
    assert _has(at, "Optimum 11,0 nach **") and not at.error


@pytest.mark.parametrize("control", list(C.CONTROLS))
def test_every_control_renders(control):
    net = "bigm" if control == "nofeas" else "rounding"           # ohne Zulässigkeitsschnitte hängt nur ein Netz, das der Master nicht sofort trifft; die Rundungs-Falle bekommt ihren Schnitt vorab
    at = _run(lambda a: (a.session_state.__setitem__("net_select", net), a.session_state.__setitem__("control_radio", control)))
    assert (control == "normal") != bool(at.error)


def test_extreme_sizes_render():
    for net, vals in (("random", (("p_slider", C.P_MIN), ("d_slider", C.D_MIN), ("s_slider", C.S_MIN), ("density_slider", C.DENSITY_MIN), ("spread_slider", C.SPREAD_MIN), ("load_slider", C.LOAD_MIN), ("k_slider", C.K_MIN), ("fix_slider", C.FIX_MIN))),
                      ("grid", (("gw_slider", C.GW_MIN), ("gh_slider", C.GH_MIN), ("gdensity_slider", C.GDENSITY_MIN), ("gcap_slider", C.GCAP_MIN), ("gdem_slider", C.GDEM_MIN), ("k_slider", C.K_MIN))),
                      ("grid", (("gw_slider", C.GW_MAX), ("gh_slider", C.GH_MAX), ("gdensity_slider", C.GDENSITY_MAX), ("gcap_slider", C.GCAP_MAX), ("gdem_slider", C.GDEM_MAX), ("k_slider", C.K_MAX)))):
        def setup(at, net=net, vals=vals):
            at.session_state["net_select"] = net
            at.session_state["iter_radio"] = 50
            for key, value in vals:
                at.session_state[key] = value
        at = _run(setup)
        assert not at.error


def test_a_net_that_cannot_deliver_renders_and_says_so():
    """Seed 13 im Standardnetz: die Nachfrage lässt sich auch mit allen Kanten offen nicht decken - Hinweis, kein Absturz."""
    at = _run(lambda a: a.session_state.__setitem__("seed_input", 13))
    assert any("kann die Nachfrage nicht decken" in t for t in _texts(at))


def test_iteration_slider_moves_through_all_frames():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "bundle"))
    top = int(_frame_slider(at).max)
    assert top == 8
    for value in range(1, top + 1):
        _frame_slider(at).set_value(value)
        at.run()
        assert not at.exception and _frame_slider(at).value == value


def test_a_single_iteration_has_no_slider():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "rounding"))
    assert _frame_slider(at) is not None                           # 2 Iterationen: Regler mit min < max
    at = _run(lambda a: (a.session_state.__setitem__("net_select", "rounding"), a.session_state.__setitem__("control_radio", "wrong")))
    assert not at.exception


def test_hidden_controls_keep_their_values_across_a_net_switch():
    at = _run()
    at.sidebar.slider(key="fix_slider").set_value(80)
    at.run()
    at.sidebar.selectbox(key="net_select").set_value("bigm")
    at.run()
    assert not at.exception and not [w for w in at.sidebar.slider if w.key == "fix_slider"]
    at.sidebar.selectbox(key="net_select").set_value("random")
    at.run()
    assert at.sidebar.slider(key="fix_slider").value == 80 and not at.exception


def test_permalink_settings_are_loaded_and_clamped():
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.query_params["net"] = "grid"
    at.query_params["k"] = "9"
    at.query_params["fix"] = "12"
    at.query_params["sub"] = "weak"
    at.query_params["start"] = "lp"
    at.query_params["iters"] = "50"
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="net_select").value == "grid" and at.sidebar.slider(key="k_slider").value == C.K_MAX and at.sidebar.slider(key="fix_slider").value == 10
    assert at.sidebar.radio(key="sub_radio").value == "weak" and at.sidebar.radio(key="start_radio").value == "lp" and at.sidebar.radio(key="iter_radio").value == 50


def test_option_controls_exist():
    assert OPTION_LABELS <= _labels(_run())


def test_distribution_runs_on_demand(monkeypatch):
    orig = ev.distribution
    monkeypatch.setattr(ev, "distribution", lambda params, opts=ev.DEFAULT_OPTS, seeds=C.QUALITY_SEEDS[:3]: orig(params, opts, seeds))
    at = _run()
    assert not [m for m in at.metric if m.label == "Iterationen (Median)"]
    next(b for b in at.button if b.key == "dist_start").click().run()
    assert not at.exception and _metric(at, "Iterationen (Median)") and _metric(at, "Konvergiert")[0] == "100 %"


def test_experiments_run_on_demand(monkeypatch):
    v_orig, s_orig = ev.variant_table, ev.size_table
    monkeypatch.setattr(ev, "variant_table", lambda params: v_orig(params, size=(2, 2, 4), seeds=C.VARIANT_SEEDS[:8], iters=100))
    monkeypatch.setattr(ev, "size_table", lambda params: s_orig(params, sizes=((2, 2, 4), (3, 3, 6)), seeds=C.SIZE_SEEDS, iters=60))
    at = _run()
    next(b for b in at.button if b.key == "variants_start").click().run()
    assert not at.exception and any("Das reine Verfahren verbringt die meisten Iterationen" in c.value for c in at.caption)
    next(b for b in at.button if b.key == "sizes_start").click().run()
    assert not at.exception and any("HiGHS löst das ganze MIP" in c.value for c in at.caption)
    next(b for b in at.button if b.key == "controls_start").click().run()
    assert not at.exception and any("Anteil am Optimum. Die Schnitte beschreiben" in c.value for c in at.caption)


def test_experiments_need_a_random_net():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "bundle"))
    assert not [b for b in at.button if b.key in ("dist_start", "variants_start", "sizes_start")] and sum(1 for t in _texts(at) if t.startswith("Für dieses Experiment")) == 2


def test_source_has_explicit_chart_keys_and_locked_axes():
    app = APP.read_text(encoding="utf-8")
    assert all(re.search(r"plotly_chart\(.*key=", line) for line in app.splitlines() if "st.plotly_chart(" in line)
    viz = (ROOT / "bnd_visualization.py").read_text(encoding="utf-8")
    assert viz.count("return lock_axes(fig)") >= 5 and "def lock_axes" in viz
