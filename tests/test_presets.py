"""Presets: vollständig, in den Grenzen, und jedes Beispiel zeigt, was sein Hilfetext behauptet."""

import bnd_constants as C
import bnd_evaluation as ev
import bnd_presets as P

KEYS = set(P.PRESET_KEYS)


def _params(p):
    return ev.NetParams(p["net"], p["k"], p["p"], p["d"], p["s"], p["density"], p["spread"], p["load"], p["fix"], p["seed"], p["gw"], p["gh"], p["gdensity"], p["gcap"], p["gdem"], p["gseed"])


def _opts(p):
    return ev.Opts(p["sub"], p["cut"], p["start"], p["iters"], p["control"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


def test_preset_values_are_inside_the_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert p["net"] in C.NETS and p["sub"] in C.SUBS and p["cut"] in C.CUTS and p["start"] in C.STARTS and p["iters"] in C.ITERS and p["control"] in C.CONTROLS
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        for state_key, step in P.STEPS.items():
            key = next(k for k, v in P.PRESET_KEYS.items() if v == state_key)
            assert (p[key] - P.SETTING_SPECS[state_key].lo) % step == 0, (name, key)


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS and p["gseed"] not in C.DIST_SEEDS, name


def test_defaults_equal_the_random_net_preset():
    p = C.PRESETS["🚚 Zufallsnetz"]
    assert _params(p) == ev.DEFAULT_PARAMS and _opts(p) == ev.DEFAULT_OPTS


def test_every_preset_net_is_deliverable():
    for name, p in C.PRESETS.items():
        assert ev.analyse(_params(p), _opts(p)).feasible, name


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"💸 Big-M-Falle", "🔁 Rundungs-Falle", "📦 Bündelung", "🚫 Ohne Zulässigkeitsschnitte", "❌ Ungültige Schnitte"}


def test_kept_covers_exactly_the_hideable_controls():
    hideable = {"k_slider", "p_slider", "d_slider", "s_slider", "density_slider", "spread_slider", "load_slider", "fix_slider", "seed_input", "gw_slider", "gh_slider", "gdensity_slider", "gcap_slider", "gdem_slider", "gseed_input"}
    assert set(P.KEPT) == hideable and set(P.SETTING_SPECS) - hideable == {"net_select", "sub_radio", "cut_radio", "start_radio", "iter_radio", "control_radio"}
