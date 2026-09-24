"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios; ausblendbare Regler wie in multicommodity-demo: `seed_widget` vor dem Regler, Permalink und Preset nur in KEPT)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import bnd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _int_choice(options):
    def cast(value):
        value = int(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _float_choice(options):
    def cast(value):
        value = float(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _bool(value):
    if str(value) not in ("0", "1", "True", "False"):
        raise ValueError(value)
    return str(value) in ("1", "True")


SETTING_SPECS = {
    "net_select": SettingSpec("net", _choice(C.NETS), C.DEFAULT_NET),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "p_slider": SettingSpec("p", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "d_slider": SettingSpec("d", int, C.DEFAULT_D, C.D_MIN, C.D_MAX),
    "s_slider": SettingSpec("s", int, C.DEFAULT_S, C.S_MIN, C.S_MAX),
    "density_slider": SettingSpec("density", int, C.DEFAULT_DENSITY, C.DENSITY_MIN, C.DENSITY_MAX),
    "spread_slider": SettingSpec("spread", int, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "load_slider": SettingSpec("load", int, C.DEFAULT_LOAD, C.LOAD_MIN, C.LOAD_MAX),
    "fix_slider": SettingSpec("fix", int, C.DEFAULT_FIX, C.FIX_MIN, C.FIX_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "gw_slider": SettingSpec("gw", int, C.DEFAULT_GW, C.GW_MIN, C.GW_MAX),
    "gh_slider": SettingSpec("gh", int, C.DEFAULT_GH, C.GH_MIN, C.GH_MAX),
    "gdensity_slider": SettingSpec("gdensity", int, C.DEFAULT_GDENSITY, C.GDENSITY_MIN, C.GDENSITY_MAX),
    "gcap_slider": SettingSpec("gcap", int, C.DEFAULT_GCAP, C.GCAP_MIN, C.GCAP_MAX),
    "gdem_slider": SettingSpec("gdem", int, C.DEFAULT_GDEM, C.GDEM_MIN, C.GDEM_MAX),
    "gseed_input": SettingSpec("gseed", int, C.DEFAULT_GSEED, 0, C.SEED_MAX),
    "sub_radio": SettingSpec("sub", _choice(C.SUBS), C.DEFAULT_SUB),
    "cut_radio": SettingSpec("cut", _choice(C.CUTS), C.DEFAULT_CUT),
    "start_radio": SettingSpec("start", _choice(C.STARTS), C.DEFAULT_START),
    "iter_radio": SettingSpec("iters", _int_choice(C.ITERS), C.DEFAULT_ITER),
    "control_radio": SettingSpec("control", _choice(C.CONTROLS), C.DEFAULT_CONTROL),
}
PRESET_KEYS = {"net": "net_select", "k": "k_slider", "p": "p_slider", "d": "d_slider", "s": "s_slider", "density": "density_slider", "spread": "spread_slider", "load": "load_slider",
               "fix": "fix_slider", "seed": "seed_input", "gw": "gw_slider", "gh": "gh_slider", "gdensity": "gdensity_slider", "gcap": "gcap_slider", "gdem": "gdem_slider",
               "gseed": "gseed_input", "sub": "sub_radio", "cut": "cut_radio", "start": "start_radio", "iters": "iter_radio", "control": "control_radio"}
# Regler, die je nach Netz ausgeblendet sind: Streamlit löscht ihren Zustand, sobald sie nicht gezeichnet werden - der zuletzt gewählte Wert bleibt hier erhalten
KEPT = {key: f"_kept_{key}" for key in ("k_slider", "p_slider", "d_slider", "s_slider", "density_slider", "spread_slider", "load_slider", "fix_slider", "seed_input",
                                          "gw_slider", "gh_slider", "gdensity_slider", "gcap_slider", "gdem_slider", "gseed_input")}
STEPS = {"density_slider": 10, "spread_slider": 25, "load_slider": 10, "gdensity_slider": 10, "fix_slider": 5}


def init_session_state_defaults():
    """Standardwerte für Widgets, die in jedem Lauf gezeichnet werden. Regler, die je nach Netz ausgeblendet sind (KEPT), setzen ihren Wert erst im selben Lauf, in dem sie gezeichnet werden (`seed_widget`):
    ein Wert, der in einem Lauf ohne das Widget gesetzt wurde, erscheint sonst später als Mindestwert im Regler, während die App mit dem gesetzten Wert rechnet."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in KEPT and state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def seed_widget(state_key):
    """Vor dem Zeichnen eines ausgeblendbaren Reglers: fehlt sein Zustand, kommt der zuletzt gewählte (oder der Standard-) Wert."""
    if state_key not in st.session_state:
        st.session_state[state_key] = st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default)


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[KEPT.get(state_key, state_key)] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        kept = KEPT[key]
        if kept in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[kept] = int(lo + round((st.session_state[kept] - lo) / step) * step)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if state_key in KEPT:
            st.session_state[KEPT[state_key]] = C.PRESETS[name][key]
            st.session_state.pop(state_key, None)                     # der Regler nimmt den Wert aus KEPT, sobald er gezeichnet wird
        else:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed(state_key="seed_input"):
    st.session_state[state_key] = random.randint(0, C.SEED_MAX)
    st.session_state[KEPT[state_key]] = st.session_state[state_key]
