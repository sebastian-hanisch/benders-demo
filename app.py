"""Benders-Zerlegung – Entwurf im Master, Fluss im Teilproblem – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die Benders-Zerlegung für das Fixkosten-Netzdesign - und lässt stattdessen das Beispiel wachsen.
Elftes Stück der Netzwerkfluss-Linie der "Konzepte"-Reihe, zweites im Netzwerkdesign-Ast: dasselbe Modell wie das Vorgängerstück, aber statt einer Schranke geht es um ein Verfahren, das den ganzzahligen Entwurf vom Fluss trennt. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import bnd_constants as C
import bnd_evaluation as ev
from bnd_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from bnd_visualization import (
    build_bounds,
    build_dist,
    build_master_map,
    build_sizes,
    build_variants,
)

st.set_page_config(page_title="Benders-Zerlegung – Sebastian Hanisch", layout="wide")


def _pct(x, digits=0):
    return "–" if x is None or x != x else f"{100 * x:.{digits}f} %".replace(".", ",")


def _f(x, digits=1):
    return "–" if x is None or x != x else f"{x:.{digits}f}".replace(".", ",")


def _n(x):
    return f"{int(round(x)):,}".replace(",", " ")


def _sec(x):
    return f"{x:.2f}".replace(".", ",") if x >= 0.1 else f"{x:.3f}".replace(".", ",")


@st.cache_resource(show_spinner=False, max_entries=48)
def _analysis(params, opts):
    return ev.analyse(params, opts)


st.title("🧩 Benders-Zerlegung – Entwurf im Master, Fluss im Teilproblem")
st.markdown(
    """
Beim Fixkosten-Netzdesign hängt alles Schwierige an **einer** Sorte Entscheidungen: welche Kanten geöffnet werden ($y\\in\\{0,1\\}$). Ist der Entwurf gewählt, ist der Rest ein gewöhnlicher Mehrgüterfluss - ein LP, das sich schnell lösen lässt und Dualpreise liefert.
**Benders** trennt genau das: ein **Master** wählt den Entwurf (ganzzahlig, mit einer Variable $\\theta$ für die Flusskosten), ein **Teilproblem** rechnet den Fluss dazu. Aus den **Dualpreisen der Kapazitäten** wird ein Schnitt für den Master: $\\theta\\ge v(\\bar y)+\\sum_g\\lambda_g(y_g-\\bar y_g)$ (Optimalitätsschnitt), oder, wenn der Entwurf die Nachfrage gar nicht decken kann, eine Bedingung, die alle ähnlich unzulässigen Entwürfe ausschließt (Zulässigkeitsschnitt).
Die Schranken laufen aufeinander zu: der Master gibt eine untere, der beste bisher gerechnete Entwurf mit seinem Fluss eine obere. Diese Demo lässt die Iterationen einzeln durchlaufen, misst, was das Verfahren kostet - und was die Schnitte aus dem Vorgängerstück daran ändern.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - elftes Stück der Netzwerkfluss-Linie der \"Konzepte\"-Reihe, zweites im Netzwerkdesign-Ast - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Vorgängerstück [fixkosten-netzdesign-demo](https://github.com/sebastian-hanisch/fixkosten-netzdesign-demo) misst, wie schwach die LP-Schranke ist und was Schnitte über den Entwurf y ändern; hier ist das Verfahren der Zerlegung dran. "
    "Das Folgestück: **Slope Scaling** (Heuristik für große Netze)."
)

with st.expander("So funktioniert die Zerlegung", expanded=True):
    st.markdown(
        r"""
1. **Teilproblem** (Entwurf $\bar y$ fest): $v(\bar y)=\min c\cdot x$ mit Flusserhaltung je Gut und Kapazität $\sum_k x^k_e\le u_e\bar y_g$ - ein LP. Sein Wert $v(y)$ ist in $y$ **konvex** und stückweise linear.
2. **Optimalitätsschnitt:** jede dual zulässige Lösung des Teilproblems gibt eine untere Schranke $v(y)\ge \text{const}+\text{slope}\cdot y$; im Punkt $\bar y$ ist sie exakt, wenn die Dualpreise optimal sind. $\text{slope}_g\le 0$ ist der **Preis** der Kante $g$: was das Öffnen dem Fluss spart.
3. **Zulässigkeitsschnitt:** kann $\bar y$ nicht liefern, minimiert ein Phase-1-LP die Fehlmenge; sein Dual verlangt Kapazität dort, wo sie fehlt.
4. **Master:** $\min f\cdot y+\theta$ über $y\in\{0,1\}^G$, $\theta\ge$ alle Optimalitätsschnitte, alle Zulässigkeitsschnitte erfüllt. Er wird mit jeder Iteration schärfer; **untere Schranke** = sein Wert, **obere** = bester Entwurf mit Fluss.
5. **Pareto-Schnitte (Magnanti-Wong):** oft gibt es viele optimale Dualpreise; man nimmt den, dessen Schnitt an einem inneren Punkt am höchsten liegt - gleiche Gültigkeit, stärker.
6. **Aus dem Vorgänger:** die Cut-Set-Ungleichungen (nur über $y$) sind für den Master gültig; man kann sie vorab hineinlegen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()


def _kept(key, default):
    return int(st.session_state.get(KEPT[key], default))


def _slider(label, key, help, step=None):
    kw = {"step": step} if step else {}
    seed_widget(key)
    value = st.slider(label, *bounds(key), key=key, help=help, **kw)
    st.session_state[KEPT[key]] = value
    return value


with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Das Distributionsnetz der Vorgänger (dreistufig: Werke, Verteilzentren, Filialen), ein Streckennetz (Gitter mit Start-Ziel-Aufträgen) oder eines von drei festen Lehrnetzen, an denen sich je ein Effekt von Hand nachrechnen lässt.",
    )
    show_dist, show_grid = net_key == "random", net_key == "grid"
    if show_dist or show_grid:
        K = _slider("Zahl der Güter", "k_slider", "Frische, Trocken, Kühl, Getränke, Tiefkühl (in dieser Reihenfolge). Im Distributionsnetz stellt jedes Werk jedes Gut her; im Streckennetz hat jedes Gut einen eigenen Start und ein eigenes Ziel.")
        fix = _slider("Fixkosten je Lane (Basis)", "fix_slider", "Jede Lane (im Streckennetz jede Strecke) kostet die Basis mal 1 bis 3, ein Verteilzentrum das Doppelte. Zum Vergleich: die Stückkosten je Einheit liegen bei 1 bis 9.", step=5)
    else:
        K = _kept("k_slider", C.DEFAULT_K)
        fix = _kept("fix_slider", C.DEFAULT_FIX)
    if show_dist:
        p = _slider("Werke", "p_slider", "Anzahl der Werke (oben im Netz).")
        d = _slider("Verteilzentren", "d_slider", "Anzahl der Verteilzentren; Durchsatz gemeinsam für alle Güter.")
        s = _slider("Filialen", "s_slider", "Anzahl der Filialen (unten im Netz). Mehr Filialen heißt mehr Entwurfskanten: der Master wird schwerer.")
        density = _slider("Netzdichte [%]", "density_slider", "Anteil der möglichen Lanes, die es gibt.", step=10)
        spread = _slider("Streuung der Lane-Breiten [%]", "spread_slider", "0 = alle Lanes einer Stufe gleich breit, 100 = Kapazitäten gleichverteilt von 1 bis zum Doppelten der Grundbreite.", step=25)
        load = _slider("Auslastung [% der Werkskapazität]", "load_slider", "Gesamtnachfrage der Filialen (alle Güter zusammen) in Prozent der Werkskapazität.", step=10)
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, args=("seed_input",), help="Würfelt einen neuen Zufalls-Seed. Die Verteilung über feste Netze weiter unten ändert sich dabei nicht - nur die Marke „Ihre Ziehung“.")
    else:
        p, d, s = _kept("p_slider", C.DEFAULT_P), _kept("d_slider", C.DEFAULT_D), _kept("s_slider", C.DEFAULT_S)
        density, spread, load, seed = _kept("density_slider", C.DEFAULT_DENSITY), _kept("spread_slider", C.DEFAULT_SPREAD), _kept("load_slider", C.DEFAULT_LOAD), _kept("seed_input", C.DEFAULT_SEED)
    if show_grid:
        gw = _slider("Breite des Gitters", "gw_slider", "Knoten je Zeile.")
        gh = _slider("Höhe des Gitters", "gh_slider", "Knoten je Spalte.")
        gdensity = _slider("Anteil der Gitterkanten [%]", "gdensity_slider", "Ein zufälliger Spannbaum hält das Netz zusammenhängend; jede weitere Gitterkante gibt es mit diesem Anteil. 100 % = volles Gitter.", step=10)
        gcap = _slider("Größte Kapazität je Kante", "gcap_slider", "Jede Kante trägt in beide Richtungen 1 bis zu diesem Wert, gemeinsam für alle Güter.")
        gdem = _slider("Größte Menge je Gut", "gdem_slider", "Jedes Gut fährt 1 bis zu diesem Wert von seinem Start zu seinem Ziel.")
        seed_widget("gseed_input")
        gseed = st.number_input("Zufalls-Seed (Streckennetz)", *bounds("gseed_input"), key="gseed_input", step=1)
        st.session_state[KEPT["gseed_input"]] = gseed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, args=("gseed_input",), key="rand_grid", help="Würfelt einen neuen Zufalls-Seed für das Streckennetz.")
    else:
        gw, gh, gdensity = _kept("gw_slider", C.DEFAULT_GW), _kept("gh_slider", C.DEFAULT_GH), _kept("gdensity_slider", C.DEFAULT_GDENSITY)
        gcap, gdem, gseed = _kept("gcap_slider", C.DEFAULT_GCAP), _kept("gdem_slider", C.DEFAULT_GDEM), _kept("gseed_input", C.DEFAULT_GSEED)
    if not (show_dist or show_grid):
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Die Regler für Güter, Größe, Fixkosten und Seed gehören zu den zufälligen Netzen.")
    st.subheader("Verfahren")
    sub = st.radio("Teilproblem", list(C.SUBS), key="sub_radio", format_func=lambda k: C.SUBS[k],
                   help="Wie eng der Fluss an die Entscheidung gebunden ist. Stark: zusätzlich jedes Gut einzeln gegen min(Kapazität, Nachfrage) - das Teilproblem hat dann mehr Zeilen und Preise. Beide haben dasselbe Optimum.")
    cut = st.radio("Schnitte", list(C.CUTS), key="cut_radio", format_func=lambda k: C.CUTS[k],
                   help="Normal: die Dualpreise, die der LP-Löser gerade liefert (eine beliebige optimale Wahl). Pareto: unter allen optimalen Dualpreisen die mit dem höchsten Schnitt an einem inneren Punkt.")
    start = st.radio("Start des Masters", list(C.STARTS), key="start_radio", format_func=lambda k: C.STARTS[k],
                     help="Ohne Start beginnt der Master leer und schlägt zuerst Entwürfe vor, die nichts liefern. LP-Wurzelschnitte: erst Benders am LP-Master (y zwischen 0 und 1), deren Schnitte kommen in den ganzzahligen Master; die LP-Iterationen zählen mit. Cut-Set: die Ungleichungen aus dem Vorgängerstück.")
    iters = st.radio("Iterationsgrenze", list(C.ITERS), key="iter_radio", help="Obergrenze der Iterationen des ganzzahligen Masters. Ohne Start braucht das reine Verfahren oft mehr als 100.")
    control = st.radio("Kontrolle", list(C.CONTROLS), key="control_radio", format_func=lambda k: C.CONTROLS[k],
                       help="Die Negativkontrollen zeigen, warum jede Zutat nötig ist: ohne Zulässigkeitsschnitte bleibt der Master bei einem unbedienbaren Entwurf hängen; ungültig verschobene Optimalitätsschnitte treiben die untere Schranke über das Optimum.")

sync_query_params({"net_select": net_key, "k_slider": int(K), "p_slider": int(p), "d_slider": int(d), "s_slider": int(s), "density_slider": int(density), "spread_slider": int(spread),
                   "load_slider": int(load), "fix_slider": int(fix), "seed_input": int(seed), "gw_slider": int(gw), "gh_slider": int(gh), "gdensity_slider": int(gdensity), "gcap_slider": int(gcap),
                   "gdem_slider": int(gdem), "gseed_input": int(gseed), "sub_radio": sub, "cut_radio": cut, "start_radio": start, "iter_radio": iters, "control_radio": control})

params = ev.normalise(ev.NetParams(net_key, int(K), int(p), int(d), int(s), int(density), int(spread), int(load), int(fix), int(seed), int(gw), int(gh), int(gdensity), int(gcap), int(gdem), int(gseed)))
opts = ev.Opts(sub, cut, start, int(iters), control)
with st.spinner("Rechne..."):
    a = _analysis(params, opts)
design = a.design
level, code, dat = ev.verdict(a)
is_fixed = net_key in C.FIXED_NETS

# --- Iterationen ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🧩 Iteration für Iteration")
if not a.feasible:
    st.warning("⚠️ Dieses Netz kann die Nachfrage nicht decken, auch wenn alle Kanten offen sind - es gibt nichts zu entwerfen. Weniger Auslastung, mehr Lanes oder ein anderer Seed hilft.")
    st.stop()

res = a.result
opt = dat["opt"]
ratio = dat["seconds"] / dat["highs_seconds"] if dat["highs_seconds"] else float("nan")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Optimum (HiGHS)", _f(opt, 1), delta=f"{design.G} Entwurfskanten", delta_color="off", help="Das ganzzahlige Optimum aus dem direkten MIP von HiGHS - die Vergleichsgröße.")
m2.metric("Iterationen", _n(dat["total"]), delta=f"{dat['n_opt']} Optimalität, {dat['n_feas']} Zulässigkeit", delta_color="off", help="Ganzzahlige Master-Läufe; bei LP-Wurzelschnitten zählen die LP-Iterationen mit.")
m3.metric("Untere / obere Schranke", f"{_f(dat['lb'], 1)} / {_f(dat['ub'], 1) if dat['ub'] != float('inf') else '–'}", delta=f"Lücke {_pct(dat['gap'], 1) if dat['gap'] == dat['gap'] else '–'}", delta_color="off", help="Master-Wert und bester Entwurf mit Fluss am Ende.")
m4.metric("Sekunden", _sec(dat["seconds"]), delta=f"HiGHS {_sec(dat['highs_seconds'])} s", delta_color="off", help="Rechenzeit des Verfahrens (nur zur Information); davon der Master: " + _pct(dat["master_share"]) + ".")

if code == "ok":
    pre = f" (dazu {dat['pre']} LP-Iterationen im Start)" if dat["pre"] else ""
    st.success(f"✅ Optimum {_f(opt, 1)} nach **{_n(dat['total'])} Iterationen**{pre}: {dat['n_opt']} Optimalitäts- und {dat['n_feas']} Zulässigkeitsschnitte ({_pct(dat['feas_share'])} Zulässigkeit), {dat['y_cuts']} Cut-Set-Ungleichungen im Master. "
               f"{_sec(dat['seconds'])} Sekunden - das direkte MIP von HiGHS braucht {_sec(dat['highs_seconds'])} Sekunden ({_f(ratio, 0)}-mal weniger).")
elif code == "limit":
    st.warning(f"⚠️ Nach {_n(dat['total'])} Iterationen (Grenze) sind die Schranken noch nicht zusammen: untere {_f(dat['lb'], 1)}, obere {_f(dat['ub'], 1) if dat['ub'] != float('inf') else '–'}, Optimum {_f(opt, 1)}. "
               f"{_pct(dat['feas_share'])} der Iterationen waren Zulässigkeitsschnitte - der Master lernt zuerst, welche Entwürfe überhaupt liefern.")
elif code == "stuck":
    st.error("❌ Ohne Zulässigkeitsschnitte bleibt der Master hängen: er schlägt denselben Entwurf immer wieder vor, weil er kein Wissen darüber hat, dass dieser die Nachfrage nicht deckt (der billigste Entwurf öffnet fast nichts).")
else:
    st.error(f"❌ Die verschobenen Optimalitätsschnitte sind ungültig: die untere Schranke {_f(dat['lb'], 1)} liegt {'über' if dat['lb'] > opt else 'nahe'} dem Optimum {_f(opt, 1)}, das Verfahren beweist etwas Falsches. Ein Schnitt muss für **jeden** Entwurf gelten.")

frames = res.frames
opt_open = {g for g in range(design.G) if a.opt.y[g] > 0.5}
owner = (params, opts)
if st.session_state.get("bnd_owner") != owner:
    st.session_state["bnd_frame"] = len(frames)
    st.session_state["bnd_owner"] = owner
step_col, play_col = st.columns([5, 2])
with step_col:
    if len(frames) > 1:
        it_no = st.slider("Iteration", 1, len(frames), key="bnd_frame", help="Jede Iteration: der Master schlägt einen Entwurf vor, das Teilproblem rechnet den Fluss (oder findet die Fehlmenge), ein Schnitt kommt hinzu. Die letzte Iteration ist das Ende.")
    else:
        it_no = 1
        st.caption("In diesem Netz gibt es nur eine Iteration.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _caption(i):
    f = frames[i]
    n_open = int((f.y > 0.5).sum())
    text = f"**Iteration {f.iteration}:** der Master öffnet {n_open} von {design.G} Kanten (Fixkosten {_n(f.fixed_cost)}, θ = {_f(f.theta, 1)}). "
    if f.kind == "feas":
        text += f"Das Teilproblem kann {_f(f.shortfall, 1)} Einheiten nicht liefern - **Zulässigkeitsschnitt**. "
    else:
        text += f"Der Fluss kostet {_f(f.flow_cost, 1)}, der Entwurf zusammen {_f(f.fixed_cost + f.flow_cost, 1)} - **Optimalitätsschnitt**. "
    text += f"Untere Schranke {_f(f.lb, 1)}" + (f", obere {_f(f.ub, 1)}." if f.ub != float("inf") else ", noch keine obere (noch kein zulässiger Entwurf).")
    return text


def _render(i):
    with view_slot.container():
        f = frames[i]
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Iteration {f.iteration}:** der Entwurf des Masters, der Fluss des Teilproblems und die Preise im Schnitt")
        c2.markdown("**Schranken je Iteration**")
        c1.plotly_chart(build_master_map(design, f, opt_open), width="stretch", key=f"master_map_{i}")
        c2.plotly_chart(build_bounds(frames, opt, i), width="stretch", key=f"bounds_chart_{i}")
        st.caption(_caption(i))
        if f.cut is not None:
            st.markdown("Der Schnitt dieser Iteration in Worten: " + ev.cut_words(design, f.cut))


if auto_play:
    for i in range(len(frames)):
        _render(i)
        time.sleep(min(0.6, 6.0 / max(len(frames), 1)))
else:
    _render(it_no - 1)

st.caption("Links: blau = vom Master geöffnet, grau = geschlossen, orange = Fluss des Teilproblems (Beschriftung Fluss/Kapazität), violette Unterlage = Kante mit Kapazitätspreis im Schnitt (Dicke ~ Preis), rote Unterlage = im ganzzahligen Optimum offen. "
           "Rechts: die untere Schranke (der Master) steigt, die obere (bester Entwurf mit Fluss) fällt; Kreuz = Zulässigkeitsschnitt, Punkt = Optimalitätsschnitt, rot das Optimum.")

st.markdown("---")

# --- Verteilung ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📊 Nicht nur dieses eine Netz")
if is_fixed:
    st.info("Festes Netz: es gibt nur diese eine Ziehung. Für die Verteilung über viele Netze ein zufälliges Netz wählen.")
else:
    kind = "Streckennetze" if net_key == "grid" else "Distributionsnetze"
    slow = " Ohne Start (reines Benders) dauert das mehrere Minuten." if start == "none" else ""
    st.caption(f"Iterationen über {len(C.QUALITY_SEEDS)} feste {kind} mit den Einstellungen oben (Güter {K}, Fixkosten {fix}), getrennt vom Seed.{slow}")
    if st.button(f"Verteilung über {len(C.QUALITY_SEEDS)} Netze rechnen", key="dist_start"):
        st.session_state["dist_on"] = True
    if st.session_state.get("dist_on"):
        with st.spinner("Rechne die festen Netze..."):
            dist = ev.distribution(params, opts)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Iterationen (Median)", _n(dist["iterations_median"]), delta=f"Mittel {_n(dist['iterations_mean'])}", delta_color="off", help="Bis untere und obere Schranke zusammenfallen (oder bis zur Grenze).")
        p2.metric("Zulässigkeitsschnitte", _pct(dist["feas_share_mean"]), delta="Anteil der Iterationen", delta_color="off", help="Mittel über die Netze.")
        p3.metric("Konvergiert", _pct(dist["converged_share"]), delta=f"{dist['n_feasible']} lieferbare Netze", delta_color="off", help="Anteil der Netze, in denen die Schranken innerhalb der Grenze zusammenfallen.")
        p4.metric("Sekunden", _sec(dist["seconds_mean"]), delta=f"HiGHS {_sec(dist['highs_mean'])} s", delta_color="off", help="Mittel über die Netze, nur zur Information.")
        st.plotly_chart(build_dist(dist, dat["total"] if control == "normal" else None), width="stretch", key="dist_chart")
        st.caption(f"Über {dist['n_feasible']} lieferbare Netze ({dist['n_infeasible']} nicht lieferbar) liegt der Median bei {_n(dist['iterations_median'])} Iterationen; Raute = die Ziehung oben.")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Was jede Zutat leistet")
st.caption("Sechs Varianten auf festen Netzen (3 Werke, 3 Verteilzentren, 6 Filialen, 8 Netze mit den Einstellungen oben): das reine Verfahren, das starke Teilproblem, Pareto-Schnitte, LP-Wurzelschnitte und die Cut-Set-Ungleichungen aus dem Vorgänger im Master.")
if is_fixed:
    st.info("Für dieses Experiment ein zufälliges Netz wählen.")
else:
    if st.button("Varianten durchrechnen (etwa eine halbe Minute)", key="variants_start"):
        st.session_state["variants_on"] = True
    if st.session_state.get("variants_on"):
        with st.spinner("Rechne..."):
            vt = ev.variant_table(params)
        st.plotly_chart(build_variants(vt), width="stretch", key="variants_chart")
        st.table({"Variante": [r["name"] for r in vt["rows"]], "Iterationen (Median)": [_n(r["iterations"]) for r in vt["rows"]], "Zulässigkeitsschnitte": [_pct(r["feas_share"]) for r in vt["rows"]],
                  "konvergiert": [f"{r['converged']} von {r['n']}" for r in vt["rows"]], "Sekunden": [_sec(r["seconds"]) for r in vt["rows"]]})
        st.caption(f"{vt['n']} lieferbare Netze mit im Mittel {vt['groups']:.0f} Entwurfskanten; HiGHS löst dasselbe in {_sec(vt['highs'])} Sekunden. "
                   "Das reine Verfahren verbringt die meisten Iterationen mit Zulässigkeitsschnitten: der Master weiß nichts darüber, welche Entwürfe überhaupt liefern. Starkes Teilproblem und Pareto-Schnitte ändern daran wenig; die Cut-Set-Ungleichungen (Vorgängerstück) räumen es auf.")

st.subheader("🔬 Benders gegen das direkte MIP")
st.caption("Benders (starkes Teilproblem, Pareto-Schnitte, Cut-Set im Master) gegen HiGHS über wachsende Netze, 6 feste Netze je Größe (die lieferbaren zählen), Grenze 100 Iterationen.")
if is_fixed:
    st.info("Für dieses Experiment ein zufälliges Netz wählen.")
else:
    if st.button("Größen durchrechnen (etwa eine Minute)", key="sizes_start"):
        st.session_state["sizes_on"] = True
    if st.session_state.get("sizes_on"):
        with st.spinner("Rechne..."):
            sz = ev.size_table(params)
        st.plotly_chart(build_sizes(sz), width="stretch", key="sizes_chart")
        st.table({"Werke/Verteilzentren/Filialen": [f"{r['size'][0]}/{r['size'][1]}/{r['size'][2]}" for r in sz], "Entwurfskanten": [_f(r["groups"], 0) for r in sz], "Iterationen": [_n(r["iterations"]) for r in sz],
                  "konvergiert": [f"{r['converged']} von {r['n']}" for r in sz], "Sekunden Benders": [_sec(r["seconds"]) for r in sz], "Sekunden HiGHS": [_sec(r["highs"]) for r in sz]})
        st.caption("HiGHS löst das ganze MIP mit eigener Vorverarbeitung und eigenen Schnitten in Bruchteilen einer Sekunde; Benders braucht bei wachsender Zahl der Entwurfskanten mehr Iterationen, und jede Iteration löst ein ganzzahliges Master-Problem, das mitwächst.")

st.subheader("🔬 Negativkontrollen und die LP-Schranke des Masters")
st.caption("Die Negativkontrollen auf dem gezeigten Netz - und die Theorie-Aussage, dass die LP-Relaxation des Masters mit allen Schnitten genau die schwache LP-Schranke des Vorgängerstücks ist.")
if st.button("Kontrollen durchrechnen", key="controls_start"):
    st.session_state["controls_on"] = True
if st.session_state.get("controls_on"):
    with st.spinner("Rechne..."):
        ct = ev.control_table(params)
        lt = ev.lp_master_table(params) if not is_fixed else []
    labels_ = {"normal": "wie beschrieben", "nofeas": "ohne Zulässigkeitsschnitte", "wrong": "ungültige Schnitte"}
    st.table({"Variante": [labels_[r["control"]] for r in ct], "Iterationen": [_n(r["iterations"]) for r in ct],
              "Ergebnis": [("Optimum bewiesen" if r["converged"] and not r["crossed"] else "hängt" if r["stuck"] else "beweist Falsches (LB > Optimum)" if r["crossed"] else "Grenze erreicht") for r in ct]})
    if lt:
        st.table({"Seed": [str(r["seed"]) for r in lt], "LP-Master": [_pct(r["master"], 1) for r in lt], "schwache LP-Schranke": [_pct(r["weak"], 1) for r in lt], "gleich": ["ja" if r["same"] else "nein" for r in lt]})
        st.caption("Anteil am Optimum. Die Schnitte beschreiben v(y) vollständig, der LP-Master ist also die schwache Relaxation - der Fortschritt des ganzzahligen Verfahrens kommt allein aus den ganzzahligen Master-Läufen.")
    else:
        st.caption("Die Tabelle der LP-Schranke gibt es nur für zufällige Netze.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist - und wer setzt an |
|---|---|
| **Der Fluss ist der leichte Teil** | Benders lohnt, wenn das Teilproblem groß, aber das Entwurfsproblem klein ist (viele Szenarien, stochastische Programme). Hier ist es umgekehrt: 25 bis 70 Entwurfskanten, ein kleiner Fluss - das direkte MIP gewinnt. |
| **Der Master bleibt NP-schwer** | Jede Iteration löst ein ganzzahliges Programm; es wächst mit jedem Schnitt. Die Schnitte sind schwach (die LP-Schranke des Masters ist die schwache Relaxation). |
| **Zulässigkeit nur über Schnitte** | Ohne Vorwissen (Cut-Set) lernt der Master Erreichbarkeit Iteration für Iteration; das sind die meisten Iterationen im reinen Verfahren. |
| **Exakt bis zum Ende** | Bei Zeitgrenze bleibt eine Lücke; für große Netze braucht es Heuristiken. **Ansatzpunkt:** Slope Scaling (das Folgestück). |
| **Statisches Netz** | Fixkosten gelten je Kante und Tag; Fahrpläne brauchen ein Zeit-Raum-Netz (leercontainer-demo). |
"""
)
st.caption("Die Netzwerkfluss-Linie ist als Ganzes geplant: Edmonds-Karp, Dinic, Push-Relabel, Successive Shortest Paths, Cycle-Canceling, Cost Scaling, Mehrgüterfluss, Column Generation, Garg-Könemann, Fixkosten-Netzwerkdesign, Benders-Zerlegung (dieses Stück) und Slope Scaling.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Entwurf.** $\min\ \sum_g f_g y_g+\sum_{k,e}c^k_e x^k_e$ u.d.N. Flusserhaltung je Gut, $\sum_k x^k_e\le u_e y_{g(e)}$ (stark zusätzlich $x^k_e\le\min(u_e,d_k)y_{g(e)}$), $y\in\{0,1\}^G$, volle Nachfrage.

**Teilproblem und Dual.** Für festes $\bar y$: $v(\bar y)=\min\{c\cdot x: A_{eq}x=0,\ A_x x\le b-A_y\bar y,\ lo\le x\le hi\}$. Das Dual ist $\max\ b_{eq}\lambda-(b-A_y\bar y)\mu+lo\,\rho-hi\,\omega$ u.d.N. $A_{eq}^T\lambda-A_x^T\mu+\rho-\omega=c$, $\mu,\rho,\omega\ge0$. Der Dualzielwert als Funktion von $y$ ist $D(y)=\text{const}+(A_y^T\mu)\cdot y$, für **jede** dual zulässige Lösung eine untere Schranke von $v(y)$ (schwache Dualität) - das ist der Schnitt $\theta\ge D(y)$. Ist das Dual im Punkt $\bar y$ optimal, gilt $D(\bar y)=v(\bar y)$.

**Zulässigkeit.** Phase 1: $\min -\text{Lieferung}$; ist $D_1(y)>-d_{ges}$ am Entwurf, verlangt der Schnitt $D_1(y)\le-d_{ges}$.

**Pareto (Magnanti-Wong).** $\max_{(\lambda,\mu,\rho,\omega)} D(y^0)$ u.d.N. dual zulässig und $D(\bar y)\ge v(\bar y)$; Kernpunkt $y^0$ = Mitte zwischen der LP-Lösung und dem Entwurf mit allen Kanten offen.

**Master.** $\min f\cdot y+\theta$, $\theta\ge D_i(y)$ für alle Optimalitätsschnitte, $D^{feas}_j(y)\le 0$, optional die Cut-Set-Ungleichungen $\sum_e\lceil\min(u_e,R)/\delta\rceil y_e\ge\lceil R/\delta\rceil$. Untere Schranke = Master-Wert, obere = $\min(f\cdot\bar y+v(\bar y))$; Ende bei Gleichheit.

Implementiert in `bnd_benders.py` (Teilproblem, Phase 1, Dual, Pareto, Master, Schleife), `bnd_cuts.py` (Cut-Set-Ungleichungen aus dem Vorgänger), `bnd_formulation.py`, `bnd_model.py`, `bnd_scenario.py` (Netze), `bnd_evaluation.py`, `bnd_visualization.py`; das Vergleichs-MIP löst HiGHS über `scipy`.
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
