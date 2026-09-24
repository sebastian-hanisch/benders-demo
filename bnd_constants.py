"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Benders-Zerlegung: Entwurf im Master, Fluss im Teilproblem"."""

# --- Regler Distributionsnetz -----------------------------------------------------------------------------------------------------
P_MIN, P_MAX, DEFAULT_P = 2, 6, 3            # Werke
D_MIN, D_MAX, DEFAULT_D = 2, 6, 3            # Verteilzentren
S_MIN, S_MAX, DEFAULT_S = 3, 12, 8           # Filialen
DENSITY_MIN, DENSITY_MAX, DEFAULT_DENSITY = 20, 100, 60   # Anteil vorhandener Lanes in ganzen Prozent, Schritt 10
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0, 100, 50       # Streuung der Lane-Breiten in ganzen Prozent, Schritt 25
LOAD_MIN, LOAD_MAX, DEFAULT_LOAD = 30, 90, 50             # Gesamtnachfrage in Prozent der Werkskapazität, Schritt 10
FIX_MIN, FIX_MAX, DEFAULT_FIX = 5, 150, 30                # Fixkosten-Basis je Lane (mal 1 bis 3; Verteilzentrum doppelt), Schritt 5
DEFAULT_SEED = 155
SEED_MAX = 2_000_000_000

# --- Regler Streckennetz ------------------------------------------------------------------------------------------------------------
GW_MIN, GW_MAX, DEFAULT_GW = 3, 6, 4                      # Breite des Gitters
GH_MIN, GH_MAX, DEFAULT_GH = 3, 5, 3                      # Höhe des Gitters
GDENSITY_MIN, GDENSITY_MAX, DEFAULT_GDENSITY = 40, 100, 60   # Anteil der Gitterkanten (über den Spannbaum hinaus), Schritt 10
GCAP_MIN, GCAP_MAX, DEFAULT_GCAP = 1, 4, 4                # größte Kapazität je Kante und Richtung
GDEM_MIN, GDEM_MAX, DEFAULT_GDEM = 1, 5, 2                # größte Menge je Gut
DEFAULT_GSEED = 6

K_MIN, K_MAX, DEFAULT_K = 1, 5, 3            # Zahl der Güter

NETS = {
    "random": "Distributionsnetz (wie im Vorgänger)",
    "grid": "Streckennetz (Gitter mit Start-Ziel-Aufträgen)",
    "bigm": "Big-M-Falle: eine Menge von 1, eine Kante mit Kapazität 10",
    "rounding": "Rundungs-Falle: 3 Einheiten über zwei Kanten der Kapazität 2",
    "bundle": "Bündelung: zwei Güter teilen eine Stammstrecke",
}
DEFAULT_NET = "random"
FIXED_NETS = ("bigm", "rounding", "bundle")

SUBS = {"weak": "schwach: Σ x ≤ u · y", "strong": "stark: zusätzlich x ≤ min(u, Nachfrage) · y je Gut"}
DEFAULT_SUB = "strong"
CUTS = {"normal": "Schnitte aus den Dualpreisen des Solvers", "pareto": "Pareto-optimale Schnitte (Magnanti–Wong)"}
DEFAULT_CUT = "pareto"
STARTS = {"none": "kein Start (Master beginnt leer)", "lp": "LP-Wurzelschnitte (erst der LP-Master)", "cutset": "Cut-Set-Ungleichungen im Master (aus dem Vorgänger)"}
DEFAULT_START = "cutset"
ITERS = (50, 100, 200)
DEFAULT_ITER = 200
CONTROLS = {"normal": "wie beschrieben", "nofeas": "Negativkontrolle: ohne Zulässigkeitsschnitte", "wrong": "Negativkontrolle: ungültige Schnitte (nach oben verschoben)"}
DEFAULT_CONTROL = "normal"
LEVELS = {"weak": 0, "strong": 1}

# --- feste Seed-Mengen (dieselben wie in den Vorgänger-Demos; unabhängig vom Nutzer-Seed) ------------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
QUALITY_SEEDS = DIST_SEEDS[:20]                     # Verteilung der Iterationen (Standardvariante, etwa eine halbe Minute)
VARIANT_SEEDS = DIST_SEEDS[:8]                      # Varianten-Matrix (Sekunden je Netz)
SIZE_SEEDS = DIST_SEEDS[:6]
SIZES = ((2, 2, 4), (3, 3, 8), (4, 4, 12), (5, 5, 16))     # Werke, Verteilzentren, Filialen
COLORS = {"open": "#1f77b4", "closed": "rgba(150,150,150,0.45)", "flow": "#ff7f0e", "price": "#9467bd", "opt": "#d62728", "node": "#111111", "feas": "#d62728", "optcut": "#1f77b4"}

# --- Presets -----------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", k=DEFAULT_K, p=DEFAULT_P, d=DEFAULT_D, s=DEFAULT_S, density=DEFAULT_DENSITY, spread=DEFAULT_SPREAD, load=DEFAULT_LOAD, fix=DEFAULT_FIX, seed=DEFAULT_SEED,
             gw=DEFAULT_GW, gh=DEFAULT_GH, gdensity=DEFAULT_GDENSITY, gcap=DEFAULT_GCAP, gdem=DEFAULT_GDEM, gseed=DEFAULT_GSEED,
             sub=DEFAULT_SUB, cut=DEFAULT_CUT, start=DEFAULT_START, iters=DEFAULT_ITER, control=DEFAULT_CONTROL)
PRESETS = {
    "🚚 Zufallsnetz": {**_BASE},
    "🗺️ Streckennetz": {**_BASE, "net": "grid", "k": 3, "fix": 10},
    "💸 Big-M-Falle": {**_BASE, "net": "bigm", "k": 1},
    "🔁 Rundungs-Falle": {**_BASE, "net": "rounding", "k": 1},
    "📦 Bündelung": {**_BASE, "net": "bundle", "k": 2},
    "🧱 Reines Benders": {**_BASE, "s": 6, "seed": 7, "sub": "weak", "cut": "normal", "start": "none"},
    "🚫 Ohne Zulässigkeitsschnitte": {**_BASE, "net": "bigm", "k": 1, "control": "nofeas"},
    "❌ Ungültige Schnitte": {**_BASE, "net": "rounding", "k": 1, "control": "wrong"},
}
PRESET_HELP = {
    "🚚 Zufallsnetz": "Das Netz der Vorgänger (Seed 155, drei Güter, Fixkosten 30) mit starkem Teilproblem, Pareto-Schnitten und den Cut-Set-Ungleichungen im Master: Optimum 1 784 nach 40 Iterationen (19 Optimalitäts-, 21 Zulässigkeitsschnitte, dazu 26 Cut-Set-Ungleichungen vorab). HiGHS löst dasselbe MIP in Hundertsteln einer Sekunde (Benders braucht mindestens das Fünffache).",
    "🗺️ Streckennetz": "Gitter 4 × 3 mit drei Gütern (Seed 6, Fixkosten 10): Optimum 245 nach 27 Iterationen, 25 davon Zulässigkeitsschnitte - der Master lernt zuerst, welche Strecken überhaupt liefern, und erst dann, was sie kosten.",
    "💸 Big-M-Falle": "Eine Einheit, zwei Wege: 3 Iterationen (1 Zulässigkeits-, 2 Optimalitätsschnitte), Optimum 15. Der billigste Entwurf öffnet nichts und kann nicht liefern - das lehrt der erste Zulässigkeitsschnitt; die breite Kante wird als zu teuer verworfen.",
    "🔁 Rundungs-Falle": "Drei Einheiten über zwei gleiche Kanten: 2 Iterationen bis zum Optimum 23. Die Cut-Set-Ungleichung y₁ + y₂ ≥ 2 aus dem Vorgängerstück sagt dem Master vorab, dass beide Kanten offen sein müssen.",
    "📦 Bündelung": "Zwei Güter, direkt oder gebündelt über einen Umschlagpunkt: 8 Iterationen (2 Optimalitäts-, 6 Zulässigkeitsschnitte), Optimum 11 - der Master probiert Entwürfe, bis er die Kombination findet, die beide Güter liefert.",
    "🧱 Reines Benders": "Ohne Start, schwaches Teilproblem, Schnitte aus den Dualpreisen des Solvers, kleineres Netz (3/3/6, Seed 7): 71 Iterationen, davon 63 Zulässigkeitsschnitte (89 %), Optimum 1 348. HiGHS löst dasselbe in Hundertsteln einer Sekunde, Benders braucht mehr als das Zehnfache.",
    "🚫 Ohne Zulässigkeitsschnitte": "Negativkontrolle: im Big-M-Netz ohne Zulässigkeitsschnitte schlägt der Master nach 2 Iterationen denselben unbedienbaren Entwurf wieder vor (untere Schranke 0) - er weiß nicht, dass er nicht liefert, und kommt nicht weiter.",
    "❌ Ungültige Schnitte": "Negativkontrolle: in der Rundungs-Falle um 5 % + 1 nach oben verschobene Optimalitätsschnitte. Schon nach 2 Iterationen liegt die untere Schranke bei 24,15 über dem Optimum 23 - das Verfahren beweist Falsches.",
}
