"""Entwurfsmodell: Mehrgüterfluss mit Fixkosten je Kante (Fixed-Charge-Multicommodity-Network-Design).

Grundlage ist das Mehrgütermodell der Vorgänger (`Mcf`: Quelle S, Werke, Verteilzentren als Eingang -> Ausgang gespalten, Filialen, Senke T; K Güter teilen sich die Kapazität der Kanten).
Neu: eine **Entwurfsentscheidung** y_g in {0, 1} je Gruppe von Kanten (Lane, Verteilzentrum-Durchsatz; im Streckennetz beide Richtungen einer Strecke). Eine offene Gruppe kostet ihre Fixkosten und
erlaubt bis zur Kapazität Fluss; eine geschlossene erlaubt keinen. Die volle Nachfrage muss geliefert werden - der Entwurf entscheidet nur, *womit*.
Alles ganzzahlig, eigene Zufallsströme (SplitMix64).
"""

from dataclasses import dataclass

import bnd_scenario as sc
from bnd_scenario import SplitMix64

BIG = 10 ** 6                       # "unbeschränkt" für Obergrenzen
GOODS = ("Frische", "Trocken", "Kühl", "Getränke", "Tiefkühl")
FACTORS = (2, 1, 3, 1, 4)           # Kostenfaktor auf den Lanes je Gut
GOOD_COLORS = ("#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf")
M_REWARD = 1000                     # größer als jeder Weg (höchste Wegkosten < 200)


@dataclass(frozen=True)
class Mcf:
    net: sc.Net
    names: tuple         # Gutnamen
    factors: tuple       # Kostenfaktor je Gut auf Lanes
    ub: tuple            # ub[k][e]: Obergrenze je Gut und Kante (BIG = unbeschränkt)
    joint: tuple         # joint[e]: gemeinsame Kapazität (Summe über die Güter)
    reward: tuple        # reward[e]: Kante in die Senke (Belohnung M je Einheit)
    M: int
    layout: str = "layered"   # "layered" (Distributionsnetz), "pairs" (Frachtnetz) oder "grid" (Streckennetz: S/T-Kanten werden nicht gezeichnet)

    @property
    def K(self):
        return len(self.names)

    @property
    def m(self):
        return self.net.m

    def cost(self, k, e):
        """Kosten je Einheit von Gut k auf Kante e (ohne Belohnung)."""
        _, _, _, c, kind = self.net.arcs[e]
        return self.factors[k] * c if kind in (sc.K_LANE_IN, sc.K_LANE_OUT) else c

    def demand(self, k):
        """Gesamtnachfrage von Gut k (Summe der Obergrenzen der Senkenkanten)."""
        return sum(self.ub[k][e] for e in range(self.m) if self.reward[e])

    def total_demand(self):
        return sum(self.demand(k) for k in range(self.K))


def _split(total, weights):
    """Ganzzahlige Aufteilung von `total` im Verhältnis der Gewichte; der Rest geht an das schwerste Gewicht."""
    w = sum(weights)
    parts = [total * x // w for x in weights]
    parts[max(range(len(weights)), key=lambda i: weights[i])] += total - sum(parts)
    return parts


def generate_mcf(n_plants, n_dcs, n_stores, density, spread, load, seed, K):
    """Zufälliges Distributionsnetz mit K Gütern. Das Einzelgutnetz (`sc.generate`) liefert Kapazitäten und Gesamtnachfrage; darauf werden Güter verteilt:
    jedes Werk stellt ein Gut mit 70 % Wahrscheinlichkeit her (jedes Gut hat mindestens ein Werk), die Nachfrage einer Filiale wird im Verhältnis zufälliger Gewichte auf die Güter verteilt."""
    net = sc.generate(n_plants, n_dcs, n_stores, density, spread, load, seed)
    rng = SplitMix64(seed ^ 0x6D63665F)
    supply = [i for i, a in enumerate(net.arcs) if a[4] == sc.K_SUPPLY]
    demand_arcs = [i for i, a in enumerate(net.arcs) if a[4] == sc.K_DEMAND]
    ub = [[BIG] * net.m for _ in range(K)]
    for k in range(K):
        allowed = [rng.below(100) < 70 for _ in supply]
        if not any(allowed):
            allowed[rng.below(len(supply))] = True
        for j, e in enumerate(supply):
            ub[k][e] = net.arcs[e][2] if allowed[j] else 0
    for e in demand_arcs:
        parts = _split(net.arcs[e][2], [1 + rng.below(10) for _ in range(K)])
        for k in range(K):
            ub[k][e] = parts[k]
    joint = tuple(a[4] != sc.K_DEMAND for a in net.arcs)
    reward = tuple(a[1] == net.t for a in net.arcs)
    return Mcf(net, GOODS[:K], FACTORS[:K], tuple(tuple(r) for r in ub), joint, reward, M_REWARD)








def _terminals(net_n, pairs, K, internal_arcs, pos, names, labels, layout):
    """Gemeinsamer Aufbau der Frachtnetze: Angebotskante S -> Start und Nachfragekante Ziel -> T je Gut (nur für das eigene Gut offen), alle inneren Kanten gemeinsam."""
    arcs = list(internal_arcs)
    internal = len(arcs)
    for a, b, dem in pairs:
        arcs.append((0, 2 + a, dem, 0, sc.K_SUPPLY))
    for a, b, dem in pairs:
        arcs.append((2 + b, 1, dem, 0, sc.K_DEMAND))
    net = sc.Net(tuple(names), tuple(labels), tuple(pos), tuple(arcs), 0, 1, False)
    ub = [[BIG] * len(arcs) for _ in range(K)]
    for k, (a, b, dem) in enumerate(pairs):
        for j in range(K):
            ub[k][internal + j] = dem if j == k else 0
            ub[k][internal + K + j] = dem if j == k else 0
    joint = tuple(i < internal for i in range(len(arcs)))
    reward = tuple(a[1] == 1 for a in arcs)
    return Mcf(net, GOODS[:K], (1,) * K, tuple(tuple(r) for r in ub), joint, reward, M_REWARD, layout)


def generate_grid(width, height, density, cap_max, dem_max, K, seed):
    """Streckennetz: ein Gitter width x height. Ein zufälliger Spannbaum hält das Netz zusammenhängend, jede weitere Gitterkante gibt es mit `density` Prozent. Jede Kante trägt in beide Richtungen
    dieselbe Kapazität 1..cap_max und dieselben Kosten 1..9 (Länge), gemeinsam für alle Güter. Gut k fährt von seinem Start zu seinem Ziel (verschiedene Knoten, Menge 1..dem_max).
    Die Zufallszahlen werden je Gitterkante in fester Reihenfolge gezogen, egal ob sie existiert - so ändert `density` nur, welche Kanten es gibt."""
    rng = SplitMix64(seed ^ 0x67726964)
    n = width * height
    node = lambda i, j: 2 + j * width + i
    edges = [(node(i, j), node(i + 1, j)) for j in range(height) for i in range(width - 1)] + [(node(i, j), node(i, j + 1)) for j in range(height - 1) for i in range(width)]
    draws = [(rng.below(100), 1 + rng.below(cap_max), 1 + rng.below(9)) for _ in edges]
    tree = SplitMix64(seed ^ 0x74726565)
    order = list(range(len(edges)))
    for i in range(len(order) - 1, 0, -1):                      # Fisher-Yates
        j = tree.below(i + 1)
        order[i], order[j] = order[j], order[i]
    parent = list(range(n + 2))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    keep = set()
    for e in order:
        a, b = find(edges[e][0]), find(edges[e][1])
        if a != b:
            parent[a] = b
            keep.add(e)
    arcs = []
    for e, (u, v) in enumerate(edges):
        r, cap, cost = draws[e]
        if e in keep or r < density:
            arcs.append((u, v, cap, cost, sc.K_OTHER))
            arcs.append((v, u, cap, cost, sc.K_OTHER))
    pairs = []
    for _ in range(K):
        a = rng.below(n)
        b = (a + 1 + rng.below(n - 1)) % n
        pairs.append((a, b, 1 + rng.below(dem_max)))
    names = ["Quelle S", "Senke T"] + [f"Knoten ({i + 1}, {j + 1})" for j in range(height) for i in range(width)]
    labels = ["S", "T"] + [""] * n
    xs = [8 + 84 * i // max(1, width - 1) for i in range(width)]
    ys = [88 - 72 * j // max(1, height - 1) for j in range(height)]
    pos = [(50, 100), (50, 0)] + [(xs[i], ys[j]) for j in range(height) for i in range(width)]
    return _terminals(n, pairs, K, arcs, pos, names, labels, "grid")


# --- Entwurf: Fixkosten und Entwurfsgruppen ----------------------------------------------------------------------------------------------

DESIGN_KINDS = (sc.K_LANE_IN, sc.K_THROUGHPUT, sc.K_LANE_OUT, sc.K_OTHER)     # Kanten, über die der Entwurf entscheidet


@dataclass(frozen=True)
class Design:
    mcf: Mcf
    group: tuple          # group[e]: Nummer der Entwurfsentscheidung, zu der Kante e gehört (-1: Kante gibt es immer, z. B. Werks- und Nachfragekanten)
    fixed: tuple          # fixed[g]: Fixkosten der Gruppe g (ganzzahlig)

    @property
    def net(self):
        return self.mcf.net

    @property
    def K(self):
        return self.mcf.K

    @property
    def m(self):
        return self.mcf.m

    @property
    def G(self):
        return len(self.fixed)

    def arcs_of(self, g):
        return [e for e in range(self.m) if self.group[e] == g]

    def cap(self, e):
        return self.net.arcs[e][2]

    def demand_arcs(self):
        return [e for e in range(self.m) if self.mcf.reward[e]]

    def all_open_cost_floor(self):
        """Fixkosten aller Gruppen zusammen (obere Grenze für jeden Entwurf)."""
        return sum(self.fixed)


def _designed(mcf, fixed_of):
    """Entwurfsgruppen bilden: jede Kante der Art `DESIGN_KINDS` eine eigene Gruppe (Streckennetz: die beiden Richtungen einer Strecke gemeinsam). `fixed_of(g_index, arc)` liefert die Fixkosten."""
    group, fixed = [], []
    pair_of = {}
    for e, arc in enumerate(mcf.net.arcs):
        if arc[4] not in DESIGN_KINDS:
            group.append(-1)
            continue
        key = (min(arc[0], arc[1]), max(arc[0], arc[1])) if mcf.layout == "grid" else e
        if key not in pair_of:
            pair_of[key] = len(fixed)
            fixed.append(fixed_of(len(fixed), arc))
        group.append(pair_of[key])
    return Design(mcf, tuple(group), tuple(fixed))


def _every_plant_makes_every_good(mcf):
    """Im Entwurf stellt jedes Werk jedes Gut her (der Entwurf entscheidet über Lanes, nicht über Sortimente): sonst ließe sich die Nachfrage eines Guts oft gar nicht decken."""
    net = mcf.net
    ub = [list(r) for r in mcf.ub]
    for e, arc in enumerate(net.arcs):
        if arc[4] == sc.K_SUPPLY:
            for k in range(mcf.K):
                ub[k][e] = arc[2]
    return Mcf(net, mcf.names, mcf.factors, tuple(tuple(r) for r in ub), mcf.joint, mcf.reward, mcf.M, mcf.layout)


def generate_design(n_plants, n_dcs, n_stores, density, spread, load, seed, K, fix):
    """Distributionsnetz mit Fixkosten: eine Lane kostet `fix` mal 1..3, ein Verteilzentrum (Durchsatzkante) das Doppelte (Fixkosten fallen an, sobald es überhaupt genutzt wird).
    Die Zufallszahlen kommen aus einem eigenen Strom je Entwurfskante in fester Reihenfolge."""
    mcf = generate_mcf(n_plants, n_dcs, n_stores, density, spread, load, seed, K)
    mcf = _every_plant_makes_every_good(mcf)
    rng = SplitMix64(seed ^ 0x66697865)
    return _designed(mcf, lambda g, arc: fix * (1 + rng.below(3)) * (2 if arc[4] == sc.K_THROUGHPUT else 1))


def generate_design_grid(width, height, density, cap_max, dem_max, K, seed, fix):
    """Streckennetz mit Fixkosten: jede Strecke (beide Richtungen zusammen) kostet `fix` mal 1..3."""
    mcf = generate_grid(width, height, density, cap_max, dem_max, K, seed)
    rng = SplitMix64(seed ^ 0x66697867)
    return _designed(mcf, lambda g, arc: fix * (1 + rng.below(3)))


def _teaching(names, labels, pos, arcs, fixed, pairs, layout="pairs"):
    """Lehrnetz von Hand: `arcs` = (von, nach, Kapazität, Stückkosten) auf den Knoten 2.. (Knoten 0 = S, 1 = T), `fixed` je Kante, `pairs` = (Start, Ziel, Menge) je Gut."""
    internal = [(u, v, c, w, sc.K_OTHER) for u, v, c, w in arcs]
    mcf = _terminals(len(names) - 2, pairs, len(pairs), internal, pos, names, labels, layout)
    fixed = list(fixed)
    return Design(mcf, tuple(range(len(arcs))) + (-1,) * (2 * len(pairs)), tuple(fixed))


def bigm_net():
    """Big-M-Falle: eine Menge von 1 Einheit, zwei Wege von A nach B. Die breite Kante (Kapazität 10, Stückkosten 1, Fixkosten 30) sieht im LP fast kostenlos aus: y = 0,1 genügt für die eine Einheit
    (Kosten 1 + 3 = 4). Die schmale (Kapazität 1, Stückkosten 3, Fixkosten 12) ist ganzzahlig billiger: 15 gegen 31. Die starke Kopplung x <= min(u, d)·y setzt y = 1 und schließt die Lücke."""
    return _teaching(["Quelle S", "Senke T", "A", "B"], ["S", "T", "A", "B"], [(50, 96), (50, 4), (14, 50), (86, 50)],
                     [(2, 3, 10, 1), (2, 3, 1, 3)], [30, 12], [(0, 1, 1)])


def rounding_net():
    """Rundungs-Falle: 3 Einheiten von A nach B über zwei gleiche Kanten (Kapazität 2, Fixkosten 10, Stückkosten 1). Das LP öffnet jede Kante zu 0,75 (Kosten 3 + 15 = 18); ganzzahlig braucht es
    beide (3 + 20 = 23). Die starke Kopplung hilft nicht (min(2, 3) = 2), erst die gerundete Schnittungleichung y1 + y2 >= 2 schließt die Lücke."""
    return _teaching(["Quelle S", "Senke T", "A", "B"], ["S", "T", "A", "B"], [(50, 96), (50, 4), (14, 50), (86, 50)],
                     [(2, 3, 2, 1), (2, 3, 2, 1)], [10, 10], [(0, 1, 3)])


def bundle_net():
    """Bündelung: zwei Güter (je 1 Einheit) von A bzw. B nach D. Direkt kostet jede Strecke 8 Fixkosten (Kapazität 2); über den Umschlagpunkt H teilen sich beide die Stammstrecke H -> D (Kapazität 4,
    Fixkosten 5) und zahlen nur 2 für den Zubringer: 9 statt 16 Fixkosten. Optimum 11 (beide über H); das schwache LP sieht nur 6,5 (Fixkosten anteilig), die starke Kopplung liefert schon 11."""
    names = ["Quelle S", "Senke T", "A", "B", "D", "H"]
    return _teaching(names, ["S", "T", "A", "B", "D", "H"], [(50, 96), (50, 4), (12, 78), (12, 22), (88, 50), (50, 50)],
                     [(2, 4, 2, 1), (3, 4, 2, 1), (2, 5, 2, 1), (3, 5, 2, 1), (5, 4, 4, 0)], [8, 8, 2, 2, 5], [(0, 2, 1), (1, 2, 1)])
