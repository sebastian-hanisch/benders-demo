"""Szenario: ein Distributionsnetz als Flussnetz (Werke -> Verteilzentren -> Filialen), Grundlage des Mehrgütermodells (`bnd_model.py`).

Alles ist ganzzahlig und läuft über einen eigenen Zufallsgenerator (SplitMix64 auf Python-Ints) statt über
`numpy.random`: numpy garantiert keine über Versionen stabilen Zufallsströme, die CI installiert aber wöchentlich die
neueste Version. So sind Voreinstellungen, Seeds und jede im Text genannte Zahl auf Windows und Linux dieselben.

Das Netz hat eine Super-Quelle S und eine Super-Senke T. Ein Verteilzentrum ist als Eingang -> Ausgang gesplittet; die
Verbindung dazwischen trägt den Durchsatz (Knotenkapazität als Kantenkapazität). Jede Kante hat eine Art (`kind`), damit
sich ein Schnitt später einer Stufe des Netzes zuordnen lässt.
"""

from dataclasses import dataclass

_MASK = (1 << 64) - 1
MAP_W = 100
LANE_IN_BASE, LANE_OUT_BASE = 25, 12   # mittlere Lane-Kapazität Werk -> DC (Sammellinien) bzw. DC -> Filiale (Verteilung)
LAYER_Y = {"S": 100, "plant": 82, "dc_in": 62, "dc_out": 44, "store": 24, "T": 6}

# Art einer Kante (Stufe des Distributionsnetzes)
K_SUPPLY, K_LANE_IN, K_THROUGHPUT, K_LANE_OUT, K_DEMAND, K_OTHER = range(6)
KIND_LABELS = {
    K_SUPPLY: "Werkskapazität",
    K_LANE_IN: "Lane Werk → DC",
    K_THROUGHPUT: "DC-Durchsatz",
    K_LANE_OUT: "Lane DC → Filiale",
    K_DEMAND: "Filialnachfrage",
    K_OTHER: "Kante",
}


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, n):
        """Ganzzahl in 0..n-1 (die Modulo-Verzerrung bei n <= 101 liegt um 1e-17)."""
        return self.next() % n


@dataclass(frozen=True)
class Net:
    names: tuple      # Anzeigename je Knoten (Hover)
    labels: tuple     # Kurzbeschriftung je Knoten (Karte)
    pos: tuple        # ((x, y), ...) je Knoten
    arcs: tuple       # ((u, v, Kapazität, Kosten je Einheit, Art), ...)
    s: int
    t: int
    logistic: bool    # True: Werke/DCs/Filialen; False: Lehrnetz mit frei benannten Knoten

    @property
    def n(self):
        return len(self.names)

    @property
    def m(self):
        return len(self.arcs)

    def total_capacity_out_of_s(self):
        return sum(c for u, _, c, _, _ in self.arcs if u == self.s)


def _row_x(count):
    """Gleichmäßige x-Positionen für `count` Knoten einer Schicht."""
    return [(2 * i + 1) * MAP_W // (2 * count) for i in range(count)]


def generate(n_plants, n_dcs, n_stores, density, spread, load, seed):
    """Zufälliges Distributionsnetz. `density`: Anteil vorhandener Lanes in ganzen Prozent; `spread`: 0 = alle Lanes einer Stufe gleich
    breit, 100 = Kapazität gleichverteilt 1..2x Basis; `load`: Gesamtnachfrage in Prozent der gesamten Werkskapazität.
    Die Zufallszahlen werden für jede mögliche Lane in fester Reihenfolge gezogen, egal ob sie existiert - so ändert
    ein anderer Wert für `density` nur, welche Lanes es gibt, nicht ihre Breite."""
    rng = SplitMix64(seed)
    P, D, S = n_plants, n_dcs, n_stores

    plant_cap = [20 + rng.below(21) for _ in range(P)]
    total_supply = sum(plant_cap)
    demand_raw = [5 + rng.below(11) for _ in range(S)]
    demand = [max(1, d * total_supply * load // (100 * sum(demand_raw))) for d in demand_raw]
    dc_cap = [max(1, total_supply * (30 + rng.below(31)) // 100) for _ in range(D)]

    def lane_cap(base):
        u = 1 + rng.below(2 * base)
        return max(1, (base * (100 - spread) + u * spread) // 100)

    def lane_row(count_from, count_to, base):
        rows = []
        for _ in range(count_from):
            row = []
            for _ in range(count_to):
                exists = rng.below(100) < density
                cap, cost = lane_cap(base), 1 + rng.below(9)
                row.append((exists, cap, cost))
            rows.append(row)
        return rows

    lanes_in = lane_row(P, D, LANE_IN_BASE)
    lanes_out = lane_row(D, S, LANE_OUT_BASE)
    # Kein Werk und keine Filiale ohne Anschluss: fehlt jede Lane, wird eine zufällige ergänzt (eigener Zufallsstrom, damit alle übrigen Werte
    # dieselben bleiben, egal ob und wie viele Lanes ergänzt werden müssen)
    fix = SplitMix64(seed ^ 0x5DEECE66D)
    for p in range(P):
        if not any(e for e, _, _ in lanes_in[p]):
            d = fix.below(D)
            lanes_in[p][d] = (True, lanes_in[p][d][1], lanes_in[p][d][2])
    for s in range(S):
        if not any(lanes_out[d][s][0] for d in range(D)):
            d = fix.below(D)
            lanes_out[d][s] = (True, lanes_out[d][s][1], lanes_out[d][s][2])
    plant_cost = [1 + rng.below(5) for _ in range(P)]
    dc_cost = [1 + rng.below(3) for _ in range(D)]

    # Knoten: 0 = S, 1 = T, dann Werke, DC-Eingänge, DC-Ausgänge, Filialen
    names, labels, pos = ["Quelle S", "Senke T"], ["S", "T"], [(50, LAYER_Y["S"]), (50, LAYER_Y["T"])]
    plant0 = 2
    dc_in0 = plant0 + P
    dc_out0 = dc_in0 + D
    store0 = dc_out0 + D
    for i, x in enumerate(_row_x(P)):
        names.append(f"Werk {i + 1}"); labels.append(f"W{i + 1}"); pos.append((x, LAYER_Y["plant"]))
    for i, x in enumerate(_row_x(D)):
        names.append(f"DC {i + 1} (Eingang)"); labels.append(f"D{i + 1}"); pos.append((x, LAYER_Y["dc_in"]))
    for i, x in enumerate(_row_x(D)):
        names.append(f"DC {i + 1} (Ausgang)"); labels.append(""); pos.append((x, LAYER_Y["dc_out"]))
    for i, x in enumerate(_row_x(S)):
        names.append(f"Filiale {i + 1}"); labels.append(f"F{i + 1}"); pos.append((x, LAYER_Y["store"]))

    arcs = []
    for p in range(P):
        arcs.append((0, plant0 + p, plant_cap[p], plant_cost[p], K_SUPPLY))
    for p in range(P):
        for d in range(D):
            exists, cap, cost = lanes_in[p][d]
            if exists:
                arcs.append((plant0 + p, dc_in0 + d, cap, cost, K_LANE_IN))
    for d in range(D):
        arcs.append((dc_in0 + d, dc_out0 + d, dc_cap[d], dc_cost[d], K_THROUGHPUT))
    for d in range(D):
        for s in range(S):
            exists, cap, cost = lanes_out[d][s]
            if exists:
                arcs.append((dc_out0 + d, store0 + s, cap, cost, K_LANE_OUT))
    for s in range(S):
        arcs.append((store0 + s, 1, demand[s], 0, K_DEMAND))
    return Net(tuple(names), tuple(labels), tuple(pos), tuple(arcs), 0, 1, True)
