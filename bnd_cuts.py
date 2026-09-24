"""Schnittungleichungen (Cut-Set-Ungleichungen) für das Fixkosten-Netzdesign, Trennung und Schnittschleife.

Für eine Knotenmenge W (ohne Quelle S und Senke T) muss so viel in W hinein fließen, wie die Nachfrage in W über das Angebot in W hinausgeht:
    R(W) = max( sum_k max(0, Nachfrage_k(W) - Angebot_k(W)),  Nachfrage(W) - Werkskapazität(W) )
Alles, was hineinfließt, läuft über die Kanten, die W von außen betreten (δ⁻(W)), und jede dieser Kanten trägt nur etwas, wenn sie offen ist:
    plain:  sum_{e in δ⁻(W)} u_e · y_e >= R(W)                                     (schon im schwachen LP enthalten, verletzt dort nie)
    cap:    sum min(u_e, R) · y_e >= R                                             (Kapazitäten auf R gekappt: mehr als R braucht der Schnitt nie)
    round:  sum ceil(min(u_e, R) / δ) · y_e >= ceil(R / δ)                         (Chvátal-Gomory-Rundung mit Teiler δ; y ist ganzzahlig, die linke Seite auch)
Alle drei gelten für jeden zulässigen Entwurf; das prüfen die Tests durch Aufzählen aller Entwürfe kleiner Netze.
"""

from dataclasses import dataclass
from math import ceil

import bnd_formulation as fm

MODES = ("plain", "cap", "round")
VIOL_TOL = 1e-6


@dataclass(frozen=True, eq=False)
class Cut:
    coef: dict            # Entwurfsgruppe -> ganzzahliger Koeffizient
    rhs: int
    W: frozenset          # Knotenmenge
    kind: str             # "plain", "cap" oder "round"
    delta: int            # Teiler der Rundung (0 bei plain/cap)
    need: int             # R(W)
    violation: float = 0.0

    def value(self, y):
        return sum(a * y[g] for g, a in self.coef.items())

    def holds(self, y, tol=1e-9):
        return self.value(y) >= self.rhs - tol

    def key(self):
        return (tuple(sorted(self.coef.items())), self.rhs)

    @property
    def score(self):
        """Verletzung relativ zur rechten Seite: vergleichbar zwischen den Formen (`cap` und `round` mit Teiler R sind dieselbe Ungleichung, nur um den Faktor R verschieden skaliert)."""
        return self.violation / max(1, self.rhs)


def inner_nodes(design):
    net = design.net
    return [v for v in range(net.n) if v not in (net.s, net.t)]


def required(design, W):
    """Was mindestens in die Knotenmenge W hinein fließen muss (R(W), ganzzahlig)."""
    mcf, net = design.mcf, design.net
    dem = [0] * design.K
    sup = [0] * design.K
    sup_cap = 0
    for e, (u, v, cap, _, _) in enumerate(net.arcs):
        if mcf.reward[e] and u in W:
            for k in range(design.K):
                dem[k] += mcf.ub[k][e]
        elif u == net.s and v in W:
            sup_cap += cap
            for k in range(design.K):
                sup[k] += min(mcf.ub[k][e], cap)
    per_good = sum(max(0, dem[k] - sup[k]) for k in range(design.K))
    return int(max(per_good, sum(dem) - sup_cap, 0))


def entering(design, W):
    """Entwurfskanten, die W betreten. None, wenn eine immer offene innere Kante W betritt (dann ist kein Schnitt über y möglich)."""
    net = design.net
    coef = {}
    for e, (u, v, cap, _, _) in enumerate(net.arcs):
        if v in W and u not in W and u != net.s:
            g = design.group[e]
            if g < 0:
                return None
            coef[g] = coef.get(g, 0) + cap
    return coef


def forms(design, W, modes=MODES, wrong=False):
    """Alle Formen des Schnitts zur Menge W (nur die in `modes` erlaubten); `wrong=True` erzeugt die absichtlich ungültige Variante (rhs + 1) für die Negativkontrolle."""
    R = required(design, W)
    if R <= 0:
        return []
    base = entering(design, W)
    if not base:
        return []
    out = []
    if "plain" in modes:
        out.append(Cut(dict(base), R, frozenset(W), "plain", 0, R))
    capped = {g: min(a, R) for g, a in base.items()}
    if "cap" in modes:
        out.append(Cut(capped, R, frozenset(W), "cap", 0, R))
    if "round" in modes:
        for d in sorted({a for a in capped.values() if a > 1}):
            coef = {g: ceil(a / d) for g, a in capped.items()}
            out.append(Cut(coef, ceil(R / d) + (1 if wrong else 0), frozenset(W), "round", d, R))
    return out


def _violated(cut, y):
    return cut.rhs - cut.value(y)


def best_form(design, W, y, modes=MODES, wrong=False):
    """Am stärksten verletzte Form zur Menge W bei der LP-Lösung y (None, wenn keine verletzt ist)."""
    best = None
    for cut in forms(design, W, modes, wrong):
        v = _violated(cut, y)
        if v > VIOL_TOL:
            cand = Cut(cut.coef, cut.rhs, cut.W, cut.kind, cut.delta, cut.need, v)
            if best is None or cand.score > best.score - 1e-9:              # bei gleichem Wert gewinnt die spätere (stärker gerundete) Form
                best = cand
    return best


# --- Trennung: Kandidatenmengen aus Min-Cuts ------------------------------------------------------------------------------------------

def _max_flow_sink_side(n, arcs, s, t):
    """Kleines Edmonds-Karp mit Gleitkommakapazitäten; liefert die Knoten, die im Restnetz von s aus NICHT erreichbar sind (Senkenseite eines minimalen Schnitts)."""
    head = [[] for _ in range(n)]
    to, cap = [], []
    for u, v, c in arcs:
        if c <= 1e-12:
            continue
        head[u].append(len(to)); to.append(v); cap.append(c)
        head[v].append(len(to)); to.append(u); cap.append(0.0)
    while True:
        parent = {s: -1}
        queue = [s]
        for u in queue:
            for a in head[u]:
                if cap[a] > 1e-12 and to[a] not in parent:
                    parent[to[a]] = a
                    queue.append(to[a])
        if t not in parent:
            return {v for v in range(n) if v not in parent}
        path, v = [], t
        while parent[v] != -1:
            path.append(parent[v]); v = to[parent[v] ^ 1]
        f = min(cap[a] for a in path)
        for a in path:
            cap[a] -= f; cap[a ^ 1] += f


def candidate_sets(design, y):
    """Kandidaten für verletzte Schnitte: Einzelknoten, Senkenseiten der Min-Cuts zu jedem Nachfrageknoten (alle Güter gemeinsam und je Gut), deren paarweise Vereinigungen und alle Komplemente."""
    net, mcf, K = design.net, design.mcf, design.K
    inner = set(inner_nodes(design))
    demand_nodes = sorted({net.arcs[e][0] for e in design.demand_arcs()} & inner)
    base = [(u, v, cap * y[design.group[e]]) for e, (u, v, cap, _, _) in enumerate(net.arcs) if design.group[e] >= 0]
    sets = {frozenset([v]) for v in inner}
    for k in [None] + list(range(K)):
        supply = [(u, v, cap if k is None else min(cap, mcf.ub[k][e])) for e, (u, v, cap, _, _) in enumerate(net.arcs) if u == net.s]
        for v in demand_nodes:
            if k is not None and not any(mcf.ub[k][e] > 0 for e in design.demand_arcs() if net.arcs[e][0] == v):
                continue
            side = _max_flow_sink_side(net.n, base + supply, net.s, v) & inner
            if side:
                sets.add(frozenset(side))
    found = sorted(sets, key=lambda w: (len(w), sorted(w)))
    unions = set()
    for i, a in enumerate(found):
        for b in found[i + 1:]:
            if len(unions) < 300:
                unions.add(a | b)
    sets |= unions
    sets |= {frozenset(inner - w) for w in list(sets) if inner - w}
    return sorted(sets, key=lambda w: (len(w), sorted(w)))


def _improve(design, W, y, modes, wrong, cut):
    """Lokale Suche: einen Knoten hinzunehmen oder herausnehmen, solange die Verletzung wächst."""
    inner = inner_nodes(design)
    for _ in range(2 * len(inner)):
        best = cut
        for v in inner:
            W2 = W - {v} if v in W else W | {v}
            if not W2:
                continue
            c2 = best_form(design, frozenset(W2), y, modes, wrong)
            if c2 is not None and (best is None or c2.score > best.score + 1e-9):
                best = c2
        if best is cut:
            return cut
        cut, W = best, best.W
    return cut


def separate(design, y, modes=MODES, limit=10, wrong=False, local_search=True, seeds=8):
    """Die am stärksten verletzten Schnitte bei y (höchstens `limit`, ohne Duplikate). Aus den Kandidatenmengen werden die `seeds` stärksten per lokaler Suche verbessert."""
    found = {}
    scored = []
    for W in candidate_sets(design, y):
        cut = best_form(design, W, y, modes, wrong)
        if cut is not None:
            scored.append(cut)
    scored.sort(key=lambda c: (-c.score, len(c.W)))
    pool = list(scored)
    if local_search:
        pool += [_improve(design, c.W, y, modes, wrong, c) for c in scored[:seeds]]
    for cut in pool:
        if cut.key() not in found or found[cut.key()].score < cut.score:
            found[cut.key()] = cut
    return sorted(found.values(), key=lambda c: (-c.score, len(c.W)))[:limit]


def separate_exact(design, y, modes=MODES, wrong=False):
    """Stärkste verletzte Form über ALLE Knotenmengen (Aufzählen, nur für kleine Netze). Maßstab für die Heuristik."""
    inner = inner_nodes(design)
    best = None
    for mask in range(1, 1 << len(inner)):
        W = frozenset(v for i, v in enumerate(inner) if mask >> i & 1)
        cut = best_form(design, W, y, modes, wrong)
        if cut is not None and (best is None or cut.score > best.score):
            best = cut
    return best


# --- Schnittschleife -------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Round:
    bound: float          # LP-Schranke nach dieser Runde
    added: tuple          # in dieser Runde hinzugefügte Schnitte
    y: object             # LP-Entwurf dieser Runde
    solution: object      # ganze LP-Lösung


@dataclass(frozen=True)
class CutResult:
    rounds: tuple
    cuts: tuple
    converged: bool       # True: keine verletzte Ungleichung mehr gefunden (False: Rundengrenze)
    infeasible: bool = False   # True: die Schnitte haben das LP unlösbar gemacht (nur mit ungültigen Schnitten)

    @property
    def bound(self):
        return float("inf") if self.infeasible else self.rounds[-1].bound

    @property
    def final(self):
        return self.rounds[-1].solution


def cut_and_solve(design, modes=MODES, max_rounds=30, per_round=10, wrong=False, level=fm.STRONG, local_search=True):
    """Schnittschleife: LP lösen, verletzte Schnitte suchen, hinzufügen, erneut lösen - bis keiner mehr verletzt ist."""
    cuts = []
    lp = fm.solve_lp(design, level)
    rounds = [Round(lp.objective, (), lp.y, lp)]
    converged = False
    for _ in range(max_rounds):
        new = separate(design, lp.y, modes, per_round, wrong, local_search)
        if not new:
            converged = True
            break
        cuts += new
        try:
            lp = fm.solve_lp(design, level, cuts)
        except fm.Infeasible:
            return CutResult(tuple(rounds), tuple(cuts), False, True)
        rounds.append(Round(lp.objective, tuple(new), lp.y, lp))
    return CutResult(tuple(rounds), tuple(cuts), converged)
