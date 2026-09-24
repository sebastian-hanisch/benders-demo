# Benders-Zerlegung – Entwurf im Master, Fluss im Teilproblem – Streamlit-Demo

*(noch nicht deployed)*

Elftes Stück der **Netzwerkfluss-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", zweites im Netzwerkdesign-Ast:
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – die **Benders-Zerlegung** für das Fixkosten-Netzdesign – an einem wachsenden Beispiel.
Das Modell ist das des Vorgängers: Mehrgüterfluss mit einer Ja/Nein-Entscheidung $y$ je Kante (Fixkosten). Ist der Entwurf gewählt, ist der Rest ein LP. **Benders** trennt genau das: ein **Master** wählt den Entwurf (ganzzahlig, mit einer Variable $\theta$ für die Flusskosten), ein **Teilproblem** rechnet den Fluss dazu; aus den **Dualpreisen der Kapazitäten** entsteht ein **Optimalitätsschnitt** $\theta\ge v(\bar y)+\sum_g\lambda_g(y_g-\bar y_g)$, aus einer Fehlmenge ein **Zulässigkeitsschnitt**. Die Schranken laufen aufeinander zu: untere = Master-Wert, obere = bester Entwurf mit Fluss.
Vehikel: das Distributionsnetz der Vorgänger (Standard, Seed 155), ein **Streckennetz** (Gitter) und drei feste Lehrnetze (Big-M-Falle, Rundungs-Falle, Bündelung).

**Einordnung in die Reihe (die Kanten des Graphen):** Das Modell und die Bausteine (Netze, Formulierung, Cut-Set-Ungleichungen) stammen aus [fixkosten-netzdesign-demo](https://github.com/sebastian-hanisch/fixkosten-netzdesign-demo); dort ging es um die *Schranke*, hier um das *Verfahren*, das den Entwurf vom Fluss trennt – und um die Frage, was die Schnitte des Vorgängers im Master bewirken. Benders passt nicht zum Rucksack (dort gibt es keinen LP-Teil, den man abspalten könnte); das Netzdesign ist sein natürliches Zuhause, das die Exakte-Suche-Linie offen ließ. Das Folgestück, **Slope Scaling**, ist die Heuristik für die Netze, die Benders nicht mehr schafft. Bisher gebaut: die ersten elf Stücke.
```
edmonds-karp-demo (Wurzel: Restgraph, Rückkanten, Max-Flow = Min-Cut)                  [gebaut]
  ├─ dinic-demo (viele kürzeste Wege je Phase: Niveaugraph, blockierender Fluss)        [gebaut]
  ├─ push-relabel-demo (kein Weg: Überschüsse schieben, Höhen anheben)                 [gebaut]
  └─ ssp-demo (Kosten: der billigste Weg im Restgraphen, Potenziale)                    [gebaut]
       ├─ cycle-canceling-demo (negative Kreise löschen) → Netzwerksimplex               [gebaut]
       │    (network-flow-demo)                                                          [gebaut als Fall-Demo]
       ├─ cost-scaling-demo (Push-Relabel + ε-Skalierung, das nutzt OR-Tools)           [gebaut]
       └─ multicommodity-demo (mehrere Güter teilen Kapazität: Kanten-LP, Preise)       [gebaut]
            ├─ mcf-column-generation-demo (Pfade als Spalten, Pricing = Dijkstra)       [gebaut]
            ├─ garg-koenemann-demo (Näherung mit Preisen, ohne LP-Löser)                [gebaut]
            └─ fixkosten-netzdesign-demo (Fixkosten: Schranke und Schnitte)             [gebaut]
                 ├─ benders-demo (Entwurf im Master, Fluss im Teilproblem)              [dieses Stück]
                 └─ Slope Scaling (Heuristik für große Netze)                           [geplant]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` belegt: Beispielnetze über ihre Seeds, Verteilungen über feste Netze (Seeds ab 100000, dieselben wie in den Vorgänger-Demos): 20 Netze für die Verteilung, 8 für die Varianten (6 lieferbar) und die LP-Schranke des Masters, 6 je Größe. Standard: 3 Güter, 3 Werke, 3 Verteilzentren, 8 Filialen, Netzdichte 60 %, Streuung 50 %, Auslastung 50 %, Fixkosten-Basis 30, starkes Teilproblem, Pareto-Schnitte, Cut-Set-Ungleichungen im Master.
Das **Optimum** rechnet HiGHS (MIP), Teilproblem und Master sind LPs bzw. MIPs in HiGHS, Dualpreise, Schnitte und Schleife eigener Code. Zahlen stehen gerundet („etwa“): die Iterationen hängen von der Wahl unter gleichwertigen Dualpreisen und Master-Lösungen ab, die Tests prüfen Bänder; Sekunden werden nur als grobes Verhältnis geprüft. Die Kopien aus dem Vorgänger sind bewacht (`tests/test_copies.py`).

**Das Verfahren funktioniert – und ist langsam.** Im Standardnetz (Seed 155) erreicht Benders das Optimum **1 784 nach etwa 40 Iterationen** (etwa 19 Optimalitäts- und 21 Zulässigkeitsschnitte, dazu 26 Cut-Set-Ungleichungen vorab): untere und obere Schranke fallen exakt zusammen, geprüft gegen das MIP von HiGHS und, auf Kleinstnetzen, gegen das Aufzählen aller Entwürfe. Das direkte MIP von HiGHS löst dasselbe Netz in Hundertsteln einer Sekunde, Benders braucht mindestens das Fünffache.
Über 20 feste Netze (18 lieferbar) liegt der Median bei **18 Iterationen**, 49 % davon sind Zulässigkeitsschnitte, alle konvergieren, und im Mittel braucht Benders mehr als das Fünffache der Zeit von HiGHS.

**Die Zulässigkeitsschnitte dominieren.** Das reine Verfahren (schwaches Teilproblem, Schnitte aus den Dualpreisen des Solvers, kein Start; 3/3/6-Netze) braucht im Median **69 Iterationen, 86 % davon Zulässigkeitsschnitte**: der Master beginnt mit „nichts öffnen“ (am billigsten) und lernt Schnitt für Schnitt, welche Entwürfe die Nachfrage überhaupt decken. Beispiel (Seed 7): 71 Iterationen, davon 63 Zulässigkeit, Optimum 1 348. Wie stark jede Zutat hilft (Median der Iterationen, 6 lieferbare Netze):
das reine Verfahren 69 · starkes Teilproblem 69 · dazu **Pareto-Schnitte 50** · dazu LP-Wurzelschnitte 76 (einschließlich der LP-Iterationen) · **Cut-Set-Ungleichungen im Master 14** · Pareto + Cut-Set **12**. Die Cut-Set-Ungleichungen aus dem Vorgängerstück sind für den Master gültig (sie gehen nur über $y$) und ersetzen im Kern die Zulässigkeitsschnitte – der Aufwand fällt auf etwa ein Fünftel, die Sekunden auf unter ein Drittel; jede Variante bleibt mindestens fünfmal langsamer als HiGHS.

**Die LP-Relaxation des Masters ist die schwache Schranke.** Mit allen Schnitten ist der LP-Master in allen 8 Netzen **exakt** die schwache LP-Schranke des Vorgängerstücks (66 bis 78 % des Optimums): die Benders-Schnitte beschreiben den Fluss-Wert $v(y)$ vollständig, mehr als die schwache Formulierung steckt nicht drin. Der Fortschritt des Verfahrens kommt allein aus den ganzzahligen Master-Läufen.

**Es skaliert nicht.** Benders (Pareto + Cut-Set) gegen HiGHS, Grenze 100 Iterationen: bei etwa 23 Entwurfskanten (3/3/8) im Mittel 14 Iterationen und alle 6 Netze konvergiert; bei etwa 42 Kanten (4/4/12) im Mittel 66 Iterationen, nur 3 von 5 Netzen erreichen das Optimum innerhalb der Grenze, und Benders braucht mehr als das Zwanzigfache der Zeit von HiGHS.
Lehrnetze: **Big-M-Falle** 3 Iterationen (1 Zulässigkeit, 2 Optimalität), Optimum 15; **Rundungs-Falle** 2 Iterationen, Optimum 23 (die Ungleichung $y_1+y_2\ge2$ liegt vorab im Master); **Bündelung** etwa 8 Iterationen, davon 6 Zulässigkeit, Optimum 11. Streckennetz (Gitter 4 × 3, Seed 6): Optimum 245 nach etwa 27 Iterationen, davon etwa 25 Zulässigkeitsschnitte.
**Negativkontrollen:** Ohne Zulässigkeitsschnitte hängt der Master im Big-M-Netz nach 2 Iterationen (immer derselbe Entwurf, untere Schranke 0). Um 5 % + 1 nach oben verschobene Optimalitätsschnitte sind ungültig: in der Rundungs-Falle liegt die untere Schranke nach 2 Iterationen bei 24,15 über dem Optimum 23 – das Verfahren beweist Falsches.

## Was nicht funktioniert hat / Vorab-Hypothesen

Vor dem Bau standen acht Vermutungen im Plan. Gemessen:

- **„Zulässigkeitsschnitte kommen nur in den ersten Iterationen“ – widerlegt.** Sie sind im reinen Verfahren die **Mehrheit** (86 %), im Streckennetz etwa 25 von 27. Der Master hat kein Vorwissen darüber, welche Entwürfe die Nachfrage decken, und lernt es ein Netz nach dem anderen. Die Vermutung „Tailing-off am Ende“ trifft deshalb nur halb: die Zeit geht am *Anfang* verloren.
- **„Das starke Teilproblem senkt die Iterationen stark“ – widerlegt.** 69 gegen 69 Iterationen. Im Vorgängerstück verschiebt die starke Kopplung die LP-Schranke auf Gittern; für die Schnitte des Teilproblems ändert sie kaum etwas, weil der Master keine Schnitte an gebrochenen Entwürfen sieht.
- **„Pareto-Schnitte senken die Iterationen deutlich“ – nur schwach.** 69 → 50 (etwa ein Viertel) im reinen Verfahren, mit den Cut-Set-Ungleichungen 14 → 12. Ein hilfreicher, aber kein entscheidender Baustein.
- **Bestätigt: Cut-Set-Ungleichungen im Master senken die Iterationen drastisch** (69 → 14, dazu die Sekunden auf unter ein Drittel). Sie sind gewissermaßen vorweggenommene Zulässigkeitsschnitte – das Kernstück, das Stück 10 und dieses Stück verbindet.
- **LP-Wurzelschnitte helfen nicht bei den Iterationen** (76 gegen 69 einschließlich der LP-Iterationen), nur beim Anteil der Zulässigkeitsschnitte (86 → 61 %): der LP-Master lernt einen Teil der Erreichbarkeit billig, aber nicht genug.
- **Bestätigt: Benders verliert gegen das direkte MIP** – auf jeder gemessenen Größe, mindestens fünffach langsamer (bei 42 Entwurfskanten über zwanzigfach), und die Lücke wächst mit der Netzgröße (14 → 66 Iterationen von 23 auf 42 Entwurfskanten). Das Verfahren lohnt, wenn der Fluss-Teil groß und der Entwurf klein ist (viele Szenarien); hier ist es umgekehrt.
- **Bestätigt (Theorie): LP-Master = schwache LP-Schranke**, in allen 8 Netzen exakt.
- **Negativkontrollen bestätigt:** ohne Zulässigkeitsschnitte kein Fortschritt; ungültige Schnitte beweisen Falsches. Die Tests prüfen die **Gültigkeit jedes Schnitts durch Aufzählen aller Entwürfe** kleiner Netze (Optimalitätsschnitt ≤ $v(y)$, Zulässigkeitsschnitt von jedem zulässigen Entwurf erfüllt, Pareto-Schnitt gültig und am Kernpunkt mindestens so hoch wie der normale) und an gebrochenen Entwürfen (Konvexität von $v$).

## Was die Demo zeigt

- **Iteration für Iteration:** ein Regler durch die Iterationen; je Iteration die Netzkarte mit dem Entwurf des Masters (blau/grau), dem Fluss des Teilproblems (orange), den Kapazitätspreisen im Schnitt (violett) und dem Optimum (rot), die Schranken (untere/obere) je Iteration mit Kreuz für Zulässigkeits- und Punkt für Optimalitätsschnitt, der Schnitt in Worten.
- **Verfahren wählbar:** Teilproblem schwach/stark, Schnitte normal/Pareto, Start (keiner / LP-Wurzelschnitte / Cut-Set im Master), Iterationsgrenze, Negativkontrollen.
- **Experimente (auf Abruf):** Verteilung über feste Netze, Varianten (Iterationen und Sekunden), Benders gegen HiGHS über Größen, Negativkontrollen und LP-Schranke des Masters.
- **Wo die Annahmen enden:** Fluss ist der leichte Teil, Master bleibt NP-schwer, Zulässigkeit nur über Schnitte, Lücke bei Zeitgrenze, statisches Netz.

## Modell und Verfahren

- **Teilproblem** (Entwurf $\bar y$ fest): $v(\bar y)=\min\{c\cdot x: A_{eq}x=0,\ A_x x\le b-A_y\bar y,\ lo\le x\le hi\}$; die Kapazitätszeilen sind $\sum_k x^k_e\le u_e y_g$ (stark zusätzlich $x^k_e\le\min(u_e,d_k)y_g$).
- **Schnitt aus dem Dual:** jede dual zulässige Lösung $(\lambda,\mu,\rho,\omega)$ gibt $v(y)\ge D(y)=\text{const}+(A_y^T\mu)\cdot y$ (schwache Dualität); die Duale, die im Punkt $\bar y$ optimal sind, liefern $D(\bar y)=v(\bar y)$. Koeffizienten unter $10^{-7}$ werden ohne Verlust der Gültigkeit entfernt.
- **Zulässigkeitsschnitt:** Phase-1-LP (Lieferung maximieren); ist die Fehlmenge positiv, verlangt der Schnitt $D_1(y)\le -d_{ges}$.
- **Pareto (Magnanti-Wong):** unter allen im Punkt $\bar y$ optimalen Dualen das mit dem größten $D(y^0)$ am Kernpunkt $y^0$ (Mitte zwischen der LP-Lösung und „alles offen“), gelöst als Hilfs-LP über das explizite Dual.
- **Master:** $\min f\cdot y+\theta$ mit allen Schnitten, HiGHS `milp`; untere Schranke = Master-Wert, obere = $\min(f\cdot\bar y+v(\bar y))$; Ende bei UB − LB ≤ 1e-6.
- **Starts:** *LP-Wurzelschnitte* (Benders am LP-Master, die Schnitte kommen in den ganzzahligen Master, die LP-Iterationen zählen mit) und *Cut-Set* (`bnd_cuts.cut_and_solve` aus dem Vorgänger, Ungleichungen nur über $y$).

## Dateien

```
app.py                  Oberfläche (Streamlit)
bnd_benders.py          Teilproblem, Phase 1, Dual, Pareto, Master, Schleife, Trace
bnd_evaluation.py       Auswertung, Verteilung, Experiment-Tabellen
bnd_visualization.py    Plotly-Abbildungen
bnd_presets.py          Permalink, Presets, Zufalls-Seed
bnd_constants.py        Konstanten, Regler-Grenzen, feste Seed-Mengen, Preset-Texte
bnd_model.py            Entwurfsmodell und Netze (Kopie aus fixkosten-netzdesign-demo)
bnd_formulation.py      LP-Stufen, MIP, Aufzählen (Kopie)
bnd_cuts.py             Cut-Set-Ungleichungen (Kopie)
bnd_scenario.py         Distributionsnetz und Zufallsgenerator (Kopie)
tests/                  Benders (Gültigkeit), Auswertung, Presets, Behauptungen, Kopien, App, Regler-Zustand
```

## Lokal starten

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
venv/Scripts/streamlit run app.py
```

## Tests ausführen

```bash
venv/Scripts/pip install -r requirements-dev.txt
venv/Scripts/python -m pytest tests/ -v
```

Die Tests laufen einige Minuten (die Behauptungen rechnen die festen Netze, die App-Tests durchlaufen jedes Preset und jede Kombination der Optionen).
