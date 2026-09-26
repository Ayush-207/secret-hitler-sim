# Secret Hitler Trust-Model Simulation

A Monte Carlo simulator testing whether liberals in a 6-player Secret Hitler game
can win more often with a purely statistical trust model — Bayesian posteriors
updated only on *observable moves* (enacted policies, nominations), with no
reliance on what players say.

**Answer: yes, decisively** — the best liberal configuration wins ≥70% against
every fascist doctrine tested, including ones designed with full knowledge of the
liberal model. The single biggest lever is reading **nomination behavior**.

## Usage

```bash
python3 sim.py 10000    # full grid of liberal × fascist strategies, 10k games/cell
```

Python 3 stdlib only. Results print as a table (lib win%, checkmate%, shot-Hitler%,
elections, chaos, executions). Canonical rerun: `results_full_grid.txt`.

## Rules modeled

6 players: 4 liberals, 1 fascist, 1 Hitler (official setup). 17-card deck (11F/6L),
3-card presidential draws, public discards (deck composition = common knowledge).
5–6p power track: F2 investigate, F3 special election, F4 execution, F5 execution +
veto. Chaos (3 failed elections) enacts the top card with no power. Term limits on
the last successful government. Electing Hitler chancellor at ≥3F = instant fascist
win (the "checkmate").

## The model

Every liberal maintains a posterior over all 30 possible (Hitler, fascist) seat
assignments — the hidden state is fully described by an ordered pair of seats. Each
world is reweighted by how likely the observed play was *if that world were true*,
then renormalized. Decisions are queries against this one distribution:

- **Vote** — yes iff P(president & chancellor both liberal) ≥ threshold and
  chancellor's P(Hitler) ≤ cap. Thresholds tighten as fascist policies accumulate
  (0.28 → 0.40 joint; Hitler-cap 0.40 → 0.25 at 3F, the checkmate zone).
- **Nominate** — the argmax of P(president & candidate both liberal) among eligible
  seats, same cap.
- **Execute** — argmax P(Hitler). **Investigate** — argmax P(fascist); the reveal is
  a hard constraint (zeroes inconsistent worlds).

The enactment likelihood table is *derived from deck physics*, not chosen: e.g. a
mixed government enacts F with probability P(draw has ≥2 F) ≈ 0.73 (fresh deck), a
both-liberal government only P(FFF) ≈ 0.24. Likelihoods are clamped to
[0.03, 0.97] so no world dies on behavioral evidence — the model's humility about
its own assumptions.

## Strategies tested

**Liberal brains**

| brain | votes/executions | nomination | evidence channels |
|---|---|---|---|
| `naive` | yes unless the seat ever enacted F; random powers | next player clockwise | crude blacklist only |
| `stats1` | Bayesian, as above | argmax (deterministic) | enactments |
| `stats2` | = stats1 | random top-2 argmax | enactments |
| `stats1-9g` | = stats1 | argmax from private array | enactments **+ nomination-deviation (9:1)** |
| `hybrid` | = stats1-9g | **blind rotation at ≥3F**, argmax below | enactments + nomination-deviation |

**Fascist doctrines** (Hitler + the one fascist share it)

| doctrine | enact behavior | nomination behavior |
|---|---|---|
| `greedy` | F whenever the hand allows; president passes FF | partner / Hitler at 3F |
| `strategic` | F only when deniable (deck F-heavy or board ≥4F) | partner / Hitler at 3F |
| `groom` | Hitler plays perfectly clean (never voluntarily F); the fascist is greedy | partner / Hitler at 3F |
| `compliant` | greedy | **public-consensus pick always** (silences the nomination tell) |
| `blend` | strategic timing; Hitler clean | public-consensus pick always |

The nomination-deviation rule (`stats1-9g`'s addition): a **public belief**
(conditioned on no one's seat, updated on public evidence) defines the
group-consensus chancellor pick each election. A president nominating anything else
is a deviation — evidence at 9:1 against their seat being liberal, folded into every
liberal's private array. Compliance carries no update. The consensus is computable
from public information only, so the rule is well-defined without communication.

## Results

Liberal win rate by configuration, 10,000 games per cell (columns: liberal win %, %
of games ending in the Hitler-chancellor checkmate, % of games where Hitler was
executed, average elections and chaos events per game):

| liberals \ fascists | greedy | strategic | groom | compliant | blend |
|---|---|---|---|---|---|
| naive | 61.1 | 65.0 | 68.1 | 68.1 | 68.2 |
| stats1 | 71.7 | 65.8 | 62.1 | 62.1 | 62.9 |
| **stats1-9g** | **93.0** | **91.9** | **91.7** | **70.6** | **77.2** |
| hybrid | 57.3 | 57.2 | 57.8 | 57.8 | 57.6 |
| stats2 | 64.1 | 59.7 | 56.1 | 56.1 | 56.3 |

Checkmate% and shot-Hitler% for the stats1-9g row (the diagnostic columns that
explain the win rates):

| fascists | checkmate% | shot H% |
|---|---|---|
| greedy | 4.5 | 17.9 |
| strategic | 6.3 | 13.3 |
| groom | 6.1 | 12.3 |
| compliant | 26.1 | 4.0 |
| blend | 19.9 | 8.7 |

(Naive/stats1/stats2 rows against `compliant`/`blend` are identical to their `groom`
rows: those brains have no public belief, so compliant/blend fascists silently fall
back to groom behavior against them.)

## Findings

**1. The nomination tell is the biggest single lever.** Fascist presidents must
route power (partner, or Hitler at 3F) and almost never coincide with the public
consensus — measured deviation rates: **~90% for fascist presidents vs ~5% for
liberals**. That separation is nearly perfect signal. Adding the consensus-deviation
rule took liberals from 62.1% → 91.7% against groom, and instrumented runs confirm
the array pins both fascist seats from nominations alone, on top of enactment
evidence. A hard label (one strike, permanent) performs no better than 9:1 soft
evidence — the signal saturates.

**2. The enactment channel alone has a groom-shaped hole.** Plain `stats1` beats
greedy fascists convincingly (71.7% vs 61.1% naive) because greedy play feeds the
model maximal, well-attributed evidence. But groom — a Hitler who never voluntarily
enacts F — is invisible by construction: clean enactments are *positive* evidence,
so his posterior sinks while the greedy accomplice soaks up the suspicion and the
bullets. Argmax nomination then crowns him at 3F: checkmate 31.5% vs naive's 15.5%,
and a net loss vs naive. The model's own trust ranking becomes the attack's delivery
system.

**3. Trust-concentration is the vulnerability; the consensus rule fixes it as a side
effect.** `stats2` (random top-2 crowns) made things *worse* — under grooming,
Hitler is permanently in the top-2, so widening the lottery guarantees he's always
in it (56.1% vs groom, the worst liberal cell in the grid). The blind-crown `hybrid`
also failed (57–58% everywhere): rotation hands Hitler the chancellorship on a
guaranteed schedule, wastes tempo (chaos nearly doubles), and its blind nominations
trip the deviation rule (muddying the signal at exactly the checkmate-critical
phase). The winning response to a poisoned ranking was not to randomize the crown —
it was to make nominations themselves carry evidence, which starves groom of its
delivery vehicle.

**4. Fascist counter-adaptation recovers ~21 points, not a flip.** `compliant`
(nominate the consensus always, progress purely via enactments) blinds the
nomination channel: liberal win rate falls 91.7% → 70.6%. Perfect compliance is
itself too passive (fascist wins 29.4% — the best counter found); a 70/30 mix does
worse (fascist wins 23.2%) because deviated nominations are 9:1 evidence events, and
the `blend` (strategic enactment timing + clean Hitler + compliant nominations) does
worse still (liberals 77.2%) — delayed F's prolong the game while evidence
accumulates, and tempo beats stealth when stealth is already maxed.

**5. The escalation ladder** (liberal win rate vs `stats1-9g` unless noted):

```
greedy fascists        →  liberals (stats1) win 71.7%
  fascists adapt: groom   →  stats1 falls to 62.1% (checkmate 31.5%)
    liberals adapt: +nomination tell →  91.7%
      fascists adapt: compliant →  70.6%
        fascists adapt more: blend →  no improvement (liberals 77.2%)
liberals adapt: blind-crown hybrid →  fails (57.8%)
```

**Best-vs-best (minimax cell): `stats1-9g` vs `compliant` — liberals 70.6%.** Both
sides at their strongest; liberals ahead on every axis (policy race, executions,
checkmate denial).

## Why the rule works (the information argument)

If fascist behavior exactly matched liberal behavior, no observer could distinguish
them — the model would degenerate to priors. But fascists *must* control where power
flows to win: they need their F policies enacted (presidents shape hands) and
eventually need Hitler crowned at 3F. Every act of power-routing is a nomination,
and nominations are public and frequent (every election, including failed ones).
The consensus test prices exactly that: deviate-from-group-best is a behavioral
fingerprint fascists cannot avoid without giving up routing — and giving up routing
costs them the game's tempo. The equilibrium found here favors liberals because the
evidence channels fascists need (enactments, nominations) are the same channels
that incriminate them.

## Caveats

- The liberal likelihood tables assume greedy-ish behavior. A fascist doctrine
  *optimized against the exact tables* (not our heuristics) could do better — the
  blend was a step that direction and failed, but the true minimax fascist is
  unproven.
- The ~90% fascist deviation rate is a property of the tested doctrines, not a law.
  Real adversaries who know the rule can comply perfectly — at the measured cost of
  tempo (29.4% win rate).
- 6 players only. The board, Hitler-knowledge asymmetry, and posterior size
  (30 worlds) all change at other player counts.
- Compute realism: 30-world posteriors × 6 players is a laptop's job, not a human's.
  At a real table this is a bounded approximation (6 trust axes) at best.
- Discards are public in this sim (a defensible house rule); real games hide the
  president's discard, which removes some deck-composition information the model
  uses.
