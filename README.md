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
3-card presidential draws, **private discards** (as in the real game — players see
enacted policies, never the discarded card). Deck-dependent likelihoods use only the
**public pool**: cards not on the board, `(11 − F enacted, 6 − L enacted)`. 5–6p power track: F2 investigate, F3
special election, F4 execution, F5 execution + veto. Chaos (3 failed elections)
enacts the top card with no power. Term limits on the last successful government.
Electing Hitler chancellor at ≥3F = instant fascist win (the "checkmate").

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
both-liberal government only P(FFF) ≈ 0.24. Since the draw pile's true split is
hidden, these are computed from the public pool — a 3-card draw from the pile is
distributed like a draw from the pool when you don't know which cards were
discarded. Likelihoods are clamped to
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
| `strategic` | F only when deniable (public pool F-heavy or board ≥4F — in practice almost always); Hitler clean | partner / Hitler at 3F |
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
| naive | 61.1 | 65.0 | 68.1 | 68.1 | 68.1 |
| stats1 | 71.6 | 67.2 | 62.9 | 62.9 | 62.9 |
| **stats1-9g** | **93.0** | **93.0** | **92.4** | **70.8** | **77.6** |
| hybrid | 58.5 | 57.8 | 56.9 | 56.9 | 56.9 |
| stats2 | 64.4 | 59.6 | 56.7 | 56.7 | 56.7 |

Checkmate% and shot-Hitler% for the stats1-9g row (the diagnostic columns that
explain the win rates):

| fascists | checkmate% | shot H% |
|---|---|---|
| greedy | 4.5 | 17.6 |
| strategic | 5.4 | 13.8 |
| groom | 6.0 | 13.0 |
| compliant | 25.9 | 5.3 |
| blend | 19.5 | 9.3 |

(Naive/stats1/hybrid/stats2 rows against `compliant`/`blend` are identical to their
`groom` rows: those brains have no public belief, so compliant/blend fascists fall
back to groom behavior against them. `blend` differs from `groom` only in its
nominations, since its fascist's deniability check is almost always true on public
information.)

## Findings

**1. The nomination tell is the biggest single lever.** Fascist presidents must
route power (partner, or Hitler at 3F) and almost never coincide with the public
consensus — measured deviation rates: **~90% for fascist presidents vs ~5% for
liberals**. That separation is nearly perfect signal. Adding the consensus-deviation
rule took liberals from 62.9% → 92.4% against groom, and instrumented runs confirm
the array pins both fascist seats from nominations alone, on top of enactment
evidence. A hard label (one strike, permanent) performs no better than 9:1 soft
evidence — the signal saturates.

**2. The enactment channel alone has a groom-shaped hole.** Plain `stats1` beats
greedy fascists convincingly (71.6% vs 61.1% naive) because greedy play feeds the
model maximal, well-attributed evidence. But groom — a Hitler who never voluntarily
enacts F — is invisible by construction: clean enactments are *positive* evidence,
so his posterior sinks while the greedy accomplice soaks up the suspicion and the
bullets. Argmax nomination then crowns him at 3F: checkmate 31.2% vs naive's 15.5%,
and a net loss vs naive. The model's own trust ranking becomes the attack's delivery
system.

**3. Trust-concentration is the vulnerability; the consensus rule fixes it as a side
effect.** `stats2` (random top-2 crowns) made things *worse* — under grooming,
Hitler is permanently in the top-2, so widening the lottery guarantees he's always
in it (56.7% vs groom, the worst liberal cell in the grid). The blind-crown `hybrid`
also failed (57–59% everywhere): rotation hands Hitler the chancellorship on a
guaranteed schedule, wastes tempo (chaos nearly doubles), and its blind nominations
trip the deviation rule (muddying the signal at exactly the checkmate-critical
phase). The winning response to a poisoned ranking was not to randomize the crown —
it was to make nominations themselves carry evidence, which starves groom of its
delivery vehicle.

**4. Fascist counter-adaptation recovers ~21 points, not a flip.** `compliant`
(nominate the consensus always, progress purely via enactments) blinds the
nomination channel: liberal win rate falls 92.4% → 70.8%. Perfect compliance is
the best counter found (fascist wins 29.2%); a 70/30 mix did worse in an earlier run
(fascist wins ~23%) because deviated nominations are 9:1 evidence events, and the
`blend` (strategic enactment timing + clean Hitler + compliant nominations) does
worse still (liberals 77.6%) — delayed F's prolong the game while evidence
accumulates, and tempo beats stealth when stealth is already maxed.

**5. The escalation ladder** (liberal win rate vs `stats1-9g` unless noted):

```
greedy fascists        →  liberals (stats1) win 71.6%
  fascists adapt: groom   →  stats1 falls to 62.9% (checkmate 31.2%)
    liberals adapt: +nomination tell →  92.4%
      fascists adapt: compliant →  70.8%
        fascists adapt more: blend →  no improvement (liberals 77.6%)
liberals adapt: blind-crown hybrid →  fails (56.9%)
```

**Best-vs-best (minimax cell): `stats1-9g` vs `compliant` — liberals 70.8%.** Both
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
  tempo (29.2% win rate).
- 6 players only. The board, Hitler-knowledge asymmetry, and posterior size
  (30 worlds) all change at other player counts.
- Compute realism: 30-world posteriors × 6 players is a laptop's job, not a human's.
  At a real table this is a bounded approximation (6 trust axes) at best.
- Discards are private, as in the real game, and no strategy sees the draw pile's
  true split — deck-based likelihoods use the public pool (cards not on the board).
  That pool ignores one real effect: liberal presidents discard F's, so the discard
  pile leans F and the draw pile is slightly more liberal than the pool suggests. A
  full model would reason about that skew too. (An earlier version read the true
  draw pile — information no player has; removing it moved every cell by <1.5
  points.)
- With only public information the `strategic` fascist's deniability test is almost
  always true (the pool stays F-majority), so that doctrine now plays near-greedy.
  A better timing trigger is an open item.
