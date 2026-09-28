# Secret Hitler Trust-Model Simulation

A Monte Carlo simulator testing whether liberals in a 6-player Secret Hitler game
can win more often with a purely statistical trust model — Bayesian posteriors
updated only on *observable moves* (enacted policies, nominations), with no
reliance on what players say.

**Answer: yes, decisively** — the best liberal configuration wins ≥70% against
every fascist doctrine tested, including ones designed with full knowledge of the
liberal model. The single biggest lever is reading **nomination behavior**.

Code walkthrough (how each component works, how the model evolved): [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Usage

```bash
python3 sim.py 10000          # full grid of liberal × fascist strategies, 10k games/cell
python3 sim.py 10000 pool     # same, with the simpler shared-pool deck model
```

Python 3 stdlib only. Results print as a table (lib win%, checkmate%, shot-Hitler%,
elections, chaos, executions). Canonical rerun: `results_full_grid.txt`.

## Rules modeled

6 players: 4 liberals, 1 fascist, 1 Hitler (official setup). 17-card deck (11F/6L),
3-card presidential draws, **private discards** (as in the real game — players see
enacted policies, never the discarded card); no strategy sees the draw pile's true
split. 5–6p power track: F2 investigate, F3 special election, F4 execution, F5 execution + veto (from 5F the chancellor may propose discarding both cards; if the
president agrees nothing is enacted and it counts as a failed election — liberals use it
to escape a forced 6th F). Chaos (3 failed elections)
enacts the top card with no power. Term limits on the last successful government.
Electing Hitler chancellor at ≥3F = instant fascist win (the "checkmate").

## The model

Every liberal maintains a posterior over all 30 possible (Hitler, fascist) seat
assignments — the hidden state is fully described by an ordered pair of seats. Each
world is reweighted by how likely the observed play was *if that world were true*,
then renormalized. Decisions are queries against this one distribution:

- **Vote** — yes iff P(president & chancellor both liberal) ≥ threshold and
  chancellor's P(Hitler) ≤ cap. Thresholds: 0.32 flat, 0.40 at ≥3F
  (calibrated by sweep, see below); Hitler-cap 0.40 → 0.10 at 3F
  (calibrated, see below).
- **Nominate** — the argmax of P(president & candidate both liberal) among eligible
  seats, same cap.
- **Execute** — argmax P(Hitler). **Investigate** — argmax P(fascist); the reveal is
  a hard constraint (zeroes inconsistent worlds).

The enactment likelihood table is *derived from deck physics*, not chosen: e.g. a
mixed government enacts F with probability P(draw has ≥2 F) ≈ 0.73 (fresh deck), a
both-liberal government only P(FFF) ≈ 0.24. Likelihoods are clamped to [0.03, 0.97] so no world dies on behavioral evidence — the model's humility about
its own assumptions.

**Deck tracking.** Discards are secret, so nobody knows the draw pile's exact split
between reshuffles. But inside one world every role is known, so that world can
infer how many F each drawn hand held from what was enacted — and the draw pile
loses exactly the drawn hand, whatever is discarded afterwards. Each world therefore
carries its own distribution over "F cards left in the draw pile", reset to the exact
value at every reshuffle (the new pile is every card not on the board). Chaos cards
are evidence too: each world predicts their colour from its own deck estimate. A
simpler alternative (`pool` mode) gives every world the same estimate — the cards
not on the board — which ignores that discards are chosen by role. For the true
world the tracker's error on the pile's F count is 0.48 cards vs 0.60 for the pool;
switching between them moves every cell by ≤1.2 points (within noise).

## Strategies tested

**Liberal brains**

| brain | votes/executions | nomination | evidence channels |
|---|---|---|---|
| `naive` | yes unless the seat ever enacted F; random powers | next player clockwise | crude blacklist only |
| `stats1` | Bayesian, as above | argmax (deterministic) | enactments |
| `stats2` | = stats1 | random top-2 argmax | enactments |
| `stats1-9g` | = stats1, **unified public-belief vote** | **public-consensus pick** (comply) | enactments **+ nomination-deviation (9:1)** + provably-not-Hitler + crown cap 0.10 at ≥3F |
| `hybrid` | = stats1-9g | **blind rotation at ≥3F**, argmax below | enactments + nomination-deviation |

**Fascist doctrines** (Hitler + the one fascist share it)

| doctrine | enact behavior | nomination behavior |
|---|---|---|
| `greedy` | F whenever the hand allows; president passes FF | partner / Hitler at 3F |
| `strategic` | F only when deniable (remaining cards F-heavy or board ≥4F — in practice almost always); Hitler clean | partner / Hitler at 3F |
| `groom` | Hitler plays perfectly clean (never voluntarily F); the fascist is greedy | partner / Hitler at 3F |
| `compliant` | greedy | **public-consensus pick always** (silences the nomination tell) |
| `blend` | strategic timing; Hitler clean | public-consensus pick always |

The nomination-deviation rule (`stats1-9g`'s addition): a **public belief**
(conditioned on no one's seat, updated on public evidence) defines the
group-consensus chancellor pick each election. A president nominating anything else
is a deviation — evidence at 9:1 against their seat being liberal, folded into every
liberal's private array. Compliance carries no update. Liberal presidents comply
too (it costs nothing measurable and drops innocent deviations ~30×), so in
equilibrium the tell fires almost exclusively on fascists forced to route power
at ≥3F — the channel only punishes those who *must* lie. The consensus is computable
from public information only, so the rule is well-defined without communication.

## Results

Liberal win rate by configuration, 10,000 games per cell (columns: liberal win %, %
of games ending in the Hitler-chancellor checkmate, % of games where Hitler was
executed, average elections and chaos events per game):

| liberals \ fascists | greedy | strategic | groom | compliant | blend |
|---|---|---|---|---|---|
| naive | 64.2 | 66.9 | 69.8 | 64.2 | 69.8 |
| stats1 | 75.0 | 69.8 | 65.8 | 75.0 | 65.8 |
| **stats1-9g** | **97.0** | **96.8** | **96.3** | **76.3** | **87.1** |
| hybrid | 58.5 | 60.5 | 59.4 | 58.5 | 59.4 |
| stats2 | 73.2 | 68.3 | 63.9 | 73.2 | 63.9 |

The `stats1-9g` row includes all four liberal adaptations (crown-phase counters
A+B, nomination compliance, unified voting — see Findings 6–7 and the
calibration sections below). Without any of them the row reads 95.2 / 93.6 /
93.0 / 58.9 / 78.2 — the adaptations lift the worst case by +17.4 points at no
cost to any other cell.

Checkmate% and shot-Hitler% for the stats1-9g row (the diagnostic columns that
explain the win rates):

| fascists | checkmate% | shot H% |
|---|---|---|
| greedy | 1.4 | 16.8 |
| strategic | 2.4 | 13.3 |
| groom | 3.0 | 12.8 |
| compliant | 13.8 | 9.8 |
| blend | 10.8 | 8.2 |

(Rows for the brains *without* a public belief follow a simple pattern: against
`compliant` they equal their `greedy` row, and against `blend` their `groom` row.
The compliant/blend nomination rule only activates when a public belief exists
to compute a consensus from — against these brains the fascists fall back to
greedy/groom routing, and blend's enactment timing is near-indistinguishable
from groom's since its deniability check is almost always true.)

### Calibrating the crown cap (why 0.10)

The Hitler-cap at ≥3F (the checkmate zone) started life as a hand-pick: 0.25.
Against `compliant` it does almost nothing — a groomed Hitler reaches the crown
vote at a median public P(Hitler) of 0.074, far below any cap meant to catch the
*suspicious*. Tightening the cap attacks the opposite end: when the chancellorship
is a loaded gun, don't just avoid the suspicious — demand near-certainty. The
parameter is `CROWN_CAP` in `sim.py`; sweeping it (4,000 games per cell, full
data in `cap_sweep.txt`):

| CROWN_CAP | compliant | greedy | groom | blend | strategic | worst case |
|---|---|---|---|---|---|---|
| 0.40 | 58.5 | 95.0 | 92.5 | 79.8 | 93.4 | 58.5 |
| 0.25 (old default) | 58.7 | 95.0 | 92.6 | 79.8 | 93.5 | 58.7 |
| 0.15 | 59.5 | 96.2 | 93.8 | 80.0 | 94.7 | 59.5 |
| **0.10** | **62.8** | **96.8** | **94.5** | **81.6** | **95.4** | **62.8** |
| 0.05 | 57.7 | 96.0 | 94.8 | 82.2 | 95.9 | 57.7 |
| 0.02 | 52.3 | 94.5 | 94.1 | 79.9 | 95.1 | 52.3 |

The curve peaks at 0.10 and falls off on **both** sides, for structural reasons:

- **Too loose (0.40 → 0.25):** flat. Groomed Hitler at 0.074 passes any cap in
  this range — the cap was calibrated to block seats at 0.25–0.40, and compliant
  fascists simply stay under it.
- **Too tight (0.05 → 0.02):** the cap blocks *liberals* too. At 0.02 almost no
  candidate passes, governments fail, chaos enacts cards with no liberal
  oversight, and the policy race collapses (52.3%).

So 0.10 is not magic — it is the measured interior optimum of a two-sided
trade-off. Its exact location could shift against a differently-groomed Hitler
(the 0.10-vs-0.15 gap exceeds the ±1 point noise, but the true peak may sit
anywhere in 0.07–0.15); the *existence* of an interior peak does not, since both
failure modes are structural. Pairing the cap with an exact rule — a seat elected
chancellor at ≥3F while the game continued is **provably not Hitler** (zero those
worlds) — covers the cap's blind spot (records clean enough to fall under any
threshold) and lifts the compliant cell further; see Findings 6.

### Calibrating the vote thresholds (why 0.32)

The joint-liberal vote gates (`VOTE_T`, one threshold per board state) were
hand-picked early — 0.28/0.28/0.30/0.40 — to break the original deadlock where
0.40 everywhere passed *no* government. They were never re-examined, even after
unified voting made the threshold govern *every* liberal vote as one block.
Sweeping flat values (3,000 games per cell, full data in `vote_sweep.txt`):

| flat t | compliant | greedy | groom | blend | strategic | worst case |
|---|---|---|---|---|---|---|
| 0.24 | 74.5 | 96.0 | 95.1 | 86.1 | 95.8 | 74.5 |
| 0.28 (old default) | 75.5 | 96.2 | 95.4 | 86.5 | 96.1 | 75.5 |
| **0.32** | **77.5** | **96.7** | **95.9** | **87.1** | **96.6** | **77.5** |
| 0.36 | 76.8 | 96.9 | 95.7 | 87.0 | 96.4 | 76.8 |
| 0.40 | 15.6 | 93.5 | 93.0 | 64.9 | 93.7 | **15.6** |

Same shape as the cap sweep: an interior peak (a 0.30–0.34 plateau, +2.0 worst
case over the old default) with a structural collapse past 0.36 — at 0.40 flat
the threshold exceeds the pair-prior at game start, no government passes, and
chaos hands fascists the game (15.6%: the deadlock that motivated the original
hand-pick, reproduced as a measured tail). The endgame threshold t₃ (board ≥3F)
is *insensitive* across 0.28–0.44 — endgame votes are governed by the crown cap
and the provably-not-Hitler rule, not the vote gate. New default:
`VOTE_T = {0: 0.32, 1: 0.32, 2: 0.32, 3: 0.40}`.

## Findings

**1. The nomination tell is the biggest single lever.** Fascist presidents must
route power (partner, or Hitler at 3F) and almost never coincide with the public
consensus — measured deviation rates: **~90% for fascist presidents vs ~0.2% for
liberals** (after liberal compliance; ~5–6% before it). That separation is nearly
perfect signal. Adding the consensus-deviation
rule took liberals from 64.2% → 93.0% against groom, and instrumented runs confirm
the array pins both fascist seats from nominations alone, on top of enactment
evidence. A hard label (one strike, permanent) performs no better than 9:1 soft
evidence — the signal saturates.

**2. The enactment channel alone has a groom-shaped hole.** Plain `stats1` beats
greedy fascists convincingly (75.3% vs 64.2% naive) because greedy play feeds the
model maximal, well-attributed evidence. But groom — a Hitler who never voluntarily
enacts F — is invisible by construction: clean enactments are *positive* evidence,
so his posterior sinks while the greedy accomplice soaks up the suspicion and the
bullets. Argmax nomination then crowns him at 3F: checkmate 31.4% vs naive's 15.5%,
and a net loss vs naive. The model's own trust ranking becomes the attack's delivery
system.

**3. Trust-concentration is the vulnerability; the consensus rule fixes it as a side
effect.** `stats2` (random top-2 crowns) made things *worse* — under grooming,
Hitler is permanently in the top-2, so widening the lottery guarantees he's always
in it (58.4% vs groom, the worst liberal cell in the grid). The blind-crown `hybrid`
also failed (59–63%, below naive in every column): rotation hands Hitler the chancellorship on a
guaranteed schedule, wastes tempo (chaos nearly doubles), and its blind nominations
trip the deviation rule (muddying the signal at exactly the checkmate-critical
phase). The winning response to a poisoned ranking was not to randomize the crown —
it was to make nominations themselves carry evidence, which starves groom of its
delivery vehicle.

**4. Fascist counter-adaptation recovers ~21 points, not a flip.** `compliant`
(nominate the consensus always, progress purely via enactments) blinds the
nomination channel: liberal win rate falls 93.0% → 72.2%. Perfect compliance is
the best counter found (fascist wins 27.8%); a 70/30 mix did worse in an earlier run
(fascist wins ~23%) because deviated nominations are 9:1 evidence events, and the
`blend` (strategic enactment timing + clean Hitler + compliant nominations) does
worse still (liberals 78.2%) — delayed F's prolong the game while evidence
accumulates, and tempo beats stealth when stealth is already maxed.

**5. The escalation ladder** (liberal win rate vs `stats1-9g` unless noted):

```
greedy fascists        →  liberals (stats1) win 75.3%
  fascists adapt: groom   →  stats1 falls to 64.2% (checkmate 31.4%)
    liberals adapt: +nomination tell →  93.0%
      fascists adapt: compliant →  72.2% (58.9% after the greedy-card fix)
        fascists adapt more: blend →  no improvement (liberals 78.2%)
liberals adapt: blind-crown hybrid →  fails (58.6%)
liberals adapt again: crown-phase counters (A+B+nomination-comply) →  69.3%
liberals adapt once more: +unified voting +calibrated thresholds →  76.3%
```

**6. Why compliant hurts, and the counter that works.** Under `compliant` the
nomination tell only fires on innocent liberals (fascist deviation rate 0.0%),
and compliant Hitler enacts F at 79% — almost exactly the liberal-baseline 73%
the likelihood table expects from a mixed hand. The enactment channel fires but
the evidence ratio is ~1.0: he is not more innocent than the model expects, he
is *exactly as* innocent as the model expects. Median public P(Hitler) at
crowning: 0.074; 98% of the crowns that killed the liberals were the model's own
consensus pick, and 1,626 of 1,963 were nominated by *liberal* presidents. The
strategy delivers Hitler to the checkmate itself.

Three counters, all in the final grid:

- **A — provably-not-Hitler (exact, no parameters).** A seat elected chancellor
  at ≥3F whose game continued cannot be Hitler. Zero those worlds.
- **B — crown cap 0.10 at ≥3F (calibrated, see above).** The cap alone is
  gameable by a clean enough record; A is not, and together they cover each
  other's blind spots.
- **C — liberal compliance.** Liberal presidents *also* nominate the
  public-consensus pick. This drops the innocent-deviation rate from ~6% to
  ~0.2% against honest-signaling doctrines — so any future deviation event is
  near-certain evidence of fascism, and the channel goes from 90:6 separation
  to 91:0.2. It wins in every cell (+0.9 to +4.3 points), for a structural
  reason: the president's private self-knowledge leaks into the public record
  anyway when they deviate (with a 9:1 tax attached); complying prices the same
  information into the *vote* instead, which carries no tax. A residual 1.9%
  innocent-deviation rate against `compliant` traced to a spurious tell — when
  the consensus pick itself exceeded the cap, no legitimate consensus existed,
  so deviations are no longer counted against a consensus that wasn't legal to
  follow. Fixing that lifted compliant a further point (68.1 → 69.2 at 4k).
  The general lesson: an evidence rule needs a *doable* reference action —
  comparing behavior against an impossible consensus manufactures signal out of
  shared rational behavior, and punishing innocence is exactly the failure mode
  compliant exploits.

Net effect of the counters plus calibrated thresholds on the 10k grid:
worst case 58.9 → **76.3**, with every other cell *improving* (the cleaner
channel helps everywhere).

**7. Anatomy of the remaining losses — and the voting fix.** With the counters
in, an autopsy of the ~31% of games `stats1-9g` still loses to `compliant`
(5,000-game instrumentation) split them into two near-equal paths:

- **Checkmate (~50%):** 84% of crown votes on Hitler are nominated by *liberal*
  presidents as the consensus pick, median public P(Hitler) at crowning 0.053,
  passing 5–0. The enactment channel cannot see groom (see Finding 6), and the
  counters made him *more* groomable by making everything else more suspicious.
- **Policy race (~50%):** final boards typically 6F/4L — the fascists win by one
  policy in long games (12.9 elections vs 10.7 for liberal wins). The tempo
  route, not the trap.

The policy path had a mechanical cause the vote records exposed: fascist votes
are decisive only in one direction. Liberals hold 4 of 6 seats and pass every
government they like *individually* — but liberal votes were cast from each
voter's **private** array, so borderline elections split 3–2 and fascist
yes-votes (their only vote lever: they cannot block anything, 2 < 4) flipped
1,195 of 26,700 elections. The fix mirrors nomination compliance: **all liberals
cast the same vote computed from the public belief** (`LIB_VOTE_COMPLY`) — a
4-vote block. Policy losses vs compliant dropped 450 → 255 (4k); checkmates
were untouched (the crown vote was already 5–0). The gain is voting mechanics,
not new evidence — and like every fix that removed liberal-side noise, it lifted
every cell: worst case 69.3 → 75.9 at 4k, **76.3 in the final 10k grid** after
the vote-threshold calibration below. One calibration trap from this fix: the
0.32 threshold was measured on the *public-belief* vote path, and does **not**
transfer to brains voting on private arrays — self-knowledge zeroing lowers
private pair scores enough that 0.32 deadlocks them (16% wins, 8.7 chaos/game).
The two paths keep their own constants (`VOTE_T` vs a private 0.28).

Two channels were *measured and rejected* on falsification grounds:

- **Vote-record evidence** (roll-calls as Bayesian events): with the liberals'
  likelihood set to the fascists' actual vote rule, liberals hit 99.8–100% —
  a stacked-deck result (we wrote both sides). Against 20% vote noise the edge
  falls to 83.7; at 40% it *inverts* (49.9 vs 78.2 ignoring votes). Enactment
  evidence survives misspecification because deck physics cannot adapt; vote
  evidence does not, because an adversary reshapes it at near-zero cost. The
  channel stays off.
- **Doctrine-mixture modeling (counter C)**: best worst case (66.8 vs 69.2 for
  A+B+comply) but ~5 points worse vs `blend`; rejected on the trade-off.
- Vote noise also has a smarter, targeted form than randomization: fascists
  voted yes on 6,066 hopeless governments (their votes decisive only in 1,195
  *passing* ones) — a "decisive-only" vote doctrine would keep the tempo and
  gut the fingerprint. Untested; noted as the refined threat model.
  (A final-config rerun with the vote channel accidentally left on confirmed
  its raw power even without disguise: every Bayesian brain scored 99.6–100%,
  including `stats1`, which has no public belief at all.)

**Best-vs-best (minimax cell): `stats1-9g` vs `compliant` — liberals 72.2%**
before the greedy-card fix, 58.9% after it, **76.3% through the full liberal
response** (crown-phase counters + compliance on both channels + calibrated
thresholds; 10k grid). Both sides at their strongest; the escalation ladder
keeps tilting back to the liberals because the channels they read are the
channels fascists must use.

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
  Real adversaries who know the rule can comply almost perfectly — at the measured
  cost of tempo (fascist win rate 30.7% vs compliant in the final grid).
- 6 players only. The board, Hitler-knowledge asymmetry, and posterior size
  (30 worlds) all change at other player counts.
- Compute realism: 30-world posteriors × 6 players is a laptop's job, not a human's.
  At a real table this is a bounded approximation (6 trust axes) at best.
- Each world's deck estimate assumes greedy-ish fascists (the same assumption as the
  enactment table), so against other doctrines it is slightly off. (An earlier
  version read the true draw pile — information no player has; removing it moved
  every cell by <1.5 points.)
- Fascists judge deniability from the cards not on the board, which stay F-majority,
  so the `strategic` test is almost always true and that doctrine plays near-greedy.
  A better timing trigger is an open item.
