# Beating Secret Hitler with Statistics — the code

Companion to the [README](README.md), which carries the results, the strategy
matchup tables, and the findings. This doc is the code: what each component
does, why it is built that way, and how the implementation evolved.

## The question

Can liberals win Secret Hitler more often by ignoring everything players *say* and reading only what they *do*? In a simulated 6-player game (4 liberals, 1 fascist, 1 Hitler), yes: the best statistical liberals win **76–97%** of games against every fascist strategy tested (full grid in the README). Liberals who don't reason at all win 64–70%. The code generalizes to 5–10 players (official role/deck tables); scaling results in the README.

The whole simulation is one Python file ([`sim.py`](sim.py), standard library only). Every snippet below comes from it, trimmed for reading.

| Term | Meaning |
| --- | --- |
| F / L | A fascist / liberal policy card |
| World | One guess at the hidden roles: which seat is Hitler and which seats are fascists. 6p: 6 × 5 = 30 worlds; 10p: 10 × C(9,3) = 840 |
| Belief | One liberal's probability for each world |
| Likelihood | How probable an observed move is *if* a given world were true |
| Update | Multiply each world's probability by its likelihood, then rescale so they sum to 1 (Bayes' rule) |
| Argmax | Pick the option with the highest score |
| Checkmate | Electing Hitler chancellor once 3 F are on the board: instant fascist win |
| Chaos | 3 failed elections in a row: the top card of the deck is enacted automatically |

## The game in code

One `Game` object is one game. Setup deals the roles, shuffles the deck, and gives every seat a belief:

```python
SETUP = {5: (3,1,1), 6: (4,1,1), 7: (4,2,1), 8: (5,2,1),
         9: (5,3,1), 10: (6,3,1)}      # (liberals, fascists, hitler)
DECK_TABLE = {5: (11,6), 6: (11,6), 7: (12,6), 8: (12,6),
              9: (14,6), 10: (15,6)}   # (F, L) policy cards

def __init__(self, lib, fas, rng, n=6):
    nl, nf, nh = SETUP[n]
    df, dl = DECK_TABLE[n]
    roles = ["L"] * nl + ["F"] * nf + ["H"] * nh
    rng.shuffle(roles)                  # fresh random deal every game
    rng.shuffle(self.deck)
    self.beliefs = [Belief(i, n) for i in range(n)]
```

The rules the code enforces:

| Step | Rule |
| --- | --- |
| Election | The presidency rotates. The president nominates a chancellor; a majority vote passes the pair |
| Term limits | The last passed pair can't be the next chancellor |
| Policy | The president draws 3 and discards 1 secretly; the chancellor enacts 1 of the 2 left |
| Powers (on an F) | 2F investigate a party card · 3F pick the next president · 4F and 5F execute · 5F+ veto |
| Liberals win | 5 L on the board, or Hitler executed |
| Fascists win | 6 F on the board, or checkmate |

*Veto*: from 5F on, the chancellor may propose throwing away both cards; if the president agrees, nothing is enacted and the round counts as a failed election.

Liberals see only **public** events: who nominated whom, the votes, the policy enacted, and the chaos card. They never see the hands or the discards.

## The belief: 30 worlds, not 6 trust scores

The whole hidden state fits in one (Hitler, fascist-set) assignment. Each liberal keeps a probability for every assignment:

```python
# 6p: (hitler, frozenset-of-fascists), 30 worlds. 10p: 840.
HYPS = [(a, frozenset(fs))
        for a in range(n)
        for fs in combinations(range(n), nf)
        if a not in fs]

class Belief:
    def __init__(self, owner, n):
        self.w = [1.0 / len(self.hyps)] * len(self.hyps)
        for idx, (h, fs) in enumerate(self.hyps):
            if owner == h or owner in fs:   # "I know I'm liberal"
                self.w[idx] = 0.0
```

A "trust score" is not stored. It's read out of the 30 numbers when needed:

```python
def marginals(self):             # P(seat is Hitler), P(seat is bad)
    for idx, (h, f) in enumerate(HYPS):
        ph[h] += self.w[idx]
        pf[h] += self.w[idx]
        pf[f] += self.w[idx]

def p_pair_liberal(self, a, b):  # P(seats a and b are both liberal)
    return sum(w for (h, f), w in zip(HYPS, self.w)
               if a not in (h, f) and b not in (h, f))
```

Why not 6 separate scores? Because suspicions are linked. There are exactly two bad seats, so evidence against one seat clears the others. In a traced game, when seat 3 was caught nominating its partner, seat 2's suspicion fell from 0.57 to 0.35 with no new evidence about seat 2.

At the start, each other seat is bad in 8 of the 20 remaining worlds, so every liberal begins at P(bad) = 0.40 for everyone else.

## Evidence 1: which policy got enacted

The likelihoods come from counting cards, not from guesses. With a fresh deck (11 F, 6 L), a random 3-card hand is:

| F in hand | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| Probability | 0.03 | 0.24 | 0.49 | 0.24 |

A liberal president throws away an F whenever they can; a fascist throws away an L. Following each hand through those choices gives the chance each government ends up enacting F:

| Government in this world | P(enacts F) | Reason |
| --- | --- | --- |
| Both liberal | 0.24 | Only an FFF hand forces it |
| One fascist | 0.73 | Any hand with 2+ F |
| Both fascist | 0.97 | Anything but LLL |

The update asks every world: *if you were true, how likely was what we just saw?*

```python
for idx, (h, f) in enumerate(HYPS):
    tp = "F" if pres in (h, f) else "L"   # president's role in THIS world
    tc = "F" if chan in (h, f) else "L"   # chancellor's role in THIS world
    base = table[(tp, tc)]                 # that world's prediction
    lk = base if enacted_f else 1 - base
    lk = min(max(lk, 0.03), 0.97)          # never fully rule a world out
    self.w[idx] *= lk
self._renorm()                             # rescale to sum 1
```

The clamp to [0.03, 0.97] matters. The table assumes fascists always push F, and real players might not. Clamping means no single event can wipe out a world.

An enacted F is weak evidence, since honest governments get forced into it. An enacted L is strong evidence, since two fascists almost never pass one. That imbalance becomes important in the groom strategy below.

## Decisions: every action is a question to the belief

**Vote** yes only if the pair is probably clean and the chancellor probably isn't Hitler:

```python
joint = b.p_pair_liberal(pres, chan)
cap = 0.40 if self.board_f < 3 else 0.10              # stricter once checkmate is possible
t = {0: 0.32, 1: 0.32, 2: 0.32, 3: 0.40}.get(self.board_f, 0.40)
return joint >= t and ph[chan] <= cap
```

**Nominate** the argmax partner, but skip anyone who might be Hitler:

```python
scored = sorted(cands, key=lambda c: b.p_pair_liberal(pres, c), reverse=True)
ok = [c for c in scored if ph[c] <= cap]
return ok[0]
```

**Investigate** the argmax of P(bad). The party card seen is a certain fact, so inconsistent worlds are set to exactly 0, for the investigator only.

**Execute** the argmax of P(Hitler).

For comparison, the baseline `naive` liberal uses no belief. It nominates the next seat clockwise, votes no on anyone who has ever enacted an F, and uses powers on random targets.

## Evidence 2: the nomination tell

The rule: **if a president nominates someone other than the group's best-scoring candidate, treat it as evidence that the president is fascist.** Fascists have to route power to their partner, so they break from the consensus. This single rule was worth +20–30 points.

**Version 1 failed.** At first there was no shared consensus. Each liberal judged the president against *their own* top pick. But each liberal rules out a different seat (their own), so their rankings differ. Almost every nomination looked like a deviation to somebody, and innocent liberals got penalised constantly:

| Evidence strength | vs greedy fascists |
| --- | --- |
| off | 71.7% |
| 3:1 | 60.4% |
| 9:1 | 46.4% |
| hard label | 34.1% |

**Version 2 adds a public belief.** It's the same 30 worlds, but it rules out no seat. It is updated only on public events, so every liberal computes the **same** consensus pick:

```python
self.public = Belief(None)   # owner=None: no self-knowledge, identical for everyone

def consensus_pick(self):
    b = self.public
    scored = sorted(cands, key=lambda c: b.p_pair_liberal(self.pres, c), reverse=True)
    return [c for c in scored if ph[c] <= cap][0]
```

The public belief is only the referee. Votes, nominations and shots still use each liberal's private belief. A deviation is a public event, so it goes into every liberal's belief:

```python
if chan != consensus:                      # the president deviated
    for i in liberals:
        self.beliefs[i].update_nomination(self.pres, True, 9)
    self.public.update_nomination(self.pres, True, 9)

# update_nomination: worlds where the president is bad x0.9, others x0.1 (9:1)
lk = 9 / 10 if pres in (h, f) else 1 / 10
```

Measured deviation rates show why it works: **fascist presidents deviate ~90% of the time, liberal presidents ~5%.** With a true consensus the signal is almost clean. The strength barely matters (3:1 ≈ 9:1 ≈ hard label), because the signal saturates.

## The fascist strategies

Each one was built to beat the liberal model before it:

| Strategy | Enacting | Nominating | What it attacks |
| --- | --- | --- | --- |
| `greedy` | F whenever possible | partner; Hitler at 3F | nothing: the baseline |
| `strategic` | F only when "deniable" | partner; Hitler at 3F | the enactment evidence |
| `groom` | Hitler never passes F voluntarily | partner; Hitler at 3F | the trust ranking itself |
| `compliant` | greedy | always the consensus pick | the nomination tell |
| `blend` | strategic + clean Hitler | always the consensus pick | everything at once |

The core fascist nomination, and the compliant override that hides it:

```python
if self.fas in ("compliant", "blend"):
    return self.consensus_pick()           # look exactly like a liberal
if self.board_f >= 3 and self.hitler in cands:
    return self.hitler                     # go for checkmate
for i in cands:
    if i in self.fascists:
        return i                           # route power to the partner
```

**Groom** is the dangerous one against enactment evidence alone. A Hitler who only ever enacts L builds up exactly the evidence that clears a player. His suspicion score sinks, he rises to the top of every trust ranking, and argmax nomination crowns him at 3F. The model's own ranking delivers the checkmate.

## One round, end to end

Each pass of the main loop is one election:

```python
while True:
    chan = self.nominate()                        # 1. president picks a chancellor
    if chan != consensus: apply_nomination_tell() # 2. evidence from the nomination
    votes = [self.vote(i, self.pres, chan) for i in alive]   # 3. vote
    if sum(votes) > len(alive) // 2:
        if chan == hitler and self.board_f >= 3:
            return "F"                            # checkmate
        self.resolve_government(chan)             # 4. draw, discard, veto?, enact,
                                                  #    enactment evidence, powers
    else:
        self.fail += 1                            # 3 fails in a row -> chaos card
        if self.fail >= 3: self.chaos()
    self.pres = self.next_alive(self.pres)        # 5. presidency moves on
```

A real traced game: seat 3 = Hitler, seat 4 = fascist, and we follow liberal seat 0's P(bad).

| Election | What happened | Seat 0's belief |
| --- | --- | --- |
| 1 | s4 (fascist) nominates s3 (Hitler). Consensus was s0: deviation. Liberals vote no | s4: 0.40 → **0.86** |
| 2 | s5 + s0 pass, enact L | s5: 0.29 → 0.13 |
| 3 | s0 + s1 draw FFF and are forced to enact F | s1: 0.33 → 0.55 (innocent, unlucky) |
| 4 | s1 + s5 enact L | s1: 0.55 → 0.33 (recovers) |
| 5 | s2 + s0 draw FFF again. Power: s2 investigates s4 and privately sees Fascist | s2: 0.36 → 0.57 |
| 6 | s3 (Hitler) nominates s4: deviation. Vote fails | s3: 0.22 → **0.72**; s2 falls to 0.35 |
| 7 | s4 nominates s3: deviation. Third failure, chaos card L | s4: **0.96** |
| 8–9 | Clean liberal governments enact L, L | 5 L: **liberals win** |

Elections 3–4 show why the evidence is soft: an innocent player gets blamed for a bad draw, then clears themselves.

## What changed the scores: the evolution table

Each step answered a weakness the step before exposed. Win rates are the ones measured at the time; the full narrative and final grid live in the README.

| # | Change | Why | Result |
| --- | --- | --- | --- |
| 1 | `naive` baseline | control group | 61% vs greedy |
| 2 | `stats1`: 30-world belief, enactment evidence | read moves, not words | 72% vs greedy, but 62% vs groom (checkmate 31%) |
| 3 | `stats2`: pick randomly among the top 2 | stop groomed Hitler from sitting at #1 | worse: he's always in the top 2 anyway |
| 4 | Nomination tell, judged against each liberal's own pick | fascists must route power | collapsed to 34%: private rankings disagree |
| 5 | Public belief as the group consensus | one pick everyone agrees on | **91–93%** vs greedy, strategic, groom |
| 6 | Fascists answer with `compliant` | silence the tell | liberals down to 71% |
| 7 | `hybrid`: rotate the chancellorship blindly from 3F | deny groom its crown | failed (57–59%): Hitler gets the crown on a fixed schedule |
| 8 | Read the president's discard | extra evidence vs compliant | worse everywhere: same fact counted twice; removed |
| 9 | Stop reading the true draw pile | players can't see it | <1.5 point change |
| 10 | Track the draw pile separately in each world | the discards are secret but role-dependent | pile error 0.60 → 0.48 cards; ≤1.2 points |
| 11 | Fix reversed veto conditions | liberals were vetoing LL and enacting the losing 6th F | +0.7 to +3.8 points |
| 12 | `compliant` fascists play greedy cards too | the doc described the old mix; greedy cards are the stronger doctrine | stats1-9g vs compliant: 72.2 → **58.9** |
| 13 | Counter A: provably-not-Hitler | chancellor at ≥3F + game continued = not Hitler; zero those worlds (exact, no parameters) | 58.1 → 60.2 vs compliant (4k prototypes) |
| 14 | Counter B: crown cap 0.25 → 0.10 at ≥3F | demand near-certainty when the crown is a weapon | 58.1 → 63.3; A+B: 64.9 |
| 15 | Liberal presidents comply too | innocent deviations (6%) were the only ones compliant produced; remove them | 63.8 → 68.1 vs compliant; every cell gains |
| 16 | Ignore deviations from an illegal consensus | when the consensus pick exceeded the cap there was nothing to deviate from; penalizing it punished innocence | 68.1 → 69.2 vs compliant |
| 17 | Liberals vote as a block (`LIB_VOTE_COMPLY`) | private-vote splits went 3–2 on borderline elections; fascist yes-votes flipped 1,195/26,700 | worst case 69.2 → **76.3** (10k, with calibrated thresholds); policy losses 450 → 255 |
| 18 | Calibrate the vote thresholds (`VOTE_T` = 0.32 flat) | swept like the crown cap; the hand-picked 0.28 sat at a plateau's edge; 0.40 flat reproduces the original deadlock as a measured tail (15.6%) | +2.0 worst case; final grid 76.3 vs compliant |
| 19 | Generalize 6p → 5–10 players | official setup/deck/power tables; worlds become (Hitler, fascist-set): 30 → 840 | every cell ≥ 6p grid at every count tested; see README scaling table |
| 20 | Re-sweep the gate ratios at 9–10p | test the transfer hypothesis; only the crown cap moved — it flips direction with the tempo/checkmate balance (looser = better when tempo dominates) | worst case +3 to +5.7 at 7–10p; 6p regression still exact (25/25 rows) |

**The row-19 scaling lesson: gates are prior-dependent, and the regression
catches what review doesn't.** Two findings from generalizing. First, the 6p
calibration produced *absolute* gates (vote 0.32, crown cap 0.10), but the pair
prior at game start falls from 0.60 (6p) toward 0.31 (10p) as the fascist set
grows — copying the absolutes to 10p would deadlock every election into chaos
(the row-18 0.40-flat tail at every count). The gates are re-derived per count
as fixed ratios to that count's own game-start priors, with ratios chosen so
6p reproduces the published constants exactly. Second, an earlier draft also
scaled the enactment-threshold logic with the fascist count (`nf + 1 - …`); the
greedy-discard logic is actually count-independent (`3 - bads`). Nothing but
the 10k 6p regression grid — an exact match to the pre-parameterization
results, to the decimal — caught it.

A calibration lesson from row 18 that cost a full grid rerun to find: **a
threshold calibrated on one belief architecture does not transfer to another.**
The 0.32 was measured on the unified public-belief vote path. Brains voting on
*private* arrays (stats1/stats2/hybrid) score pairs lower — self-knowledge
zeroing removes worlds and drags the pair score down to ~0.30 at game start —
so 0.32 deadlocked them (16% wins, 8.65 chaos/game). The two vote paths keep
their own constants.

## Limits

- The fascist strategies are hand-written rules, not an optimal opponent. A fascist tuned against the exact likelihood tables could do better.
- The liberal model assumes fascists push F when they can. Against other styles its numbers are a little off, which is why every likelihood is clamped.
- Votes are public in the real game. As *evidence* they were measured and rejected: encoding the fascists' vote rule into the likelihood hit 99.8% (stacked deck — we wrote both sides), but 20% fascist vote noise drops it to 83.7 and 40% inverts it to 49.9 (worse than ignoring votes). As *mechanics* they matter: unified public-belief voting (row 17) is worth +6.7 on the worst case. Veto proposals/acceptances and the *targeting choices* of liberal executive powers are also unmodeled.
- Player counts 5–10 are implemented with gates re-derived per count as ratios
  to that count's priors. The vote-gate ratio was confirmed to transfer (flat
  sweep at 9p); the crown-cap ratio was re-swept at 9–10p and moves to 0.80 for
  all multi-fascist counts (row 20). The 7p cap sweep was not run — 7–8p
  inherit the 9–10p value.
- 840 worlds × 10 players is a laptop's job. At a real table, this is a guide to reasoning, not a procedure to follow.
