#!/usr/bin/env python3
"""
Secret Hitler Monte Carlo, 5-10 players (6p default).

Question: does a purely statistical trust model (Bayesian posteriors over role
assignments, updated ONLY on observable moves -- enacted policies; no claims,
no speech) raise the liberals' win rate above a naive baseline, and how does
the answer degrade against different fascist play styles?

Assumptions:
- Player counts 5-10 with the official role/deck tables (SETUP / DECK_TABLE)
  and official board powers (POWERS). 6p default: 4 liberals, 1 fascist,
  1 Hitler. Veto unlocks at 5F on every board; 5L / 6F wins at every count.
- Discards are PRIVATE (as in the real game): players observe enacted
  policies, nominations, and votes -- never the discarded card's color.
  The pile's card COUNT is public (matters only for reshuffles).
- Chaos (3 failed elections) enacts the top card; no executive power fires.
- Term limits: the president+chancellor of the last successful government
  cannot be chancellor candidates in the immediately following election.
- Hitler-chancellor checkmate: vote passing with Hitler as chancellor and
  >=3F on the board = instant fascist win.
- The liberal belief model ASSUMES fascists are greedy (enact F whenever the
  hand allows). Real sim fascists may deviate -> deliberate misspecification;
  measuring robustness to it is the point of the experiment.

Player-count scaling of the calibrated gates: the 6p calibration produced
ABSOLUTE numbers (vote 0.32, crown cap 0.10, pre-3F cap 0.40, private vote
0.28). They are prior-dependent, so at every other count the gates are
re-derived as RATIOS to that count's own game-start priors (read off the
belief machinery itself in Game.__init__), with ratios chosen so 6p
reproduces the published constants exactly. See the ratio block below.
"""

import math
import random
from collections import Counter
from itertools import combinations

# ---------------- official setup tables ----------------
# roles per player count: (liberals, fascists, hitler)
SETUP = {5: (3, 1, 1), 6: (4, 1, 1), 7: (4, 2, 1), 8: (5, 2, 1),
         9: (5, 3, 1), 10: (6, 3, 1)}
# policy deck per count: (F, L)   [rulebook table]
DECK_TABLE = {5: (11, 6), 6: (11, 6), 7: (12, 6), 8: (12, 6),
              9: (14, 6), 10: (15, 6)}
# board powers by F count (official boards): inv = investigate, pe = special
# election, ex = execution. Veto unlocks at 5F on every board.
POWERS = {
    5:  {2: "inv", 3: "pe", 4: "ex", 5: "ex"},
    6:  {2: "inv", 3: "pe", 4: "ex", 5: "ex"},
    7:  {2: "inv", 3: "pe", 4: "ex", 5: "ex"},
    8:  {2: "inv", 3: "pe", 4: "ex", 5: "ex"},
    9:  {1: "inv", 2: "inv", 3: "pe", 4: "ex", 5: "ex"},
    10: {1: "inv", 2: "inv", 3: "pe", 4: "ex", 5: "ex"},
}
POLICY_F_WIN = 6     # fascist win: 6 F on the board (every count)
POLICY_L_WIN = 5     # liberal win: 5 L on the board (every count)

# ---------------- strategy flags (count-independent) ----------------
LIB_COMPLY = True   # liberal presidents also nominate the public-consensus pick
                    # (measured better in every cell -- see cap_sweep.txt / README)
VOTE_LK = 0.0       # vote-record evidence: OFF by default -- measured stacked-deck
                    # when on (we wrote the fascists' vote rule) and unstable under
                    # noise (40% noise -> worse than ignoring votes). See README F7.
LIB_VOTE_COMPLY = True  # liberals cast a unified public-belief vote (a block)
                        # (measured better on the worst case by +6.7 pts -- see README)
SE_TELL = False     # special-election evidence: measured NEUTRAL at 9p (38.3 vs
                    # 39.5 compliant, 66.1 vs 65.0 blend -- within noise), kept
                    # off. The channel is real (fascist presidents special-elect
                    # Hitler 86% vs 9.6% liberal) but compliant fascists can
                    # silence it the same way as nominations, and the consensus
                    # SE pick gives away the presidency half the time, which
                    # costs more than the tell gains. Code retained.

# ---------------- calibrated gates as RATIOS, not absolutes ----------------
# 6p game-start priors (measured from the belief objects in Game.__init__):
#   public pair prior  P(two seats both liberal)        = 12/30 = 0.40
#   public Hitler prior P(specific seat is Hitler)      = 1/6
#   private pair prior (observer rules out own seat)    = 6/20  = 0.30
#   private Hitler prior                                = 1/5   = 0.20
# Ratios chosen so 6p reproduces the published constants exactly:
#   vote gate      = 0.80   x public pair prior          (6p: 0.32)
#   vote gate >=3F = 1.25   x the <3F gate               (6p: 0.40)
#   crown cap (public ph, >=3F) = 0.60 x public Hitler prior (6p: 0.10)
#   pre-3F cap (public ph)      = 2.40 x public Hitler prior (6p: 0.40)
#   crown cap (private ph, >=3F)   = 0.50 x private Hitler prior (6p: 0.10)
#   pre-3F cap (private ph)        = 2.00 x private Hitler prior (6p: 0.40)
#   private vote gate = 0.9333 x private pair prior (6p: 0.28; row 18:
#   calibrated separately -- a threshold does not transfer across architectures)
#   private vote gate >=3F = 1.4286 x the <3F gate (6p: 0.40)
# The ratios are the null hypothesis that the 6p calibration transfers as
# SHAPE, not as numbers. Per-count sweeps (cap_sweep_9p.txt, cap_sweep_10p.txt)
# confirmed all but one: the crown cap over-tightens at >=7p, where the tempo
# race dominates and checkmate is rare -- a looser cap (0.80 of the Hitler
# prior, swept interior peak 0.80) gains +5-6 pts vs compliant at 9-10p. The
# cap scales with the count's fascist count; 6p keeps its own swept 0.60.
VOTE_R, VOTE_T3_R = 0.80, 1.25
CAP_PUB_R = {0: 2.40, 3: 0.60}      # x public Hitler prior (1/n)
CAP_PUB_R_MULTI = 0.80              # crown cap ratio at >=2 fascists (7p+)
CAP_PRIV_R = {0: 2.00, 3: 0.50}     # x private Hitler prior (1/(n-1))
PRIV_VOTE_R, PRIV_VOTE_T3_R = 0.28 / 0.30, 0.40 / 0.28
EPS = 0.03  # likelihood smoothing floor/ceiling
DEFAULT_N = 6

# worlds for n players: (hitler, frozenset-of-fascists), memoized per count
_HYPS_CACHE = {}


def hyps_for(n):
    """All role assignments: (hitler, frozenset of fascists). 6p: 30 worlds;
    10p: 10 * C(9,3) = 840."""
    h = _HYPS_CACHE.get(n)
    if h is None:
        nf = SETUP[n][1]
        h = [(a, frozenset(fs))
             for a in range(n)
             for fs in combinations(range(n), nf)
             if a not in fs]
        _HYPS_CACHE[n] = h
    return h


def hand_probs(df, dl):
    """P(k fascists in a 3-card draw) for k=0..3."""
    tot = math.comb(df + dl, 3)
    return [math.comb(df, k) * math.comb(dl, 3 - k) / tot for k in range(4)]


# ---------------- per-world deck tracking ----------------
# The draw pile's true split is hidden (discards are secret). But inside one
# world every role is known, so that world can infer how many F each drawn hand
# held from what was enacted -- and the draw pile loses exactly the drawn hand,
# whatever is discarded afterwards. A world's deck state depends only on the
# world and the public events since the last reshuffle, so it is keyed by
# (f0, n0, events) and memoized across observers and games.
DECK_MODEL = "tracked"   # "tracked" (per-world) | "pool" (shared public pool)
_DECK_MEMO = {}


def deck_state(key):
    """key = (f0, n0, events). Returns (dist, lik): dist[f] = P(draw pile holds
    f fascist cards) after the events; lik = P(last event | earlier events).
    Events: ("g", thresh, enacted_f) a government -- F is enacted iff the hand
    holds >= thresh F (nf+1 = both liberal, ..., 1 = all fascist, under the
    greedy-fascist assumption); ("v",) a vetoed hand; ("c", is_f) a chaos card."""
    r = _DECK_MEMO.get(key)
    if r is not None:
        return r
    f0, n0, ev = key
    if not ev:
        dist = [0.0] * (n0 + 1)
        dist[f0] = 1.0
        r = (dist, 1.0)
    else:
        prev = deck_state((f0, n0, ev[:-1]))[0]
        n = len(prev) - 1
        e = ev[-1]
        if e[0] == "c":
            is_f = e[1]
            new = [0.0] * n
            tot = 0.0
            for f, q in enumerate(prev):
                pc = (f if is_f else n - f) / n
                if q and pc:
                    new[f - is_f] += q * pc
                    tot += q * pc
            if tot <= 0:              # this world's deck belief is contradicted
                new, tot = [1.0] * n, 0.0
        else:
            cn = math.comb(n, 3)
            new = [0.0] * (n - 2)
            tot = 0.0
            for f, q in enumerate(prev):
                if not q:
                    continue
                for k in range(4):
                    pk = math.comb(f, k) * math.comb(n - f, 3 - k) / cn
                    if not pk:
                        continue
                    if e[0] == "v":
                        pc = 1.0
                    else:
                        pc = 1 - EPS if (k >= e[1]) == e[2] else EPS
                    new[f - k] += q * pk * pc
                    tot += q * pk * pc
        s = sum(new)
        r = ([x / s for x in new], max(tot, EPS))
    _DECK_MEMO[key] = r
    return r


class Belief:
    """Posterior over the role-assignment worlds for n players, conditioned on
    the owner being liberal. Updated only on observable moves."""

    def __init__(self, owner, n=DEFAULT_N):
        self.n = n
        self.hyps = hyps_for(n)
        self.owner = owner    # None = public belief (no self-knowledge zeroing)
        self.w = [1.0 / len(self.hyps)] * len(self.hyps)
        if owner is not None:
            for idx, (h, fs) in enumerate(self.hyps):
                if owner == h or owner in fs:
                    self.w[idx] = 0.0
        self._renorm()

    def _renorm(self):
        s = sum(self.w)
        if s <= 0:
            self.w = [1.0 / len(self.hyps)] * len(self.hyps)
            if self.owner is not None:
                for idx, (h, fs) in enumerate(self.hyps):
                    if self.owner == h or self.owner in fs:
                        self.w[idx] = 0.0
            self._renorm()
            return
        self.w = [x / s for x in self.w]

    def update_policy(self, pres, chan, enacted_f, df, dl):
        p = hand_probs(df, dl)
        # threshold per gov type: F enacted iff hand holds >= thresh F. The
        # greedy-discard logic is count-independent: both liberal -> only an
        # FFF hand forces F (3); mixed -> 2+ F (2); both fascist -> any F (1).
        table = {}
        for tp in ("L", "F"):
            for tc in ("L", "F"):
                nbad = (tp == "F") + (tc == "F")
                thresh = 3 - nbad
                table[(tp, tc)] = sum(p[k] for k in range(4) if k >= thresh)
        for idx, (h, fs) in enumerate(self.hyps):
            bad = set(fs) | {h}
            tp = "F" if pres in bad else "L"
            tc = "F" if chan in bad else "L"
            base = table[(tp, tc)]
            lk = base if enacted_f else 1.0 - base
            lk = min(max(lk, EPS), 1 - EPS)
            self.w[idx] *= lk
        self._renorm()

    def apply(self, lks):
        """Multiply each world by a precomputed likelihood (per-world deck model)."""
        self.w = [w * l for w, l in zip(self.w, lks)]
        self._renorm()

    def observe_party(self, target, is_fascist):
        for idx, (h, fs) in enumerate(self.hyps):
            if (target == h or target in fs) != is_fascist:
                self.w[idx] = 0.0
        self._renorm()

    def update_nomination(self, pres, deviated, strength):
        """Evidence from the president deviating from the observer's consensus
        chancellor pick. strength: 3 | 9 (soft, recoverable) or 'hard' (label).
        Compliance carries no update (avoids feedback spirals)."""
        if not deviated:
            return
        for idx, (h, fs) in enumerate(self.hyps):
            pres_f = pres == h or pres in fs
            if strength == "hard":
                lk = 1.0 if pres_f else 0.0
            else:
                lk = strength / (strength + 1) if pres_f else 1.0 / (strength + 1)
                lk = min(max(lk, EPS), 1 - EPS)
            self.w[idx] *= lk
        self._renorm()

    def not_hitler(self, seat):
        """Counter A: exact bookkeeping. A seat elected chancellor while the
        board showed >=3F policies and the game continued cannot be Hitler
        (that election would have been the checkmate). Zero those worlds."""
        for idx, (h, fs) in enumerate(self.hyps):
            if h == seat:
                self.w[idx] = 0.0
        self._renorm()

    def update_vote(self, voter, pres, chan, yes):
        """Vote-record evidence (OFF by default -- see VOTE_LK)."""
        for idx, (h, fs) in enumerate(self.hyps):
            voter_f = voter == h or voter in fs
            if not voter_f:
                continue            # liberal-side likelihood: 1 (handled by rule)
            gov_f = (pres == h or pres in fs or chan == h or chan in fs)
            predicted = gov_f       # fascist votes yes iff gov contains a fascist
            lk = VOTE_LK if (yes == predicted) else (1 - VOTE_LK)
            self.w[idx] *= min(max(lk, EPS), 1 - EPS)
        self._renorm()

    def marginals(self):
        n = self.n
        ph = [0.0] * n
        pf = [0.0] * n
        for idx, (h, fs) in enumerate(self.hyps):
            w = self.w[idx]
            ph[h] += w
            pf[h] += w
            for f in fs:
                pf[f] += w
        return ph, pf

    def p_liberal(self, a):
        """P(seat a is liberal): 1 minus the bad marginal."""
        _, pf = self.marginals()
        return 1.0 - pf[a]

    def p_pair_liberal(self, a, b):
        s = 0.0
        for idx, (h, fs) in enumerate(self.hyps):
            if a != h and a not in fs and b != h and b not in fs:
                s += self.w[idx]
        return s


# ---------------- fascist policy play ----------------

def chan_enact(role, style, pair, board_f, deck):
    has_l = "L" in pair
    if role == "L":
        return "L" if has_l else "F"
    if role == "H" and style not in ("greedy", "compliant"):   # clean Hitler
        return "L" if has_l else "F"
    if role == "F" and style == "strategic":
        df, dl = deck
        deniable = df / (df + dl) >= 0.5 or board_f >= 4
        if has_l and not deniable:
            return "L"
        return "F" if "F" in pair else "L"
    return "F" if "F" in pair else "L"             # greedy


def chan_enact_blend(role, pair, deck, board_f):
    """minimax-blend: Hitler grooms (clean), the fascist plays strategic timing."""
    has_l = "L" in pair
    if role == "L":
        return "L" if has_l else "F"
    if role == "H":                        # groom: clean Hitler
        return "L" if has_l else "F"
    df, dl = deck                          # strategic timing for the fascist
    deniable = df / (df + dl) >= 0.5 or board_f >= 4
    if has_l and not deniable:
        return "L"
    return "F" if "F" in pair else "L"


def pres_discard(role, style, hand, deck, board_f):
    """Return (discarded_card, pair_passed)."""
    hand = sorted(hand)
    nf = hand.count("F")
    greedy_like = role == "F" and (
        style in ("greedy", "compliant")
        or (style == "strategic" and
            (deck[0] / (deck[0] + deck[1]) >= 0.5 or board_f >= 4))
    )
    if greedy_like and nf <= 2:      # discard an L, keep the F's
        for i, c in enumerate(hand):
            if c == "L":
                return "L", tuple(hand[:i] + hand[i + 1:])
    if nf >= 1:                      # liberal-like: discard an F
        for i, c in enumerate(hand):
            if c == "F":
                return "F", tuple(hand[:i] + hand[i + 1:])
    return "L", tuple(hand[1:])      # hand was LLL


# ---------------- the game ----------------

class Game:
    def __init__(self, lib, fas, rng, n=DEFAULT_N):
        self.lib = lib        # 'naive' | 'stats1' | 'stats2' | 'stats1-3' | 'stats1-9' | 'stats1-hard'
        self.fas = fas        # 'greedy' | 'strategic' | 'groom' | 'compliant' | 'blend'
        self.n = n
        self.nom_strength = {"stats1-3": 3, "stats1-9": 9,
                             "stats1-hard": "hard"}.get(lib)
        self.nom_strength_g = {"stats1-3g": 3, "stats1-9g": 9,
                               "stats1-hardg": "hard"}.get(lib)
        if self.nom_strength_g:
            self.nom_strength = self.nom_strength_g
        self.rng = rng
        nl, nf, nh = SETUP[n]
        df, dl = DECK_TABLE[n]
        roles = ["L"] * nl + ["F"] * nf + ["H"] * nh
        rng.shuffle(roles)
        self.role = roles
        self.fascists = {i for i, r in enumerate(roles) if r in ("F", "H")}
        self.hitler = roles.index("H")
        self.alive = [True] * n
        self.board_f = 0
        self.board_l = 0
        self.deck = ["F"] * df + ["L"] * dl
        rng.shuffle(self.deck)
        self.discards = []
        self.wkeys = [(df, df + dl, ())] * len(hyps_for(n))  # per-world deck state
        self.fail = 0
        self.term = None            # (pres, chan) of last successful gov
        self.pres = rng.randrange(n)
        self.f_actors = set()       # chancellors who enacted F (naive memory)
        self.beliefs = [Belief(i, n) for i in range(n)]
        # public belief: same evidence, but no self-knowledge -> identical for
        # every observer; the basis of the group-consensus chancellor pick
        self.public = Belief(None, n) if self.nom_strength else None
        # ---- calibrated gates for this count, from the ratio block ----
        # game-start priors measured from the belief machinery itself
        pub0 = Belief(None, n)
        self.pair_pub0 = pub0.p_pair_liberal(0, 1)     # both seats liberal
        ph0, _ = pub0.marginals()
        self.ph_pub0 = ph0[0]                          # 1/n
        priv0 = Belief(0, n)
        self.pair_priv0 = priv0.p_pair_liberal(1, 2)
        _ph, pf0 = priv0.marginals()
        self.ph_priv0 = _ph[1] if self.n > 2 else 0.0  # P(seat 1 Hitler | 0 lib)
        self.vote_t0 = VOTE_R * self.pair_pub0
        self.vote_t3 = VOTE_T3_R * self.vote_t0
        # crown cap: 6p keeps its own swept ratio; at >=2 fascists the tempo
        # race dominates and the checkmate-defense cap over-tightens (swept
        # per count: cap_sweep_9p.txt / cap_sweep_10p.txt -- interior peak 0.80)
        crown_r = CAP_PUB_R[3] if nf == 1 else CAP_PUB_R_MULTI
        self.cap_pub = {0: CAP_PUB_R[0] * self.ph_pub0,
                        3: crown_r * self.ph_pub0}
        self.cap_priv = {b: r * self.ph_priv0 for b, r in CAP_PRIV_R.items()}
        self.priv_t0 = PRIV_VOTE_R * self.pair_priv0
        self.priv_t3 = PRIV_VOTE_T3_R * self.priv_t0
        self.elections = 0
        self.chaos_count = 0
        self.checkmate = False
        self.winner = None
        self.shot_hitler = False
        self.executions = []

    # ---- gates ----
    def cap(self, public):
        """Hitler cap on the chancellor seat: loose below 3F, crown-tight at >=3F."""
        caps = self.cap_pub if public else self.cap_priv
        return caps[0] if self.board_f < 3 else caps[3]

    def vote_t(self, private):
        if private:
            return self.priv_t3 if self.board_f >= 3 else self.priv_t0
        return self.vote_t3 if self.board_f >= 3 else self.vote_t0

    # ---- helpers ----
    def alive_list(self):
        return [i for i in range(self.n) if self.alive[i]]

    def next_alive(self, i):
        j = i
        while True:
            j = (j + 1) % self.n
            if self.alive[j]:
                return j

    def reshuffle(self):
        self.deck += self.discards
        self.discards = []
        self.rng.shuffle(self.deck)
        # the new pile is every card not on the board: known exactly again
        df, _dl = DECK_TABLE[self.n]
        self.wkeys = [(df - self.board_f, len(self.deck), ())] * len(hyps_for(self.n))

    def draw(self, k):
        if len(self.deck) < k:
            self.reshuffle()
        got = self.deck[:k]
        self.deck = self.deck[k:]
        return got

    def deck_event(self, make_event, reweight):
        """Advance every world's deck state by one public event; if reweight,
        fold that world's likelihood of the event into all liberal beliefs."""
        hyps = hyps_for(self.n)
        lks = [0.0] * len(hyps)
        for idx, (h, fs) in enumerate(hyps):
            f0, n0, ev = self.wkeys[idx]
            key = (f0, n0, ev + (make_event(h, fs),))
            self.wkeys[idx] = key
            lks[idx] = deck_state(key)[1]
        if reweight:
            for i in range(self.n):
                if self.role[i] == "L":
                    self.beliefs[i].apply(lks)
            if self.public is not None:
                self.public.apply(lks)

    def eligible(self):
        t = self.term or ()
        return [i for i in self.alive_list()
                if i != self.pres and i not in t]

    def consensus_pick(self, observer):
        """Group consensus: the argmax-chancellor from the PUBLIC belief --
        identical for every observer. Falls back to the observer's private
        array only if no public belief exists (legacy variants).
        Note: with a public belief the president MAY query this (it is
        computable from public information by anyone, including them)."""
        if observer == self.pres and self.public is None:
            return None
        if self.public is not None:
            b, who = self.public, None
            ph, _ = b.marginals()
            cands = self.eligible()
            if not cands:
                cands = [i for i in self.alive_list() if i != self.pres]
            cap = self.cap(public=True)
            scored = sorted(cands, key=lambda c: b.p_pair_liberal(self.pres, c),
                            reverse=True)
            ok = [c for c in scored if ph[c] <= cap]
            if not ok:
                ok = sorted(cands, key=lambda c: ph[c])
            return ok[0]
        # legacy: private-array judgment (kept for the earlier variants)
        if self.lib == "naive":
            return None
        cands = self.eligible()
        if not cands:
            cands = [i for i in self.alive_list() if i != self.pres]
        b = self.beliefs[observer]
        ph, _ = b.marginals()
        cap = self.cap(public=False)
        scored = sorted(cands, key=lambda c: b.p_pair_liberal(observer, c),
                        reverse=True)
        ok = [c for c in scored if ph[c] <= cap]
        if not ok:
            ok = sorted(cands, key=lambda c: ph[c])
        return ok[0]

    # ---- decisions ----
    def nominate(self):
        pres = self.pres
        cands = self.eligible()
        if not cands:                       # degenerate: everyone term-limited
            cands = [i for i in self.alive_list() if i != pres]
        role = self.role[pres]

        if role in ("F", "H"):
            # compliant/blend doctrine: nominate the public consensus pick to
            # avoid the deviation tell (progress must come from enactments)
            if self.fas in ("compliant", "compliant-mix", "blend") and self.public is not None:
                obs = next((i for i in self.alive_list() if i != pres), None)
                consensus = self.consensus_pick(obs) if obs is not None else None
                if consensus is not None and (
                        self.fas == "compliant" or self.rng.random() < 0.7):
                    return consensus
            if self.board_f >= 3 and self.hitler in cands:
                return self.hitler          # go for the checkmate
            for i in cands:                 # otherwise a co-fascist
                if i in self.fascists:
                    return i
            return self.rng.choice(cands)

        if self.lib == "naive":
            j = pres
            for _ in range(self.n):
                j = self.next_alive(j)
                if j in cands:
                    return j
            return cands[0]

        # stats liberal nomination: top-k by P(self & cand liberal), Hitler-capped
        # hybrid: at >=3F the crown is a checkmate weapon -> trust-blind rotation
        if self.lib == "hybrid" and self.board_f >= 3:
            j = pres
            for _ in range(self.n):
                j = self.next_alive(j)
                if j in cands:
                    return j
            return cands[0]
        # LIB_COMPLY: liberal presidents also nominate the public-consensus pick
        # (silences the deviation channel entirely -- no innocent deviations).
        # Costs the president's self-knowledge: their private P(pair liberal)
        # conditions on their own seat and beats the public belief's.
        if LIB_COMPLY and self.public is not None:
            consensus = self.consensus_pick(pres)
            if consensus is not None:
                ph_pub, _ = self.public.marginals()
                if ph_pub[consensus] <= self.cap(public=True):
                    return consensus
        b = self.beliefs[pres]
        ph, _ = b.marginals()
        cap = self.cap(public=False)
        scored = sorted(cands, key=lambda c: b.p_pair_liberal(pres, c), reverse=True)
        ok = [c for c in scored if ph[c] <= cap]
        if not ok:
            ok = sorted(cands, key=lambda c: ph[c])   # least-Hitler-ish
        k = 2 if self.lib == "stats2" else 1
        return self.rng.choice(ok[:k])

    def vote(self, voter, pres, chan):
        role = self.role[voter]
        if role in ("F", "H"):
            gov = {pres, chan}
            return bool(gov & self.fascists)
        if self.lib == "naive":
            return pres not in self.f_actors and chan not in self.f_actors
        # LIB_VOTE_COMPLY: all liberals cast the SAME vote derived from the
        # public belief (a voting block). Private arrays stay for nominations,
        # executions, investigations -- only the roll-call is unified.
        if LIB_VOTE_COMPLY and self.public is not None:
            ph_pub, _ = self.public.marginals()
            t = self.vote_t(private=False)
            return (self.public.p_pair_liberal(pres, chan) >= t
                    and ph_pub[chan] <= self.cap(public=True))
        b = self.beliefs[voter]
        ph, pf = b.marginals()
        # PRIVATE path (brains without a public belief): a public-path gate
        # deadlocks here -- self-knowledge zeroing lowers private pair scores
        # (see HOW_IT_WORKS row 18). Private votes keep their own calibration.
        t = self.vote_t(private=True)
        joint = b.p_pair_liberal(pres, chan)
        return joint >= t and ph[chan] <= self.cap(public=False)

    def investigate_target(self, pres):
        if self.lib == "naive":
            pool = [i for i in self.alive_list() if i != pres]
            return self.rng.choice(pool)
        b = self.beliefs[pres]
        _, pf = b.marginals()
        pool = [i for i in self.alive_list() if i != pres]
        return max(pool, key=lambda i: (pf[i], self.rng.random()))

    def special_elect(self, pres):
        if self.role[pres] in ("F", "H"):
            # compliant/blend doctrine: special-elect the public consensus pick
            # to avoid the SE deviation tell (mirrors the nomination doctrine).
            # Gated on SE_TELL: without the tell there is nothing to hide from,
            # and the compliance play itself perturbs the calibrated 6p path.
            if SE_TELL and self.fas in ("compliant", "blend") and self.public is not None:
                consensus = self.consensus_pick_se(pres)
                if consensus is not None:
                    return consensus
            if self.hitler in self.alive_list() and self.hitler != pres:
                return self.hitler
            for i in self.alive_list():
                if i != pres and i in self.fascists:
                    return i
            return self.next_alive(pres)
        if self.lib == "naive":
            return self.next_alive(pres)
        # LIB_COMPLY: liberal presidents also special-elect the consensus pick
        # (silences the innocent-deviation channel, same reason as nominations)
        # Gated on SE_TELL like the fascist side (one flag owns the channel).
        if SE_TELL and LIB_COMPLY and self.public is not None:
            consensus = self.consensus_pick_se(pres)
            if consensus is not None:
                ph_pub, _ = self.public.marginals()
                if ph_pub[consensus] <= self.cap(public=True):
                    return consensus
        b = self.beliefs[pres]
        pool = [i for i in self.alive_list() if i != pres]
        return max(pool, key=lambda i: (b.p_pair_liberal(pres, i), self.rng.random()))

    def consensus_pick_se(self, pres):
        """Consensus pick for the special-election channel: the public belief's
        best next-president. Distinct from consensus_pick() because the pairwise
        score differs (the pick becomes president, not chancellor)."""
        if self.public is None:
            return None
        ph, _ = self.public.marginals()
        cands = [i for i in self.alive_list() if i != pres]
        if not cands:
            return None
        cap = self.cap(public=True)
        scored = sorted(cands, key=lambda c: self.public.p_liberal(c),
                        reverse=True)
        ok = [c for c in scored if ph[c] <= cap]
        if not ok:
            ok = sorted(cands, key=lambda c: ph[c])
        return ok[0]

    def execution_target(self, pres):
        pool = [i for i in self.alive_list() if i != pres]
        if self.role[pres] in ("F", "H") or self.lib == "naive":
            return self.rng.choice(pool)
        b = self.beliefs[pres]
        ph, _ = b.marginals()
        return max(pool, key=lambda i: (ph[i], self.rng.random()))

    def veto(self, pair, deck):
        """Returns True if veto proposed AND accepted."""
        if self.board_f < 5:
            return False
        chan, pres = self.last_chan, self.last_pres
        cr, pr = self.role[chan], self.role[pres]
        # chancellor proposes
        if cr == "L" or (cr == "H" and self.fas not in ("greedy", "compliant")):
            propose = "L" not in pair      # liberal: veto a forced 6th F (FF)
        elif cr == "F" and self.fas == "strategic":
            propose = "F" not in pair      # fascist: veto a forced L (LL)
        else:
            propose = False
        if not propose:
            return False
        # president accepts
        if pr == "L":
            return True
        return chan in self.fascists   # fascist president backs fascist chancellor

    # ---- main loop ----
    def run(self):
        while self.elections < 500:
            self.elections += 1
            chan = self.nominate()

            # nomination-consensus evidence: one group consensus from the public
            # belief; the president either nominated it or deviated. Skip when
            # the consensus pick itself exceeded the cap (cap fallback): there
            # was no legitimate consensus to deviate from, and penalizing a
            # president for ignoring an illegal consensus punishes innocence.
            if self.nom_strength:
                obs = next((i for i in self.alive_list() if i != self.pres), None)
                consensus = self.consensus_pick(obs) if obs is not None else None
                if consensus is not None:
                    ph_pub, _ = self.public.marginals()
                    deviated = ((chan != consensus)
                                and ph_pub[consensus] <= self.cap(public=True))
                    if deviated:
                        # public event: every liberal folds it into their own
                        # posterior (and the public array stays current)
                        for i in range(self.n):
                            if self.role[i] == "L":
                                self.beliefs[i].update_nomination(
                                    self.pres, True, self.nom_strength)
                        self.public.update_nomination(self.pres, True,
                                                      self.nom_strength)

            votes = [self.vote(i, self.pres, chan) for i in self.alive_list()]
            # vote-record evidence (public roll-call): each liberal folds every
            # OTHER voter's roll-call vote into their posterior + the public
            # array. Only fascist-side likelihoods move (liberal-side voting
            # behavior is the voters' own rule -- using it would feed back).
            if VOTE_LK:
                for k, j in enumerate(self.alive_list()):
                    for i in range(self.n):
                        if self.role[i] == "L" and i != j:
                            self.beliefs[i].update_vote(j, self.pres, chan,
                                                        votes[k])
                    if self.public is not None:
                        self.public.update_vote(j, self.pres, chan, votes[k])
            yes = sum(votes)
            alive_n = len(self.alive_list())
            self.last_pres, self.last_chan = self.pres, chan

            if yes > alive_n // 2:
                # checkmate?
                if self.role[chan] == "H" and self.board_f >= 3:
                    self.checkmate = True
                    return "F"
                # counter A: this government passed at >=3F with a non-Hitler
                # chancellor (else the checkmate fired above) -> exact update
                if self.board_f >= 3:
                    for i in range(self.n):
                        if self.role[i] == "L":
                            self.beliefs[i].not_hitler(chan)
                    if self.public is not None:
                        self.public.not_hitler(chan)
                self.resolve_government(chan)
                if self.winner:
                    return self.winner
            else:
                self.fail += 1
                self.term = None
                if self.fail >= 3:
                    self.chaos()
                    if self.winner:
                        return self.winner

            self.pres = self.next_alive(self.pres)
        return "?"  # safety

    def resolve_government(self, chan):
        pres = self.pres
        if len(self.deck) < 3:
            self.reshuffle()
        # PUBLIC pool: cards not on the board (draw pile + discard pile). Used by
        # the fascist bots' deniability heuristic and by liberals only in "pool"
        # mode; in "tracked" mode liberals use each world's own pile estimate
        # (deck_event / deck_state).
        deck_before = (DECK_TABLE[self.n][0] - self.board_f,
                       DECK_TABLE[self.n][1] - self.board_l)
        hand = self.draw(3)
        discard, pair = pres_discard(self.role[pres], self.fas, hand,
                                     deck_before, self.board_f)
        self.discards.append(discard)

        if self.veto(pair, deck_before):
            self.deck_event(lambda h, fs: ("v",), reweight=False)
            self.discards += list(pair)
            self.fail += 1
            self.term = None
            if self.fail >= 3:
                self.chaos()
            return

        enacted = (chan_enact_blend(self.role[chan], pair, deck_before,
                                    self.board_f)
                   if self.fas == "blend"
                   else chan_enact(self.role[chan], self.fas, pair,
                                   self.board_f, deck_before))
        self.discards.append(pair[0] if pair[1] == enacted else pair[1])

        self.board_f += enacted == "F"
        self.board_l += enacted == "L"
        if enacted == "F":
            self.f_actors.add(chan)
        # Bayesian update (liberals only)
        if DECK_MODEL == "tracked":
            ef = enacted == "F"
            self.deck_event(
                lambda h, fs: ("g", 3 - ((pres == h or pres in fs)
                                         + (chan == h or chan in fs)), ef),
                reweight=True)
        else:
            for i in range(self.n):
                if self.role[i] == "L":
                    self.beliefs[i].update_policy(pres, chan, enacted == "F",
                                                  deck_before[0], deck_before[1])
            if self.public is not None:
                self.public.update_policy(pres, chan, enacted == "F",
                                          deck_before[0], deck_before[1])
        # win by policy count
        if self.board_f >= POLICY_F_WIN:
            self.winner = "F"
            return
        if self.board_l >= POLICY_L_WIN:
            self.winner = "L"
            return
        self.term = (pres, chan)
        # executive powers (never on chaos)
        if enacted == "F":
            self.powers(pres)

    def chaos(self):
        card = self.draw(1)[0]
        # the chaos card's colour is evidence too: each world predicts it from
        # its own deck estimate (pool model treats it as uninformative)
        self.deck_event(lambda h, fs: ("c", card == "F"),
                        reweight=DECK_MODEL == "tracked")
        self.board_f += card == "F"
        self.board_l += card == "L"
        self.fail = 0
        self.chaos_count += 1
        if self.board_f >= POLICY_F_WIN:
            self.winner = "F"
        elif self.board_l >= POLICY_L_WIN:
            self.winner = "L"

    def powers(self, pres):
        power = POWERS[self.n].get(self.board_f)
        if power == "inv":                       # investigate
            t = self.investigate_target(pres)
            if self.role[pres] == "L":
                self.beliefs[pres].observe_party(t, t in self.fascists)
        elif power == "pe":                      # special election
            pick = self.special_elect(pres)
            # SE deviation tell: the same logic as the nomination tell, on the
            # special-election pick (the pick becomes president publicly).
            # Skipped when the consensus pick itself exceeded the cap -- no
            # doable reference action (the row-16 lesson).
            if SE_TELL and self.nom_strength:
                consensus = self.consensus_pick_se(pres)
                if consensus is not None:
                    ph_pub, _ = self.public.marginals()
                    if ph_pub[consensus] <= self.cap(public=True) and pick != consensus:
                        for i in range(self.n):
                            if self.role[i] == "L":
                                self.beliefs[i].update_nomination(
                                    pres, True, self.nom_strength)
                        self.public.update_nomination(pres, True, self.nom_strength)
            self.pres = pick
            self.pres_override = True
        elif power == "ex":                      # execution
            t = self.execution_target(pres)
            self.alive[t] = False
            self.executions.append((pres, t))
            if t == self.hitler:
                self.shot_hitler = True
                self.winner = "L"


def run(lib, fas, n_games, seed=1234, n=DEFAULT_N):
    rng = random.Random(seed)
    stats = Counter()
    for _ in range(n_games):
        g = Game(lib, fas, rng, n)
        w = g.run()
        stats["games"] += 1
        if w == "L":
            stats["lib_win"] += 1
            stats["lib_win_shot"] += g.shot_hitler
        stats["checkmate"] += g.checkmate
        stats["shot_hitler"] += g.shot_hitler
        stats["elections"] += g.elections
        stats["chaos"] += g.chaos_count
        stats["executions"] += len(g.executions)
    n_g = stats["games"]
    return {
        "lib_win%": 100 * stats["lib_win"] / n_g,
        "checkmate%": 100 * stats["checkmate"] / n_g,
        "shot_H%": 100 * stats["shot_hitler"] / n_g,
        "avg_elections": stats["elections"] / n_g,
        "avg_chaos": stats["chaos"] / n_g,
        "avg_executions": stats["executions"] / n_g,
    }


if __name__ == "__main__":
    import sys
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    n_players = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    if len(sys.argv) > 3:            # optional: "pool" or "tracked"
        DECK_MODEL = sys.argv[3]
    libs = ["naive", "stats1", "stats1-9g", "hybrid", "stats2"]
    fass = ["greedy", "strategic", "groom", "compliant", "blend"]
    hdr = (f"{'liberals':<10}{'fascists':<11}{'lib win%':>9}{'checkmate%':>12}"
           f"{'shot_H%':>9}{'elections':>11}{'chaos':>8}{'execs':>8}")
    print(f"--- {n_players} players, {n_games} games ---")
    print(hdr)
    print("-" * len(hdr))
    for fas in fass:
        for lib in libs:
            r = run(lib, fas, n_games, n=n_players)
            print(f"{lib:<10}{fas:<11}{r['lib_win%']:>8.1f}{r['checkmate%']:>11.1f}"
                  f"{r['shot_H%']:>9.1f}{r['avg_elections']:>11.1f}{r['avg_chaos']:>8.2f}"
                  f"{r['avg_executions']:>8.2f}")
