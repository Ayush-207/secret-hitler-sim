#!/usr/bin/env python3
"""
Secret Hitler 6-player Monte Carlo.

Question: does a purely statistical trust model (Bayesian posteriors over role
assignments, updated ONLY on observable moves -- enacted policies; no claims,
no speech) raise the liberals' win rate above a naive baseline, and how does
the answer degrade against different fascist play styles?

Assumptions:
- 6 players: 4 liberals, 1 fascist, 1 Hitler. 5-6p board:
  F2 investigate, F3 special election, F4 execution, F5 execution + veto unlock.
- Deck: 11F / 6L. Discards are PRIVATE (as in the real game): players observe
  enacted policies, nominations, and votes — never the discarded card's color.
  The pile's card COUNT is public (matters only for reshuffles).
- Chaos (3 failed elections) enacts the top card; no executive power fires.
- Term limits: the president+chancellor of the last successful government
  cannot be chancellor candidates in the immediately following election.
- Hitler-chancellor checkmate: vote passing with Hitler as chancellor and
  >=3F on the board = instant fascist win.
- The liberal belief model ASSUMES fascists are greedy (enact F whenever the
  hand allows). Real sim fascists may deviate -> deliberate misspecification;
  measuring robustness to it is the point of the experiment.
"""

import math
import random
from collections import Counter

N = 6
DECK_F, DECK_L = 11, 6
HYPS = [(h, f) for h in range(N) for f in range(N) if h != f]  # (hitler, fascist)
NH = len(HYPS)
EPS = 0.03  # likelihood smoothing floor/ceiling


def hand_probs(df, dl):
    """P(k fascists in a 3-card draw) for k=0..3."""
    tot = math.comb(df + dl, 3)
    return [math.comb(df, k) * math.comb(dl, 3 - k) / tot for k in range(4)]


class Belief:
    """Posterior over the 30 ordered (Hitler, fascist) assignments, conditioned
    on the owner being liberal. Updated only on policy enactments."""

    def __init__(self, owner):
        self.owner = owner    # None = public belief (no self-knowledge zeroing)
        self.w = [1.0 / NH] * NH
        if owner is not None:
            for idx, (h, f) in enumerate(HYPS):
                if owner in (h, f):
                    self.w[idx] = 0.0
        self._renorm()

    def _renorm(self):
        s = sum(self.w)
        if s <= 0:
            self.w = [1.0 / NH] * NH
            if self.owner is not None:
                for idx, (h, f) in enumerate(HYPS):
                    if self.owner in (h, f):
                        self.w[idx] = 0.0
            self._renorm()
            return
        self.w = [x / s for x in self.w]

    def update_policy(self, pres, chan, enacted_f, df, dl):
        p = hand_probs(df, dl)
        table = {
            ("L", "L"): p[3],
            ("L", "F"): p[3] + p[2],
            ("F", "L"): p[3] + p[2],
            ("F", "F"): 1.0 - p[0],  # FF gov is forced to pass L only on an LLL draw
        }
        for idx, (h, f) in enumerate(HYPS):
            fset = {h, f}
            tp = "F" if pres in fset else "L"
            tc = "F" if chan in fset else "L"
            base = table[(tp, tc)]
            lk = base if enacted_f else 1.0 - base
            lk = min(max(lk, EPS), 1 - EPS)
            self.w[idx] *= lk
        self._renorm()

    def observe_party(self, target, is_fascist):
        for idx, (h, f) in enumerate(HYPS):
            if (target in (h, f)) != is_fascist:
                self.w[idx] = 0.0
        self._renorm()

    def update_nomination(self, pres, deviated, strength):
        """Evidence from the president deviating from the observer's consensus
        chancellor pick. strength: 3 | 9 (soft, recoverable) or 'hard' (label).
        Compliance carries no update (avoids feedback spirals)."""
        if not deviated:
            return
        for idx, (h, f) in enumerate(HYPS):
            pres_f = pres in (h, f)
            if strength == "hard":
                lk = 1.0 if pres_f else 0.0
            else:
                lk = strength / (strength + 1) if pres_f else 1.0 / (strength + 1)
                lk = min(max(lk, EPS), 1 - EPS)
            self.w[idx] *= lk
        self._renorm()

    def marginals(self):
        ph = [0.0] * N
        pf = [0.0] * N
        for idx, (h, f) in enumerate(HYPS):
            w = self.w[idx]
            ph[h] += w
            pf[h] += w
            pf[f] += w
        return ph, pf

    def p_pair_liberal(self, a, b):
        s = 0.0
        for idx, (h, f) in enumerate(HYPS):
            if a not in (h, f) and b not in (h, f):
                s += self.w[idx]
        return s


# ---------------- fascist policy play ----------------

def chan_enact(role, style, pair, board_f, deck):
    has_l = "L" in pair
    if role == "L":
        return "L" if has_l else "F"
    if role == "H" and style != "greedy":          # clean Hitler
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
        style == "greedy"
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
    def __init__(self, lib, fas, rng):
        self.lib = lib        # 'naive' | 'stats1' | 'stats2' | 'stats1-3' | 'stats1-9' | 'stats1-hard'
        self.fas = fas        # 'greedy' | 'strategic' | 'groom'
        self.nom_strength = {"stats1-3": 3, "stats1-9": 9,
                             "stats1-hard": "hard"}.get(lib)
        self.nom_strength_g = {"stats1-3g": 3, "stats1-9g": 9,
                               "stats1-hardg": "hard"}.get(lib)
        if self.nom_strength_g:
            self.nom_strength = self.nom_strength_g
        self.rng = rng
        roles = ["L"] * 4 + ["F", "H"]
        rng.shuffle(roles)
        self.role = roles
        self.fascists = {i for i, r in enumerate(roles) if r in ("F", "H")}
        self.hitler = roles.index("H")
        self.alive = [True] * N
        self.board_f = 0
        self.board_l = 0
        self.deck = ["F"] * DECK_F + ["L"] * DECK_L
        rng.shuffle(self.deck)
        self.discards = []
        self.fail = 0
        self.term = None            # (pres, chan) of last successful gov
        self.pres = rng.randrange(N)
        self.f_actors = set()       # chancellors who enacted F (naive memory)
        self.beliefs = [Belief(i) for i in range(N)]
        # public belief: same evidence, but no self-knowledge -> identical for
        # every observer; the basis of the group-consensus chancellor pick
        self.public = Belief(None) if self.nom_strength else None
        self.elections = 0
        self.chaos_count = 0
        self.checkmate = False
        self.winner = None
        self.shot_hitler = False
        self.executions = []

    # ---- helpers ----
    def alive_list(self):
        return [i for i in range(N) if self.alive[i]]

    def next_alive(self, i):
        j = i
        while True:
            j = (j + 1) % N
            if self.alive[j]:
                return j

    def draw(self, k):
        if len(self.deck) < k:
            self.deck += self.discards
            self.discards = []
            self.rng.shuffle(self.deck)
        got = self.deck[:k]
        self.deck = self.deck[k:]
        return got

    def eligible(self):
        t = self.term or ()
        return [i for i in self.alive_list()
                if i != self.pres and i not in t]

    def consensus_pick(self, observer):
        """Group consensus: the argmax-chancellor from the PUBLIC belief —
        identical for every observer. Falls back to the observer's private
        array only if no public belief exists (legacy variants)."""
        if observer == self.pres:
            return None
        if self.public is not None:
            b, who = self.public, None
            ph, _ = b.marginals()
            cands = self.eligible()
            if not cands:
                cands = [i for i in self.alive_list() if i != self.pres]
            cap = 0.40 if self.board_f < 3 else 0.25
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
        cap = 0.40 if self.board_f < 3 else 0.25
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
            for _ in range(N):
                j = self.next_alive(j)
                if j in cands:
                    return j
            return cands[0]

        # stats liberal nomination: top-k by P(self & cand liberal), Hitler-capped
        # hybrid: at >=3F the crown is a checkmate weapon -> trust-blind rotation
        if self.lib == "hybrid" and self.board_f >= 3:
            j = pres
            for _ in range(N):
                j = self.next_alive(j)
                if j in cands:
                    return j
            return cands[0]
        b = self.beliefs[pres]
        ph, _ = b.marginals()
        cap = 0.40 if self.board_f < 3 else 0.25
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
        b = self.beliefs[voter]
        ph, pf = b.marginals()
        cap = 0.40 if self.board_f < 3 else 0.25
        t = {0: 0.28, 1: 0.28, 2: 0.30, 3: 0.40}.get(self.board_f, 0.40)
        joint = b.p_pair_liberal(pres, chan)
        return joint >= t and ph[chan] <= cap

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
            if self.hitler in self.alive_list() and self.hitler != pres:
                return self.hitler
            for i in self.alive_list():
                if i != pres and i in self.fascists:
                    return i
            return self.next_alive(pres)
        if self.lib == "naive":
            return self.next_alive(pres)
        b = self.beliefs[pres]
        pool = [i for i in self.alive_list() if i != pres]
        return max(pool, key=lambda i: (b.p_pair_liberal(pres, i), self.rng.random()))

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
        if cr == "L" or (cr == "H" and self.fas != "greedy"):
            propose = "F" not in pair
        elif cr == "F" and self.fas == "strategic":
            propose = "L" not in pair
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
            # belief; the president either nominated it or deviated
            if self.nom_strength:
                obs = next((i for i in self.alive_list() if i != self.pres), None)
                consensus = self.consensus_pick(obs) if obs is not None else None
                if consensus is not None:
                    deviated = (chan != consensus)
                    if deviated:
                        # public event: every liberal folds it into their own
                        # posterior (and the public array stays current)
                        for i in range(N):
                            if self.role[i] == "L":
                                self.beliefs[i].update_nomination(
                                    self.pres, True, self.nom_strength)
                        self.public.update_nomination(self.pres, True,
                                                      self.nom_strength)

            votes = [self.vote(i, self.pres, chan) for i in self.alive_list()]
            yes = sum(votes)
            alive_n = len(self.alive_list())
            self.last_pres, self.last_chan = self.pres, chan

            if yes > alive_n // 2:
                # checkmate?
                if self.role[chan] == "H" and self.board_f >= 3:
                    self.checkmate = True
                    return "F"
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
            self.deck += self.discards
            self.discards = []
            self.rng.shuffle(self.deck)
        deck_before = (self.deck.count("F"), self.deck.count("L"))
        hand = self.draw(3)
        discard, pair = pres_discard(self.role[pres], self.fas, hand,
                                     deck_before, self.board_f)
        self.discards.append(discard)

        if self.veto(pair, deck_before):
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
        for i in range(N):
            if self.role[i] == "L":
                self.beliefs[i].update_policy(pres, chan, enacted == "F",
                                              deck_before[0], deck_before[1])
        if self.public is not None:
            self.public.update_policy(pres, chan, enacted == "F",
                                      deck_before[0], deck_before[1])
        # win by policy count
        if self.board_f >= 6:
            self.winner = "F"
            return
        if self.board_l >= 5:
            self.winner = "L"
            return
        self.term = (pres, chan)
        # executive powers (never on chaos)
        if enacted == "F":
            self.powers(pres)

    def chaos(self):
        card = self.draw(1)[0]
        self.board_f += card == "F"
        self.board_l += card == "L"
        self.fail = 0
        self.chaos_count += 1
        if self.board_f >= 6:
            self.winner = "F"
        elif self.board_l >= 5:
            self.winner = "L"

    def powers(self, pres):
        if self.board_f == 2:                       # investigate
            t = self.investigate_target(pres)
            if self.role[pres] == "L":
                self.beliefs[pres].observe_party(t, t in self.fascists)
        elif self.board_f == 3:                     # special election
            self.pres = self.special_elect(pres)
            self.pres_override = True
        elif self.board_f in (4, 5):                # execution
            t = self.execution_target(pres)
            self.alive[t] = False
            self.executions.append((pres, t))
            if t == self.hitler:
                self.shot_hitler = True
                self.winner = "L"


def run(lib, fas, n_games, seed=1234):
    rng = random.Random(seed)
    stats = Counter()
    for _ in range(n_games):
        g = Game(lib, fas, rng)
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
    n = stats["games"]
    return {
        "lib_win%": 100 * stats["lib_win"] / n,
        "checkmate%": 100 * stats["checkmate"] / n,
        "shot_H%": 100 * stats["shot_hitler"] / n,
        "avg_elections": stats["elections"] / n,
        "avg_chaos": stats["chaos"] / n,
        "avg_executions": stats["executions"] / n,
    }


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    libs = ["stats1-9g"]
    fass = ["compliant", "greedy", "groom", "blend"]
    hdr = f"{'liberals':<10}{'fascists':<11}{'lib win%':>9}{'checkmate%':>12}{'shot_H%':>9}{'elections':>11}{'chaos':>8}{'execs':>8}"
    print(hdr)
    print("-" * len(hdr))
    for fas in fass:
        for lib in libs:
            r = run(lib, fas, n)
            print(f"{lib:<10}{fas:<11}{r['lib_win%']:>8.1f}{r['checkmate%']:>11.1f}"
                  f"{r['shot_H%']:>9.1f}{r['avg_elections']:>11.1f}{r['avg_chaos']:>8.2f}"
                  f"{r['avg_executions']:>8.2f}")
