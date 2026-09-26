# Secret Hitler Trust-Model Simulation

A Monte Carlo simulator testing whether liberals in a 6-player Secret Hitler game
can win more often with a purely statistical trust model — Bayesian posteriors
over role assignments, updated only on observable moves (enacted policies,
nominations), ignoring all table talk.

**Headline result:** liberals win ≥70% against every fascist doctrine tested,
including ones designed with knowledge of the liberal model. The biggest single
lever is reading nomination behavior — see [ANALYSIS.md](ANALYSIS.md).

## Usage

```bash
python3 sim.py 10000    # full grid of liberal × fascist strategies, 10k games/cell
```

Python 3 stdlib only. Results print as a table (lib win%, checkmate%, shot-Hitler%,
elections, chaos, executions). Canonical rerun: `results_full_grid.txt`.
