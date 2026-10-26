# IQCP benchmark – computational section of the manuscript

Our algorithms are compared with Gurobi on an exact model of the full IQCP.

```
iqcp/instance.py       instances and random generators (rrt, cat, *-hc, catA2, star)
iqcp/pwl.py            piecewise-linear functions + tree covering DP
iqcp/algorithms.py     Algorithms 1-4 of the paper
iqcp/gurobi_model.py   exact MIQCP model of the full IQCP, solved by Gurobi
run_bench.py           the experiments, results go to results/*.jsonl
make_tables.py         results -> latex/ (tables, figure)
run_all.py             runs everything (checks, experiments, tables, values)
tests/validate_lp.py   independent LP cross-check of the algorithms
latex_src/computational_results_plain.tex   the section, with ?? to fill in
```

## Installation (once)

```bash
pip install numpy scipy matplotlib gurobipy
grbgetkey <your-academic-key>          # activate the Gurobi licence
```

## Run

In PyCharm: right-click **run_all.py** -> **Run** (or `python run_all.py`).
Settings (time limit, number of instances) are at the top of `run_all.py`.

1. checks: the algorithms against LP and against Gurobi on small trees
   (must end with `SELF-CHECK PASSED`, otherwise it stops);
2. experiments: scaling of the algorithms, accuracy eps, ours vs Gurobi
   for n = 17 ... 2049 (larger sizes are skipped once Gurobi times out on all
   runs of a size);
3. output: `VALUES_FOR_PAPER.txt` (the values for the ?? in
   `computational_results_plain.tex`, in order, plus the table rows) and
   `latex/iqcp_scaling.pdf` (the figure).

Interrupted? Run it again: finished runs in `results/` are kept.
Run everything on one machine with nothing else heavy running.
