#!/usr/bin/env python3
"""
ONE script that produces every number of Section 5 ("Computational results").

How to use (PyCharm):  open the project, right-click  run_all.py  ->  Run.
No command-line arguments are needed; change the settings below if you wish.

What it does, in order:
  1. checks      : the algorithms against LP and against Gurobi on small
                   trees -- stops if anything disagrees;
  2. experiments : scaling, eps and the comparison with Gurobi
                   (results are saved in results/*.jsonl; an interrupted run
                   continues where it stopped when you start it again);
  3. outputs     : latex/ (tables, figure)  and  VALUES_FOR_PAPER.txt, which
                   lists every value to type into computational_results_plain.tex
                   in the order in which the bold ?? appear, followed by the
                   rows of the two tables.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import time

# ============================================================== SETTINGS
TIME_LIMIT = 600          # seconds per Gurobi run
REPS_SOLVERS = 5          # instances per (family, size) in the solver comparison
REPS_SCALING = 10         # instances per (family, size) in the scaling study
SKIP_CHECKS = False       # set True only if the checks already passed on this machine
# ======================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import run_bench  # noqa: E402
import make_tables  # noqa: E402


def banner(msg):
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72, flush=True)


def args_for(part, **kw):
    a = argparse.Namespace(part=part, reps=None, sizes=None, families=None, solvers=None,
                           max_n=None, time_limit=TIME_LIMIT, threads=1,
                           skip_after_timeout=True, n=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def gurobi_ok():
    try:
        import gurobipy as gp
        with contextlib.redirect_stdout(io.StringIO()):
            m = gp.Model()
            x = m.addVars(2100)                 # larger than the free size-limited licence
            m.addConstr(gp.quicksum(x.values()) >= 1)
            m.Params.OutputFlag = 0
            m.optimize()
        return True
    except ImportError:
        print("gurobipy is not installed (pip install gurobipy).")
    except Exception as e:                       # licence problems end up here
        print(f"Gurobi cannot be used: {e}")
    return False


def run_checks():
    banner("1/3  CHECKS")
    import subprocess
    for mod, arg in (("tests.validate_lp", "20"),):
        print(f"-- python -m {mod} {arg}", flush=True)
        out = subprocess.run([sys.executable, "-m", mod, arg], capture_output=True, text=True)
        last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr
        print("   " + last)
        bad = re.search(r"(\d+) (failures|violations|with discrepancies)", last)
        if out.returncode != 0 or (bad and int(bad.group(1)) > 0):
            sys.exit(f"CHECK FAILED in {mod}. Stop here and send the output above.")
    print("-- selfcheck (algorithms vs Gurobi)", flush=True)
    run_bench.cmd_selfcheck(args_for("selfcheck", reps=3, time_limit=120))              # exits with an error message on a mismatch


def run_experiments():
    banner("2/3  EXPERIMENTS (results/*.jsonl; safe to interrupt and restart)")
    run_bench.cmd_env(args_for("env"))
    t0 = time.time()
    print("\n-- scaling of our algorithms", flush=True)
    run_bench.cmd_scaling(args_for("scaling", reps=REPS_SCALING))
    print("\n-- dependence on epsilon", flush=True)
    run_bench.cmd_eps(args_for("eps", reps=10))
    print("\n-- ours vs Gurobi", flush=True)
    run_bench.cmd_solvers(args_for("solvers", reps=REPS_SOLVERS))
    print(f"\nexperiments finished in {(time.time() - t0) / 3600:.2f} h")


# order of the bold ?? in computational_results_plain.tex
FILL_ORDER = [
    ("RPython", "Python version"), ("RCpu", "CPU"), ("RRam", "memory in GB"),
    ("RGurobi", "Gurobi version"), ("RTimeLimit", "time limit [s]"),
    ("RRepsSolv", "instances per family and size (solvers)"),
    ("RTimeLimit", "time limit in the caption of Table tab:solvers"),
    ("RgrbOptimal", "runs Gurobi solved to optimality"), ("RgrbMaxDev", "largest deviation ours vs Gurobi"),
    ("RgrbAtwoWins", "runs whose optimum is attained under Assumption 2"),
    ("RgrbAllSolvedRed", "Gurobi solved all runs up to n = (reduction)"),
    ("RgrbAllSolvedAug", "Gurobi solved all runs up to n = (augmentation)"),
    ("RspeedupRed", "median slow-down of Gurobi (reduction)"),
    ("RspeedupAug", "median slow-down of Gurobi (augmentation)"),
    ("RgrbMaxGap", "largest final MIP gap of Gurobi at the time limit [%]"),
    ("RReps", "instances per entry in Table tab:runtimes"),
    ("RexpMin", "smallest exponent p (without catA2)"), ("RexpMax", "largest exponent p (without catA2)"),
    ("RexpAtwoCatAtwo", "exponent p for catA2"), ("RKmax", "largest K"),
    ("RepsTimeRatio", "time ratio eps=1e-8 / eps=1e-2"),
]


def table_rows(path):
    if not os.path.exists(path):
        return ["(table not produced)"]
    rows = [l for l in open(path) if l.rstrip().endswith("\\\\") and "&" in l]
    return [r.rstrip() for r in rows if not r.lstrip().startswith(("Setting", "Algorithm", " & & &", "& &"))]


def write_values():
    banner("3/3  TABLES, FIGURE AND VALUES")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        make_tables.main()
    vals = {}
    for line in buf.getvalue().splitlines():
        m = re.match(r"\s+(R\w+)\s+(.*)$", line)
        if m:
            vals[m.group(1)] = m.group(2)
    clean = lambda v: v.replace("\\textbf{??}", "?? (not run)").replace("\\,", " ").replace("\\cdot", "x")
    out = ["VALUES FOR computational_results_plain.tex",
           "(in the order in which the bold ?? appear; '?? (not run)' = experiment missing)", ""]
    for i, (k, desc) in enumerate(FILL_ORDER, 1):
        out.append(f"{i:2d}. [{k}]  {desc}:  {clean(vals.get(k, '?? (not run)'))}")
    out += ["", "ROWS OF TABLE tab:solvers (copy the numbers into the empty cells):"]
    out += table_rows(os.path.join(HERE, "latex", "table_solvers.tex"))
    out += ["", "ROWS OF TABLE tab:runtimes:"]
    out += table_rows(os.path.join(HERE, "latex", "table_runtimes.tex"))
    out += ["", "FIGURE: upload latex/iqcp_scaling.pdf to Overleaf and replace the \\fbox{...}",
            "        by  \\includegraphics[width=\\textwidth]{iqcp_scaling.pdf}"]
    txt = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "VALUES_FOR_PAPER.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"Saved to {os.path.join(HERE, 'VALUES_FOR_PAPER.txt')} and the folder latex/")


def main():
    if not gurobi_ok():
        sys.exit("Gurobi is needed. Install gurobipy and activate the academic licence"
                 " (grbgetkey ...), then run again.")
    if not SKIP_CHECKS:
        run_checks()
    run_experiments()
    write_values()


if __name__ == "__main__":
    main()
