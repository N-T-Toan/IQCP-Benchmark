#!/usr/bin/env python3
"""Benchmark driver for the computational section of the IQCP manuscript.

Sub-commands
  selfcheck   Gurobi vs. the algorithms on small instances (run FIRST)
  scaling     running time of the algorithms on growing instances
  solvers     algorithms vs Gurobi (exact MIQCP) on growing instances
  eps         dependence of the ternary search on the accuracy epsilon
  env         record machine / software information (done automatically)
  all         env + scaling + eps + solvers with the default settings

Every run appends one JSON record per line to results/<part>.jsonl; records
already present are skipped, so an interrupted run can simply be restarted.
Then build the LaTeX material with   python make_tables.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from iqcp.instance import generate, INF
from iqcp.algorithms import alg1, alg2, alg3, alg4, EPS_DEFAULT

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALG = {"alg1": alg1, "alg2": alg2, "alg3": alg3, "alg4": alg4}

# ----------------------------------------------------------------- defaults
SCALING_SIZES = [101, 201, 401, 801, 1601, 3201, 6401]       # n = 4m+1 (caterpillars)
SCALING_A2_SIZES = [41, 81, 161, 321, 641, 1281]              # family catA2 (cubic case)
SCALING_PLAN = [  # (family, algorithms, sizes)
    ("rrt", ("alg1", "alg3"), SCALING_SIZES),
    ("cat", ("alg1", "alg3"), SCALING_SIZES),
    ("rrt-hc", ("alg2", "alg4"), SCALING_SIZES),
    ("cat-hc", ("alg2", "alg4"), SCALING_SIZES),
    ("catA2", ("alg2",), SCALING_A2_SIZES),
]
SOLVER_SIZES = [17, 33, 65, 129, 257, 513, 1025, 2049]
SOLVER_FAMILIES = ["rrt", "cat", "rrt-hc", "cat-hc"]
EPS_VALUES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def _done(path, keyfields):
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add(tuple(r.get(k) for k in keyfields))
                except json.JSONDecodeError:
                    pass
    return seen


def _append(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def _clean(v):
    return None if v == INF else v


def timed(fn, *a, min_time=0.2, **k):
    """Mean wall-clock time of fn(*a, **k); repeated until min_time elapsed."""
    reps, total, res = 0, 0.0, None
    while True:
        t0 = time.perf_counter()
        res = fn(*a, **k)
        total += time.perf_counter() - t0
        reps += 1
        if total >= min_time or reps >= 50:
            return res, total / reps


# ----------------------------------------------------------------- env
def cmd_env(args):
    info = dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), machine=platform.machine())
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if "cpu" not in info and sys.platform == "darwin":
        try:
            import subprocess
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except Exception:
            pass
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30)
    except Exception:
        try:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
        except Exception:
            pass
    try:
        import gurobipy as gp
        info["gurobi"] = ".".join(map(str, gp.gurobi.version()))
    except Exception:
        info["gurobi"] = None
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "env.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


# ----------------------------------------------------------------- scaling
def cmd_scaling(args):
    path = os.path.join(RES, "scaling.jsonl")
    done = _done(path, ("family", "n", "seed", "alg"))
    for fam, algs, sizes in SCALING_PLAN:
        if args.families and fam not in args.families:
            continue
        for n in sizes:
            if args.max_n and n > args.max_n:
                continue
            for seed in range(args.reps):
                inst = None
                for a in algs:
                    if (fam, n, seed, a) in done:
                        continue
                    inst = inst or generate(fam, n, seed)
                    res, t = timed(ALG[a], inst)
                    rec = dict(family=fam, n=n, seed=seed, alg=a, time=t,
                               value=_clean(res["value"]), tau=res.get("tau"),
                               stats={k: v for k, v in res["stats"].items() if k != "interval"})
                    _append(path, rec)
                    print(f"[scaling] {fam:7s} n={n:5d} seed={seed:2d} {a}: {t:9.4f}s  value={res['value']:.4f}")


# ----------------------------------------------------------------- solver comparison
def _alg_solve(inst, setting):
    if setting == "reduction":
        (r1, t1), (r2, t2) = timed(alg1, inst), timed(alg2, inst)
        return min(r1["value"], r2["value"]), t1 + t2, ("A1" if r1["value"] <= r2["value"] else "A2")
    (r3, t3), (r4, t4) = timed(alg3, inst), timed(alg4, inst)
    return min(r3["value"], r4["value"]), t3 + t4, ("A1" if r3["value"] <= r4["value"] else "A2")


def cmd_solvers(args):
    """Algorithms vs Gurobi (exact MIQCP)."""
    from iqcp.gurobi_model import solve_gurobi
    path = os.path.join(RES, "solvers.jsonl")
    done = _done(path, ("family", "n", "seed", "setting", "solver"))
    solvers = ["alg", "gurobi"]
    sizes = args.sizes or SOLVER_SIZES
    fams = args.families or SOLVER_FAMILIES
    tl = args.time_limit
    # once every run of a (solver, setting, family) hits the limit at some size,
    # larger sizes are skipped for it (they would only time out as well)
    given_up = set()
    for n in sizes:
        for fam in fams:
            nn = n if not fam.startswith("cat") else 4 * ((n - 1) // 4) + 1
            for setting in ("reduction", "augmentation"):
                timeouts = {s: 0 for s in solvers}
                ran = {s: 0 for s in solvers}
                for seed in range(args.reps):
                    inst = generate(fam, nn, seed)
                    for solver in solvers:
                        key = (fam, nn, seed, setting, solver)
                        if key in done or (solver, setting, fam) in given_up:
                            continue
                        rec = dict(family=fam, n=nn, seed=seed, setting=setting, solver=solver)
                        if solver == "alg":
                            val, t, win = _alg_solve(inst, setting)
                            rec.update(status="optimal" if val < INF else "infeasible", obj=_clean(val),
                                       time=t, winner=win)
                        else:
                            r = solve_gurobi(inst, setting, time_limit=tl, threads=args.threads)
                            rec.update(status=r["status"], obj=_clean(r["obj"]), bound=r["bound"],
                                       gap=_clean(r["gap"]) if r["gap"] is not None else None,
                                       time=r["time"], build_time=r.get("build_time"),
                                       nvars=r["nvars"], ncons=r["ncons"], nbin=r["nbin"], time_limit=tl)
                            ran[solver] += 1
                            timeouts[solver] += r["status"] == "time_limit"
                        _append(path, rec)
                        print(f"[solvers] {fam:7s} n={nn:5d} seed={seed} {setting:12s} {solver:6s} "
                              f"{rec['status']:10s} obj={rec['obj'] if rec['obj'] is not None else 'inf'} "
                              f"time={rec['time']:.3f}s", flush=True)
                for sv in solvers:
                    if sv != "alg" and ran[sv] and timeouts[sv] == ran[sv] and args.skip_after_timeout:
                        given_up.add((sv, setting, fam))
                        print(f"[solvers] {sv} timed out on all n={nn} {fam} {setting} runs -> larger sizes skipped")


def cmd_selfcheck(args):
    """Small trees: Gurobi (exact) must match the algorithms."""
    from iqcp.gurobi_model import solve_gurobi
    path = os.path.join(RES, "selfcheck.jsonl")
    nbad, worst = 0, 0.0
    for n in (9, 13, 17):
        for fam in SOLVER_FAMILIES:
            for seed in range(args.reps):
                inst = generate(fam, n, 1000 + seed)
                for setting in ("reduction", "augmentation"):
                    val, t, win = _alg_solve(inst, setting)
                    line = f"[selfcheck] {fam:7s} n={n:3d} seed={seed} {setting:12s} alg={val:.6f}"
                    r = solve_gurobi(inst, setting, time_limit=args.time_limit, threads=args.threads)
                    ok = True
                    if r["status"] == "optimal" and val < INF:
                        dev = (val - r["obj"]) / max(1.0, abs(r["obj"]))
                        ok = abs(dev) <= (1e-4 if setting == "reduction" else 1e-3)
                        worst = max(worst, abs(dev))
                    elif r["status"] == "optimal":
                        ok = False                    # Gurobi found a solution, the algorithms did not
                    elif r["status"] == "infeasible":
                        ok = val == INF
                    nbad += not ok
                    line += f"  gurobi={r['obj']:.6f} [{r['status']}]{'' if ok else '  <-- MISMATCH'}"
                    _append(path, dict(family=fam, n=n, seed=seed, setting=setting, solver="gurobi",
                                       alg=_clean(val), obj=_clean(r["obj"]), status=r["status"], ok=ok))
                    print(line, flush=True)
    print("\nSELF-CHECK", "PASSED" if nbad == 0 else f"FAILED ({nbad} mismatches)",
          f"- largest deviation from Gurobi {worst:.2e}")
    if nbad:
        sys.exit(1)


# ----------------------------------------------------------------- eps
def cmd_eps(args):
    path = os.path.join(RES, "eps.jsonl")
    done = _done(path, ("n", "seed", "eps"))
    n = args.n or 801
    for seed in range(args.reps):
        inst = None
        for eps in EPS_VALUES:
            if (n, seed, eps) in done:
                continue
            inst = inst or generate("star", n, seed)
            res, t = timed(alg3, inst, eps=eps)
            _append(path, dict(n=n, seed=seed, eps=eps, time=t, value=_clean(res["value"]),
                               n_F_evals=res["stats"]["n_F_evals"], n_leaves=res["stats"]["n_leaves"]))
            print(f"[eps] n={n} seed={seed} eps={eps:.0e}: {t:.4f}s evals={res['stats']['n_F_evals']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["selfcheck", "scaling", "solvers", "eps", "env", "all"])
    ap.add_argument("--reps", type=int, default=None, help="instances per (family, size)")
    ap.add_argument("--sizes", type=int, nargs="*", help="sizes for the solver comparison")
    ap.add_argument("--families", nargs="*", help="restrict to these families")
    ap.add_argument("--max-n", type=int, default=None, help="skip scaling sizes above this")
    ap.add_argument("--time-limit", type=float, default=600.0, help="Gurobi time limit per run [s]")
    ap.add_argument("--threads", type=int, default=1, help="Gurobi threads (1 = fair comparison)")
    ap.add_argument("--no-skip", dest="skip_after_timeout", action="store_false",
                    help="do not skip larger sizes after a solver timed out on all runs of a size")
    ap.add_argument("--n", type=int, default=None, help="size for the eps study")
    args = ap.parse_args()
    defaults = dict(selfcheck=3, scaling=10, solvers=5, eps=10)
    cmd_env(args)
    if args.part == "env":
        return
    parts = ["scaling", "eps", "solvers"] if args.part == "all" else [args.part]
    for p in parts:
        a = argparse.Namespace(**vars(args))
        a.reps = args.reps or defaults[p]
        if p == "selfcheck":
            cmd_selfcheck(a)
        elif p == "scaling":
            cmd_scaling(a)
        elif p == "solvers":
            cmd_solvers(a)
        elif p == "eps":
            cmd_eps(a)


if __name__ == "__main__":
    main()
