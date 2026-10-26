#!/usr/bin/env python3
"""
Solve ONE instance whose data you type below, with Algorithms 1-4 and Gurobi.

PyCharm: edit the DATA block, then right-click solve_my_data.py -> Run.
The vertex named "v*" is the prescribed vertex. Each edge row is
    (name, from, to, lead time l, capacity c, xbar, ybar, alpha, beta)
where "from" is the end nearer to v*. The same bounds and costs are used for
reduction and for augmentation; change single augmentation costs in ALPHA_PLUS.
The data below are those of Figure 1 / Table 2 of the paper.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================== DATA
SIGMA = 60
EDGES = [
    # name   from   to     l    c  xbar ybar alpha beta
    ("e1",  "v*",  "a",   28, 10, 23,  9,  27, 150),
    ("e2",  "a",   "b",   37,  3, 26,  1,  28, 110),
    ("e3",  "b",   "v1",  25,  6, 24,  3,  15, 230),
    ("e4",  "b",   "v2",  28,  2, 14,  1,  10, 100),
    ("e5",  "a",   "c",   15, 10, 10,  8,  21, 250),
    ("e6",  "c",   "v3",   8,  3,  5,  2,   8,  90),
    ("e7",  "v*",  "d",   18,  3, 17,  1,   9, 150),
    ("e8",  "d",   "f",   15,  5,  9,  3,  18, 300),
    ("e9",  "f",   "v4",  14,  2, 10,  1,  38, 210),
    ("e10", "f",   "v5",   8,  4,  7,  3,  16, 220),
    ("e11", "v*",  "g",   29, 15, 17, 11,  35, 180),
    ("e12", "g",   "h",   37,  4, 16,  3,  21, 170),
    ("e13", "h",   "v6",  18,  5, 16,  4,  15, 190),
    ("e14", "g",   "v7",  38,  8, 27,  6,  13,  80),
]
ALPHA_PLUS = {"e12": 8, "e13": 5}   # augmentation costs that differ (Example 3)
TIME_LIMIT = 600                    # seconds for Gurobi
# ====================================================================

from iqcp.instance import Instance, INF                     # noqa: E402
from iqcp.algorithms import alg1, alg2, alg3, alg4          # noqa: E402


def build_instance():
    names = ["v*"]
    for _, a, b, *_ in EDGES:
        for v in (a, b):
            if v not in names:
                names.append(v)
    idx = {v: i for i, v in enumerate(names)}
    n = len(names)
    if len(EDGES) != n - 1:
        sys.exit("The edges do not form a tree (need exactly n-1 edges).")
    parent = [-1] * n
    arr = {k: [0.0] * n for k in ("l", "c", "xb", "yb", "al", "be", "ap")}
    ename = [""] * n
    for name, a, b, l, c, xb, yb, al, be in EDGES:
        v = idx[b]
        if parent[v] != -1:
            sys.exit(f"Vertex {b} has two parents: write every edge as (from v* side, to far side).")
        parent[v] = idx[a]
        ename[v] = name
        for k, val in zip(("l", "c", "xb", "yb", "al", "be"), (l, c, xb, yb, al, be)):
            arr[k][v] = float(val)
        arr["ap"][v] = float(ALPHA_PLUS.get(name, al))
    inst = Instance(n=n, parent=parent, sigma=float(SIGMA), l=arr["l"], c=arr["c"],
                    xm=list(arr["xb"]), ym=list(arr["yb"]), am=list(arr["al"]), bm=list(arr["be"]),
                    xp=list(arr["xb"]), yp=list(arr["yb"]), ap=list(arr["ap"]), bp=list(arr["be"]),
                    name="my-data")              # separate copies: prepare() sets ybar+ = 0 on T2
    return inst.prepare(), names, ename


def show(x):
    return "infeasible" if x == INF else f"{x:.6f}"


def main():
    inst, names, ename = build_instance()
    print(f"n = {inst.n} vertices, Q1 = {inst.Q1:g}, Q2 = {inst.Q2:g}, e* = {ename[inst.primary]}")
    if inst.is_center_already():
        print("v* is already a quickest 1-center: nothing to do.")
        return
    print("\n-- Algorithms")
    res = {}
    for f, txt in ((alg1, "reduction,    Assumption 1"), (alg2, "reduction,    Assumption 2"),
                   (alg3, "augmentation, Assumption 1"), (alg4, "augmentation, Assumption 2")):
        r = f(inst)
        res[f.__name__] = r["value"]
        extra = ""
        if r.get("edge"):
            extra = f", edge {ename[r['edge']]}"
        if r.get("leaf"):
            extra = f", leaf {names[r['leaf']]}"
        tau = f"{r['tau']:.6f}" if r.get("tau") is not None else "-"
        print(f"  {f.__name__} ({txt}): value = {show(r['value'])}, tau = {tau}{extra}")
    red, aug = min(res["alg1"], res["alg2"]), min(res["alg3"], res["alg4"])
    print(f"  optimum reduction    = {show(red)}")
    print(f"  optimum augmentation = {show(aug)}")

    print("\n-- Gurobi (exact MIQCP)")
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("  gurobipy is not installed")
        return
    from iqcp.gurobi_model import build_model, _EnvGp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    for setting, ours in (("reduction", red), ("augmentation", aug)):
        m, V = build_model(inst, setting, _EnvGp(gp, env), GRB)
        m.Params.NonConvex = 2
        m.Params.MIPGap = 1e-9
        m.Params.TimeLimit = TIME_LIMIT
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            val = m.ObjVal
            red_ = setting == "reduction"
            xs = {ename[v]: round(V["x"][v].X, 6) for v in range(1, inst.n) if V["x"][v].X > 1e-6}
            ys = {}
            for v in range(1, inst.n):
                if V["ct"][v] is not None:
                    d = (inst.c[v] - V["ct"][v].X) if red_ else (V["ct"][v].X - inst.c[v])
                    if d > 1e-6:
                        ys[ename[v]] = round(d, 6)
            ok = ours < INF and abs(val - ours) <= 1e-5 * max(1.0, abs(ours))
            print(f"  {setting:12s}: Gurobi = {val:.6f}, algorithms = {show(ours)}  "
                  f"{'SAME' if ok else 'DIFFERENT'}")
            print(f"                x = {xs}   y = {ys}")
        elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            print(f"  {setting:12s}: Gurobi = infeasible, algorithms = {show(ours)}  "
                  f"{'SAME' if ours == INF else 'DIFFERENT'}")
        else:
            print(f"  {setting:12s}: Gurobi stopped with status {m.Status} (time limit?)")
        m.dispose()
    env.dispose()


if __name__ == "__main__":
    main()
