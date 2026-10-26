#!/usr/bin/env python3
"""
Examples 1-4 of the paper, solved by the Python implementation of the
algorithms and by Gurobi, side by side with the hand computation.

PyCharm: right-click run_examples.py -> Run.   No benchmark is run.
The report is printed and saved to EXAMPLES_REPORT.txt.
"""
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from iqcp.instance import paper_example, LEAF_NAMES, INF          # noqa: E402
from iqcp.algorithms import (alg1, alg2, alg3, alg4, rlt, _A2Data,  # noqa: E402
                             _aug_common, _LeafPrefix, _L_of_edges)

OUT = io.StringIO()
NBAD = 0
S30 = math.sqrt(30)


def say(s=""):
    print(s)
    OUT.write(s + "\n")


def E(v):
    return f"e{v}"


def num(x):
    if x is None:
        return "-"
    if x == INF:
        return "infeasible"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def cmp(label, code, hand, tol=1e-6):
    """Print one comparison line; code and hand are numbers or sets of numbers."""
    global NBAD
    if isinstance(hand, (set, list, tuple)):
        hs, cs = sorted(hand), sorted(code)
        ok = len(hs) == len(cs) and all(abs(a - b) <= tol for a, b in zip(hs, cs))
        cstr = "{" + ", ".join(num(x) for x in cs) + "}"
        hstr = "{" + ", ".join(num(x) for x in hs) + "}"
    elif isinstance(hand, str):
        ok = code == hand
        cstr, hstr = str(code), hand
    else:
        if hand == INF or code == INF or code is None:
            ok = code == hand
        else:
            ok = abs(code - hand) <= tol * max(1.0, abs(hand))
        cstr, hstr = num(code), num(hand)
    NBAD += not ok
    say(f"  {label:44s} code = {cstr:28s} hand = {hstr:28s} {'OK' if ok else '<-- DIFFERENT'}")


def quickest_all(inst, lmod, cmod):
    return {LEAF_NAMES[t]: inst.quickest(lmod, cmod, t) for t in inst.leaves}


# ============================================================= algorithms
def example1(inst):
    say("=" * 100)
    say("EXAMPLE 1  (reduction, Assumption 1, Algorithm 1)")
    say("=" * 100)
    cmp("Q1(l,c)", inst.Q1, 123)
    cmp("Q2(l,c)", inst.Q2, 99)
    psi = rlt(inst)
    d = psi.p[0][0]
    Bl = [q for q in psi.breakpoints() if d - 1e-9 <= q <= inst.Q1 + 1e-9]
    if not Bl or abs(Bl[-1] - inst.Q1) > 1e-9:
        Bl.append(inst.Q1)
    cmp("B^l (breakpoints of C_1^-)", Bl, {60, 86, 109, 110, 123})
    below = inst.leaves_below()
    L2 = _L_of_edges(inst, below, inst.L2)
    sig, Q1, Q2 = inst.sigma, inst.Q1, inst.Q2
    C1 = lambda tau: psi.value((tau, 0.0))[0]
    Egt, rows = [], []
    for e in range(1, inst.n):
        if inst.in_T1(e):
            continue
        cbar = inst.c[e] - inst.ym[e]
        if cbar <= 0 or L2[e] + sig / cbar <= Q2 + 1e-12:
            continue
        Egt.append(e)
        lo, hi = max(d, Q2), min(Q1, L2[e] + sig / cbar)
        cands = sorted({lo, hi} | {q for q in Bl if lo < q < hi})
        for tau in cands:
            c2 = 0.0 if tau <= Q2 + 1e-12 else inst.bm[e] * (inst.c[e] - sig / (tau - L2[e]))
            rows.append((e, tau, C1(tau), c2, C1(tau) + c2))
    cmp("E^>(T2)", [E(e) for e in Egt] and Egt, {9, 10, 12, 13})
    hand_B = {9: {99, 107}, 10: {99, 101}, 12: {99, 109, 110, 123}, 13: {99, 109, 110, 123}}
    for e in Egt:
        cmp(f"B({E(e)})", [r[1] for r in rows if r[0] == e], hand_B.get(e, set()))
    say("  F^-(tau,e) = C_1^-(tau) + capacity cost on e, at every candidate:")
    for e, tau, a, b, f in rows:
        say(f"     {E(e):4s} tau = {num(tau):10s} C_1^- = {num(a):12s} + {num(b):12s} = {num(f)}")
    r = alg1(inst)
    cmp("optimal tau", r["tau"], 110)
    cmp("optimal edge", E(r["edge"]), "e12")
    cmp("F^-* = 5430/13", r["value"], 5430 / 13)
    return r


def example2(inst):
    say("=" * 100)
    say("EXAMPLE 2  (reduction, Assumption 2, Algorithm 2)")
    say("=" * 100)
    A2 = _A2Data(inst)
    cmp("e*", E(A2.estar), "e1")
    cmp("l_0", A2.ell0, 84)
    r = alg2(inst)
    lo, hi = r["stats"]["interval"]
    cmp("lower end of the interval", lo, 99)
    cmp("upper end of the interval", hi, 114)
    Bc = [A2.ell0 + inst.sigma / A2.mt[t] for t in A2.L1 if A2.mt[t] < INF]
    cmp("B^c (inside the interval)", {b for b in Bc if lo - 1e-9 <= b <= hi + 1e-9}, {104, 114})
    say("  G^-(tau) = D_1^-(tau) + beta(e*)(c(e*) - sigma/(tau - l_0)) on B = {99, 104, 109, 114}:")
    es = A2.estar
    for tau in (99, 104, 109, 114):
        D = A2.D1(tau, +1)[0]
        G = D + inst.bm[es] * (inst.c[es] - inst.sigma / (tau - A2.ell0))
        say(f"     tau = {tau:4d}   D_1^- = {num(D):12s} G^- = {num(G)}")
    cmp("optimal tau", r["tau"], 99)
    cmp("G^-* = 1325", r["value"], 1325)
    return r


def example3(inst):
    say("=" * 100)
    say("EXAMPLE 3  (augmentation, Assumption 1, Algorithm 3)")
    say("=" * 100)
    L1, C1p, Q1cy, Qmax_v = _aug_common(inst)
    T1e = [e for e in range(1, inst.n) if inst.in_T1(e)]
    hand_L1 = {1: 93, 2: 93, 3: 90, 4: 93, 5: 51, 6: 51}
    for e in T1e:
        cmp(f"L_1({E(e)})", L1[e], hand_L1[e])
    cmp("B^c", {L1[e] + inst.sigma / inst.c[e] for e in T1e}, {57, 71, 99, 100, 113, 123})
    xplus = [inst.l[v] + inst.xp[v] for v in range(inst.n)]
    Q2x = max(inst.quickest(xplus, inst.c, t) for t in inst.L2)
    cmp("Q1(l, c + ybar+)", Q1cy, 113)
    cmp("Q2(l + xbar+, c)", Q2x, 148)
    cmp("C_1^+(118) = 6000/25 - 200", C1p(118), 6000 / 25 - 200)
    leaf = {name: t for t, name in LEAF_NAMES.items()}

    def F(name, tau):
        v = leaf[name]
        lp = _LeafPrefix(inst, v, Qmax_v)
        z = tau - inst.Q[v]
        if z > lp.M(tau) + 1e-9:
            return INF
        return C1p(tau) + lp.h(z, tau)

    cmp("F_v5(113)  (v5 infeasible)", F("v5", 113), INF)
    cmp("F_v5(123)  (v5 infeasible)", F("v5", 123), INF)
    cmp("F_v4(113)", F("v4", 113), 795)
    cmp("F_v7(113)", F("v7", 113), 853.5)
    cmp("F_v6(113)", F("v6", 113), 170)
    cmp("F_v6(115)", F("v6", 115), 6000 / 22 - 120, tol=1e-6)
    cmp("F_v6(123)", F("v6", 123), 144)
    r = alg3(inst)
    cmp("optimal leaf", LEAF_NAMES.get(r["leaf"]), "v6")
    cmp("tau+* = 93 + 5 sqrt(30)", r["tau"], 93 + 5 * S30, tol=1e-6)
    cmp("F+* = 80 sqrt(30) - 296", r["value"], 80 * S30 - 296, tol=1e-6)
    return r


def example4(inst):
    say("=" * 100)
    say("EXAMPLE 4  (augmentation, Assumption 2, Algorithm 4)")
    say("=" * 100)
    es = inst.primary
    cmp("e*", E(es), "e1")
    for name, hand in (("v4", 2), ("v5", 3), ("v6", 4), ("v7", 8)):
        t = [k for k, v in LEAF_NAMES.items() if v == name][0]
        cmp(f"c(P_v*{name})  (< c(e*) = 10)", inst.bott[t], hand)
    r = alg4(inst)
    cmp("Algorithm 4", r["value"], INF)
    return r


# ============================================================= Gurobi
def gurobi_part(inst):
    say("=" * 100)
    say("GUROBI on the exact MIQCP model of the full IQCP")
    say("=" * 100)
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        say("  gurobipy is not installed -> Gurobi part skipped")
        return
    from iqcp.gurobi_model import build_model, _EnvGp

    hand_sol = {
        "Ex1": dict(x={4: 13.0}, y={12: 22 / 13}),
        "Ex2": dict(x={1: 10.0, 3: 1.0, 4: 14.0}, y={1: 6.0}),
        "Ex3": dict(x={13: 16.0, 12: 5 * S30 - 22}, y={4: 0.4 * S30 - 2}),
    }
    cases = [
        ("Ex1", "reduction", False, 5430 / 13, "full model; optimum = Example 1 (5430/13)"),
        ("Ex2", "reduction", True, 1325.0, "Assumption 2 w.r.t. T1 forced = Example 2"),
        ("Ex3", "augmentation", False, 80 * S30 - 296, "full model; optimum = Example 3"),
        ("Ex4", "augmentation", True, INF, "Assumption 2 w.r.t. T1 forced = Example 4"),
    ]
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    for key, setting, forceA2, hand, label in cases:
        say(f"-- {key}: {setting}, {label}")
        try:
            m, V = build_model(inst, setting, _EnvGp(gp, env), GRB)
            if forceA2:
                m.addConstr(V["p"][inst.primary] == 1)
            m.Params.NonConvex = 2
            m.Params.MIPGap = 1e-9
            m.Params.FeasibilityTol = 1e-9
            m.Params.TimeLimit = 600
            m.optimize()
            if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                cmp("Gurobi optimum", INF, hand)
                m.dispose()
                continue
            if m.Status != GRB.OPTIMAL:
                cmp(f"Gurobi optimum (status {m.Status})", None, hand)
                m.dispose()
                continue
            cmp("Gurobi optimum", m.ObjVal, hand)
            red = setting == "reduction"
            x = {v: V["x"][v].X for v in range(1, inst.n)}
            ct = {v: (V["ct"][v].X if V["ct"][v] is not None else inst.c[v]) for v in range(1, inst.n)}
            y = {v: (inst.c[v] - ct[v]) if red else (ct[v] - inst.c[v]) for v in range(1, inst.n)}
            xs = {E(v): round(val, 6) for v, val in x.items() if val > 1e-6}
            ys = {E(v): round(val, 6) for v, val in y.items() if val > 1e-6}
            say(f"     Gurobi:  x = {xs}   y = {ys}")
            if key in hand_sol:
                hx = {E(v): round(val, 6) for v, val in hand_sol[key]["x"].items()}
                hy = {E(v): round(val, 6) for v, val in hand_sol[key]["y"].items()}
                say(f"     hand:    x = {hx}   y = {hy}")
                same = (set(xs) == set(hx) and set(ys) == set(hy)
                        and all(abs(xs[k] - hx[k]) <= 1e-4 for k in hx)
                        and all(abs(ys[k] - hy[k]) <= 1e-4 for k in hy))
                say("     same modification as by hand" if same else
                    "     NOTE: different modification (an alternative optimum if the cost above is OK)")
            lmod = [0.0] + [inst.l[v] - x[v] if red else inst.l[v] + x[v] for v in range(1, inst.n)]
            cmod = [0.0] + [ct[v] for v in range(1, inst.n)]
            qs = quickest_all(inst, lmod, cmod)
            say("     quickest values after the modification: " +
                ", ".join(f"{k} = {num(qs[k])}" for k in sorted(qs)))
            m.dispose()
        except Exception as ex:                           # report and go on
            global NBAD
            NBAD += 1
            say(f"  ERROR: {ex}")
    env.dispose()


def main():
    inst = paper_example()
    say("Examples 1-4 of the paper: Python code and Gurobi against the hand computation")
    say("Leaves: v1..v7 as in Figure 1; edges e1..e14; sigma = 60\n")
    example1(inst)
    example2(inst)
    example3(inst)
    example4(inst)
    gurobi_part(inst)
    say("=" * 100)
    say("ALL VALUES AGREE WITH THE HAND COMPUTATION" if NBAD == 0
        else f"{NBAD} DIFFERENCE(S) - see the lines marked <-- DIFFERENT")
    with open(os.path.join(HERE, "EXAMPLES_REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(OUT.getvalue())
    print(f"\nSaved to {os.path.join(HERE, 'EXAMPLES_REPORT.txt')}")


if __name__ == "__main__":
    main()
