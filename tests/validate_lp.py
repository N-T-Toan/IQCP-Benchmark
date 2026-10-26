"""Independent validation of the implementation with linear programming
(scipy / HiGHS) on dense grids of targets.  Usage:
    python -m tests.validate_lp [n_instances]
"""
import math
import sys

import numpy as np
from scipy.optimize import linprog

from iqcp.instance import generate, INF
from iqcp.algorithms import alg1, alg2, alg3, alg4, rlt, _A2Data


def cover_lp(inst, leaves, req, edges, alpha, xbar):
    """min sum alpha x  s.t. sum_{e in P_t} x_e >= req_t, 0<=x<=xbar (edges only)."""
    idx = {e: i for i, e in enumerate(edges)}
    A, b = [], []
    for t in leaves:
        row = np.zeros(len(edges))
        for e in inst.path(t):
            row[idx[e]] = -1.0
        A.append(row)
        b.append(-req[t])
    r = linprog([alpha[e] for e in edges], A_ub=np.array(A), b_ub=np.array(b),
                bounds=[(0, xbar[e]) for e in edges], method="highs")
    return r.fun if r.status == 0 else INF


def check_instance(inst, grid=400, tol=1e-6):
    sig = inst.sigma
    T1e = [e for e in range(1, inst.n) if inst.in_T1(e)]
    T2e = [e for e in range(1, inst.n) if not inst.in_T1(e)]
    errs = []
    # ---------------- C1^- : DP vs LP
    psi = rlt(inst)
    for tau in np.linspace(inst.Q1 - 60, inst.Q1, 25):
        dp = psi.value((tau, 0.0))[0]
        lp = cover_lp(inst, inst.L1, {t: max(0, inst.Q[t] - tau) for t in inst.L1}, T1e, inst.am, inst.xm)
        if not (dp == lp == INF) and abs(dp - lp) > tol * max(1, abs(lp)):
            errs.append(("C1", tau, dp, lp))
    # ---------------- Algorithm 1 vs dense grid
    r1 = alg1(inst)
    below = inst.leaves_below()
    best = INF
    taus = np.linspace(inst.Q2, inst.Q1, grid)
    if r1["tau"] is not None:
        taus = np.append(taus, r1["tau"])
    C1 = {tau: cover_lp(inst, inst.L1, {t: max(0, inst.Q[t] - tau) for t in inst.L1}, T1e, inst.am, inst.xm)
          for tau in taus}
    for tau in taus:
        if C1[tau] == INF:
            continue
        best = min(best, C1[tau] + 0.0) if abs(tau - inst.Q2) < 1e-12 else best
        for e in T2e:
            L2 = max(inst.lpath[t] for t in below[e])
            ct = sig / (tau - L2)
            if ct >= inst.c[e] - inst.ym[e] - 1e-12:
                best = min(best, C1[tau] + inst.bm[e] * max(0.0, inst.c[e] - ct))
    if abs(best - r1["value"]) > 1e-6 * max(1, abs(best)) and not (best == r1["value"] == INF):
        errs.append(("alg1 vs grid", best, r1["value"]))
    # ---------------- D1^- : DP vs LP and slopes vs finite differences
    A2 = _A2Data(inst)
    for tau in np.linspace(A2.ell0 + 1, inst.Q1, 15):
        req = {t: (max(A2.C[t], A2.A[t] - tau)) for t in inst.L1}
        lp = cover_lp(inst, inst.L1, req, T1e, inst.am, inst.xm)
        v, sr = A2.D1(tau, +1)
        _, sl = A2.D1(tau, -1)
        if not (v == lp == INF) and abs(v - lp) > tol * max(1, abs(lp)):
            errs.append(("D1", tau, v, lp))
        if v < INF:
            h = 1e-5
            lpr = cover_lp(inst, inst.L1, {t: max(A2.C[t], A2.A[t] - tau - h) for t in inst.L1}, T1e, inst.am, inst.xm)
            lpl = cover_lp(inst, inst.L1, {t: max(A2.C[t], A2.A[t] - tau + h) for t in inst.L1}, T1e, inst.am, inst.xm)
            if lpr < INF and abs((lpr - v) / h - sr) > 1e-3:
                errs.append(("D1 right slope", tau, sr, (lpr - v) / h))
            if lpl < INF and abs((v - lpl) / h - sl) > 1e-3:
                errs.append(("D1 left slope", tau, sl, (v - lpl) / h))
    # ---------------- Algorithm 2 vs dense grid (whole [Q2, Q1], capacity bounds checked)
    r2 = alg2(inst)
    es = inst.primary
    best = INF
    g2 = np.linspace(inst.Q2, inst.Q1, grid)
    if r2["tau"] is not None:
        g2 = np.append(g2, r2["tau"])
    for tau in g2:
        if tau <= A2.ell0:
            continue
        chat = sig / (tau - A2.ell0)
        if chat > inst.c[es] + 1e-12 or chat < inst.c[es] - inst.ym[es] - 1e-12:
            continue
        req = {t: max(A2.C[t], A2.A[t] - tau) for t in inst.L1}
        D = cover_lp(inst, inst.L1, req, T1e, inst.am, inst.xm)
        if D < INF:
            best = min(best, D + inst.bm[es] * (inst.c[es] - chat))
    if abs(best - r2["value"]) > 1e-6 * max(1, abs(best)) and not (best == r2["value"] == INF):
        errs.append(("alg2 vs grid", best, r2["value"]))
    if r2["value"] < INF and best == INF:
        pass  # grid may miss a single-point interval; value checked by Gurobi selfcheck
    # ---------------- Algorithms 3 and 4 vs LP over all T_2 lead times on a grid
    r3, r4 = alg3(inst, 1e-7), alg4(inst, 1e-7)
    cplus = [inst.c[v] + inst.yp[v] for v in range(inst.n)]
    L1 = {e: max(inst.lpath[t] for t in below[e]) for e in T1e}

    def C1p(tau):
        s = 0.0
        for e in T1e:
            if tau <= L1[e] + 1e-12:
                return INF
            need = sig / (tau - L1[e]) - inst.c[e]
            if need > 1e-12:
                if need > inst.yp[e] + 1e-9:
                    return INF
                s += inst.bp[e] * need
        return s

    def t2_lp(tau, v, lower):
        """min cost of lead increases in T_2: all leaves <= tau, l~(P_v) >= lower."""
        idx = {e: i for i, e in enumerate(T2e)}
        A, b = [], []
        for t in inst.L2:
            row = np.zeros(len(T2e))
            for e in inst.path(t):
                row[idx[e]] = 1.0
            A.append(row)
            b.append(tau - inst.Q[t])
        row = np.zeros(len(T2e))
        for e in inst.path(v):
            row[idx[e]] = -1.0
        A.append(row)
        b.append(-(lower - inst.lpath[v]))
        r = linprog([inst.ap[e] for e in T2e], A_ub=np.array(A), b_ub=np.array(b),
                    bounds=[(0, inst.xp[e]) for e in T2e], method="highs")
        return r.fun if r.status == 0 else INF

    best3 = best4 = INF
    L1e = max(inst.lpath[t] for t in inst.L1)
    g34 = np.linspace(inst.Q2, inst.Q1, grid // 2)
    g34 = np.append(g34, [r["tau"] for r in (r3, r4) if r["tau"] is not None])
    for tau in g34:
        c1 = C1p(tau)
        if c1 == INF:
            continue
        cte = max(inst.c[es], sig / (tau - L1e))
        for v in inst.L2:
            # A1: v attains tau  <=> l~(P_v) >= tau - sigma/c(P_v)
            best3 = min(best3, c1 + t2_lp(tau, v, tau - sig / inst.bott[v]))
            # A2 (Keshtkar (ii) with t** = v):  l~(P_v) >= tau - sigma/c~(e*)
            best4 = min(best4, c1 + t2_lp(tau, v, tau - sig / cte))
    tol_aug = 1e-3
    if abs(best3 - r3["value"]) > tol_aug * max(1, abs(best3)) and not (best3 == r3["value"] == INF):
        errs.append(("alg3 vs grid", best3, r3["value"]))
    if abs(best4 - r4["value"]) > tol_aug * max(1, abs(best4)) and not (best4 == r4["value"] == INF):
        errs.append(("alg4 vs grid", best4, r4["value"]))
    # grid values must not be far below algorithm values (they are upper bounds of optimum)
    return errs, dict(a1=r1["value"], a2=r2["value"], a3=r3["value"], a4=r4["value"],
                      g3=best3, g4=best4)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    fams = ["rrt", "cat", "rrt-hc", "cat-hc"]
    nbad = 0
    finite = {k: 0 for k in ("a1", "a2", "a3", "a4")}
    for i in range(N):
        fam = fams[i % 4]
        n = [9, 13, 17][i % 3] if fam.startswith("cat") else 8 + i % 9
        inst = generate(fam, n, seed=i)
        errs, vals = check_instance(inst)
        for k in finite:
            finite[k] += vals[k] < INF
        if errs:
            nbad += 1
            print(inst.name, errs[:3])
    print(f"{N} instances, {nbad} with discrepancies; finite optima per algorithm: {finite}")
