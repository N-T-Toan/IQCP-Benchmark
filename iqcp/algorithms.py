"""The four algorithms of the manuscript.

  alg1  : reduction,    Assumption 1  (Algorithm alg1, uses RLT = Prop. prop:C1dp)
  alg2  : reduction,    Assumption 2  (Algorithm alg:A2, Lemmas lem:cover, lem:slope,
                                       Eisner--Severance breakpoint recovery)
  alg3  : augmentation, Assumption 1  (Algorithm alg3, greedy Lemma lem:greedy,
                                       ternary search)
  alg4  : augmentation, Assumption 2  (Algorithm alg4, Lemma lem:filter)

Every function returns a dict with at least
  value  : optimal cost (inf if no solution of this type exists)
  tau    : optimal target
  stats  : counters used in the computational section
"""
from __future__ import annotations

import math
from typing import Dict, List

from .instance import INF, Instance
from .pwl import PWL, covering_dp

EPS_DEFAULT = 1e-6


# ====================================================================== helpers
def _L_of_edges(inst: Instance, leaves_by_v, side_leaves) -> List[float]:
    """L(e) = max l(P_{v* t}) over leaves t below e (restricted to side_leaves)."""
    side = set(side_leaves)
    L = [-INF] * inst.n
    for v in range(1, inst.n):
        vals = [inst.lpath[t] for t in leaves_by_v[v] if t in side]
        if vals:
            L[v] = max(vals)
    return L


def _subtree_edges(inst: Instance, first: int) -> List[int]:
    out, stack = [], [first]
    while stack:
        v = stack.pop()
        out.append(v)
        stack.extend(inst.children[v])
    return out


# ====================================================================== C_1^- (RLT)
def rlt(inst: Instance):
    """C_1^- = psi_{v*} of Proposition prop:C1dp (shift-invariant DP), as a PWL
    in the variable s = tau (float pairs with b = 0)."""
    req = {t: (inst.Q[t], 0.0) for t in inst.L1}
    return covering_dp(inst, [inst.primary], req, inst.am, inst.xm)


# ====================================================================== Algorithm 1
def alg1(inst: Instance) -> Dict:
    psi = rlt(inst)
    Q1, Q2, sig = inst.Q1, inst.Q2, inst.sigma
    d = psi.p[0][0]                                  # Q_1(l - xbar^-, c)
    C1 = lambda tau: psi.value((tau, 0.0))[0]
    Bl = [q for q in psi.breakpoints() if d - 1e-9 <= q <= Q1 + 1e-9]
    if not Bl or abs(Bl[-1] - Q1) > 1e-9:
        Bl.append(Q1)
    below = inst.leaves_below()
    L2 = _L_of_edges(inst, below, inst.L2)
    best = (INF, None, None)
    nevals = 0
    if d <= Q2 + 1e-12:
        best = (C1(Q2), Q2, None)
        nevals += 1
    n_Egt = 0
    for e in range(1, inst.n):
        if inst.in_T1(e):
            continue
        cbar = inst.c[e] - inst.ym[e]
        if cbar <= 0:
            continue
        taubar = L2[e] + sig / cbar
        if taubar <= Q2 + 1e-12:
            continue                                   # e not in E^>(T_2)
        n_Egt += 1
        lo, hi = max(d, Q2), min(Q1, taubar)
        if lo > hi + 1e-12:
            continue                                   # empty admissible range
        cands = {lo, hi} | {q for q in Bl if lo < q < hi}
        for tau in cands:
            if tau <= Q2 + 1e-12:
                c2 = 0.0                               # no capacity reduction needed
            else:
                c2 = inst.bm[e] * (inst.c[e] - sig / (tau - L2[e]))
            val = C1(tau) + c2
            nevals += 1
            if val < best[0]:
                best = (val, tau, e)
    return dict(value=best[0], tau=best[1], edge=best[2],
                stats=dict(n_breakpoints_C1=len(Bl), n_Egt=n_Egt, n_evals=nevals))


# ====================================================================== Algorithm 2
class _A2Data:
    def __init__(self, inst: Instance):
        sig = inst.sigma
        self.inst = inst
        self.estar = inst.primary
        self.ell0 = max(inst.lpath[t] for t in inst.L2)
        self.L1 = inst.L1
        self.A, self.C, self.X, self.mt = {}, {}, {}, {}
        for t in self.L1:
            p = inst.path(t)
            rest = [inst.c[e] for e in p if e != self.estar]
            m = min(rest) if rest else INF
            self.mt[t] = m
            self.A[t] = inst.lpath[t] + (sig / m if m < INF else -INF)
            self.C[t] = max(0.0, inst.lpath[t] - self.ell0)
            self.X[t] = sum(inst.xm[e] for e in p)
        t1 = [inst.c[v] for v in range(1, inst.n) if inst.in_T1(v) and v != self.estar]
        self.c1min = min(t1) if t1 else INF
        self.nD = 0

    def D1(self, tau: float, side: int):
        """(value, one-sided derivative) of D_1^- at tau; side=+1 right, -1 left."""
        self.nD += 1
        req = {}
        for t in self.L1:
            C = (self.C[t], 0.0)
            if self.A[t] == -INF:
                req[t] = C
                continue
            Ar = (self.A[t] - tau, -1.0 if side > 0 else 1.0)
            # lexicographic max
            if C[0] > Ar[0] + 1e-9 or (abs(C[0] - Ar[0]) <= 1e-9 and C[1] >= Ar[1]):
                req[t] = C
            else:
                req[t] = Ar
        f = covering_dp(self.inst, [self.estar], req, self.inst.am, self.inst.xm)
        a, b = f.value((0.0, 0.0))
        if a == INF:
            return INF, 0.0
        return a, (b if side > 0 else -b)


def alg2(inst: Instance) -> Dict:
    sig = inst.sigma
    A2 = _A2Data(inst)
    es, ell0 = A2.estar, A2.ell0
    Q1, Q2 = inst.Q1, inst.Q2
    cbar = inst.c[es] - inst.ym[es]
    tau_c = ell0 + sig / A2.c1min if A2.c1min < INF else ell0
    fin = [A2.A[t] - A2.X[t] for t in A2.L1 if A2.A[t] > -INF]
    tau_f = max(fin) if fin else -INF
    lo = max(Q2, ell0 + sig / inst.c[es], tau_f)
    hi = min(ell0 + sig / cbar if cbar > 0 else INF, Q1, max(lo, tau_c))
    stats = dict(n_D1_evals=0, K=0, n_B0=0, interval=(lo, hi))
    if lo > hi + 1e-9 or any(A2.C[t] > A2.X[t] + 1e-9 for t in A2.L1):
        return dict(value=INF, tau=None, stats=stats)
    if A2.D1(hi, -1)[0] == INF:
        stats["n_D1_evals"] = A2.nD
        return dict(value=INF, tau=None, stats=stats)
    Bc = [ell0 + sig / A2.mt[t] for t in A2.L1 if A2.mt[t] < INF]
    B0 = sorted({lo, hi} | {b for b in Bc if lo < b < hi})
    # Eisner--Severance on every subinterval
    cut = set()
    cache_val = {}

    def val_and_slopes(t):
        if t not in cache_val:
            vr, sr = A2.D1(t, +1)
            _, sl = A2.D1(t, -1)
            cache_val[t] = (vr, sl, sr)
        return cache_val[t]

    def es_rec(a, b, depth=0):
        fa, _, sa = val_and_slopes(a)
        fb, sb, _ = val_and_slopes(b)
        if abs(sa - sb) <= 1e-9 or b - a <= 1e-9 or depth > 200:
            return
        m = (fb - fa + sa * a - sb * b) / (sa - sb)
        if not (a + 1e-12 < m < b - 1e-12):
            return
        fm, slm, srm = val_and_slopes(m)
        if abs(fm - (fa + sa * (m - a))) <= 1e-7 * max(1.0, abs(fm)):
            cut.add(m)
            return
        es_rec(a, m, depth + 1)
        es_rec(m, b, depth + 1)

    if lo < hi - 1e-12:
        for a, b in zip(B0[:-1], B0[1:]):
            es_rec(a, b)
    else:
        val_and_slopes(lo)
    B = sorted(set(B0) | cut)
    best = (INF, None)
    for tau in B:
        D = val_and_slopes(tau)[0]
        if D == INF:
            continue
        G = D + inst.bm[es] * (inst.c[es] - sig / (tau - ell0))
        if G < best[0]:
            best = (G, tau)
    stats.update(n_D1_evals=A2.nD, K=len(cut), n_B0=len(B0))
    return dict(value=best[0], tau=best[1], stats=stats)


# ====================================================================== augmentation helpers
class _SegTree:
    """Range add / range min over an array (lazy propagation)."""

    def __init__(self, arr):
        n = 1
        while n < len(arr):
            n *= 2
        self.N = n
        self.mn = [INF] * (2 * n)
        self.lz = [0.0] * (2 * n)
        for i, v in enumerate(arr):
            self.mn[n + i] = v
        for i in range(n - 1, 0, -1):
            self.mn[i] = min(self.mn[2 * i], self.mn[2 * i + 1])

    def _upd(self, node, l, r, ql, qr, val):
        if qr < l or r < ql:
            return
        if ql <= l and r <= qr:
            self.mn[node] += val
            self.lz[node] += val
            return
        m = (l + r) // 2
        self._upd(2 * node, l, m, ql, qr, val)
        self._upd(2 * node + 1, m + 1, r, ql, qr, val)
        self.mn[node] = min(self.mn[2 * node], self.mn[2 * node + 1]) + self.lz[node]

    def _qry(self, node, l, r, ql, qr):
        if qr < l or r < ql:
            return INF
        if ql <= l and r <= qr:
            return self.mn[node]
        m = (l + r) // 2
        return min(self._qry(2 * node, l, m, ql, qr),
                   self._qry(2 * node + 1, m + 1, r, ql, qr)) + self.lz[node]

    def add_suffix(self, k, val, n):
        self._upd(1, 0, self.N - 1, k, n - 1, val)

    def min_suffix(self, k, n):
        return self._qry(1, 0, self.N - 1, k, n - 1)


class _LeafPrefix:
    """Data of the prefix-constrained subproblem (eq:prefix) for one leaf."""

    def __init__(self, inst: Instance, v: int, Qmax_v):
        self.v = v
        self.f = inst.path(v)                         # f_1..f_p
        self.p = len(self.f)
        self.Qmax = [Qmax_v[e] for e in self.f]
        self.xbar = [inst.xp[e] for e in self.f]
        self.alpha = [inst.ap[e] for e in self.f]
        self.order = sorted(range(self.p), key=lambda k: self.alpha[k])
        self.S = [0.0] * (self.p + 1)                  # S[k] = sum_{j>k} xbar (1-based k)
        for k in range(self.p - 1, -1, -1):
            self.S[k] = self.S[k + 1] + self.xbar[k]
        # S[0] = total; S[k] for k=1..p means sum over j>k

    def M(self, tau: float) -> float:
        """Largest realisable total increase M_i(tau) (proof of Lemma lem:greedy)."""
        best = self.S[0]
        for k in range(self.p):                       # prefix index k+1
            best = min(best, tau - self.Qmax[k] + self.S[k + 1])
        return best

    def h(self, z: float, tau: float):
        """Greedy rule of Lemma lem:greedy; returns cost or inf."""
        if z <= 1e-12:
            return 0.0
        w = [tau - q for q in self.Qmax]
        if min(w) < -1e-9:
            return INF
        st = _SegTree(w)
        rem, cost = z, 0.0
        for k in self.order:
            if rem <= 1e-12:
                break
            amt = min(self.xbar[k], rem, st.min_suffix(k, self.p))
            if amt > 0:
                st.add_suffix(k, -amt, self.p)
                rem -= amt
                cost += self.alpha[k] * amt
        return cost if rem <= 1e-9 else INF


def _aug_common(inst: Instance):
    below = inst.leaves_below()
    L1 = _L_of_edges(inst, below, inst.L1)
    T1e = [e for e in range(1, inst.n) if inst.in_T1(e)]
    sig = inst.sigma

    def C1p(tau):
        s = 0.0
        for e in T1e:
            need = sig / (tau - L1[e]) - inst.c[e]
            if need > 0:
                s += inst.bp[e] * need
        return s

    cplus = [inst.c[v] + inst.yp[v] for v in range(inst.n)]
    Q1cy = max(inst.quickest(inst.l, cplus, t) for t in inst.L1)
    Qmax_v = [0.0] * inst.n
    for v in range(1, inst.n):
        Qmax_v[v] = max(inst.Q[t] for t in below[v])
    return L1, C1p, Q1cy, Qmax_v


def _ternary(F, a, b, eps):
    """Ternary search for a convex F on [a,b]; returns (min value, argmin, #evals)."""
    nev = 0
    best = (INF, None)

    def ev(t):
        nonlocal nev, best
        nev += 1
        v = F(t)
        if v < best[0]:
            best = (v, t)
        return v

    ev(a)
    ev(b)
    while b - a > eps:
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if ev(m1) <= ev(m2):
            b = m2
        else:
            a = m1
    ev(0.5 * (a + b))
    return best[0], best[1], nev


# ====================================================================== Algorithm 3
def alg3(inst: Instance, eps: float = EPS_DEFAULT) -> Dict:
    L1, C1p, Q1cy, Qmax_v = _aug_common(inst)
    Q1, Q2 = inst.Q1, inst.Q2
    xplus = [inst.l[v] + inst.xp[v] for v in range(inst.n)]
    Q2x = max(inst.quickest(xplus, inst.c, t) for t in inst.L2)
    lo, hi = max(Q2, Q1cy), min(Q1, Q2x)
    stats = dict(n_leaves=0, n_F_evals=0)
    best = (INF, None, None)
    if lo > hi + 1e-12:
        return dict(value=INF, tau=None, leaf=None, stats=stats)
    for v in inst.L2:
        lp = _LeafPrefix(inst, v, Qmax_v)
        Qi = inst.Q[v]
        # tau-independent part of z <= M (k >= 1) and the upper bound (k = 0)
        if any(lp.Qmax[k] - Qi - lp.S[k + 1] > 1e-9 for k in range(lp.p)):
            continue
        a, b = lo, min(hi, Qi + lp.S[0])
        if a > b + 1e-12:
            continue
        stats["n_leaves"] += 1
        F = lambda tau, lp=lp, Qi=Qi: C1p(tau) + lp.h(tau - Qi, tau)
        val, tau, nev = _ternary(F, a, b, eps)
        stats["n_F_evals"] += nev
        if val < best[0]:
            best = (val, tau, v)
    return dict(value=best[0], tau=best[1], leaf=best[2], stats=stats)


# ====================================================================== Algorithm 4
def alg4(inst: Instance, eps: float = EPS_DEFAULT) -> Dict:
    L1, C1p, Q1cy, Qmax_v = _aug_common(inst)
    sig = inst.sigma
    es = inst.primary
    L1e = max(inst.lpath[t] for t in inst.L1)
    ce = inst.c[es]
    Q1, Q2 = inst.Q1, inst.Q2
    lo, hi = max(Q2, Q1cy), Q1
    stats = dict(n_leaves=0, n_F_evals=0)
    best = (INF, None, None)
    if lo > hi + 1e-12:
        return dict(value=INF, tau=None, leaf=None, stats=stats)
    for v in inst.L2:
        cP = inst.bott[v]
        if ce > cP + 1e-12:
            continue                                  # I_j empty (Lemma lem:filter)
        tj = max(lo, L1e + sig / cP)
        if tj > hi + 1e-12:
            continue
        lp = _LeafPrefix(inst, v, Qmax_v)
        lv = inst.lpath[v]
        z = lambda tau, lv=lv: max(0.0, max(L1e, tau - sig / ce) - lv)
        g = lambda tau, lp=lp, z=z: z(tau) - lp.M(tau)       # convex
        # I_j^h = {g <= 0} inside [tj, hi]
        gmin, tmin, _ = _ternary(g, tj, hi, 1e-10)
        if gmin > 1e-9:
            continue
        a, b = tj, hi
        if g(a) > 1e-9:
            x, y = a, tmin
            for _ in range(100):
                m = 0.5 * (x + y)
                if g(m) > 1e-9:
                    x = m
                else:
                    y = m
            a = y
        if g(b) > 1e-9:
            x, y = tmin, b
            for _ in range(100):
                m = 0.5 * (x + y)
                if g(m) > 1e-9:
                    y = m
                else:
                    x = m
            b = x
        stats["n_leaves"] += 1
        F = lambda tau, lp=lp, z=z: C1p(tau) + lp.h(z(tau), tau)
        val, tau, nev = _ternary(F, a, b, eps)
        stats["n_F_evals"] += nev
        if val < best[0]:
            best = (val, tau, v)
    return dict(value=best[0], tau=best[1], leaf=best[2], stats=stats)


# ====================================================================== drivers
def solve_reduction(inst: Instance) -> Dict:
    r1, r2 = alg1(inst), alg2(inst)
    return dict(value=min(r1["value"], r2["value"]), A1=r1, A2=r2)


def solve_augmentation(inst: Instance, eps: float = EPS_DEFAULT) -> Dict:
    r3, r4 = alg3(inst, eps), alg4(inst, eps)
    return dict(value=min(r3["value"], r4["value"]), A1=r3, A2=r4)
