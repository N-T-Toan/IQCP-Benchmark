"""Convex, non-increasing, piecewise-linear functions used by the tree
covering dynamic programme (Lemma lem:treecover of the manuscript).

A function f is stored by its breakpoints p[0] < ... < p[k] and the slopes
s[0] <= ... <= s[k-1] <= 0 of the segments between them; f = +inf left of
p[0], f = 0 right of p[k] (slope 0).  Positions are *lexicographic pairs*
(a, b) meaning a + b*eta for an infinitesimal eta > 0; this is exactly the
symbolic perturbation of Lemma lem:slope and gives one-sided derivatives.
With b = 0 everywhere the class is an ordinary float implementation.
"""
from __future__ import annotations

from functools import cmp_to_key
from typing import List, Tuple

Pair = Tuple[float, float]
INF = float("inf")
TOL = 1e-9


def lt(p: Pair, q: Pair) -> bool:
    if p[0] < q[0] - TOL:
        return True
    if p[0] > q[0] + TOL:
        return False
    return p[1] < q[1] - TOL


def _cmp(p: Pair, q: Pair) -> int:
    return -1 if lt(p, q) else (1 if lt(q, p) else 0)


def eq(p: Pair, q: Pair) -> bool:
    return abs(p[0] - q[0]) <= TOL and abs(p[1] - q[1]) <= TOL


class PWL:
    __slots__ = ("p", "s")

    def __init__(self, p: List[Pair], s: List[float]):
        self.p = p            # breakpoints
        self.s = s            # len(s) == len(p) - 1

    # ------------------------------------------------------------ constructors
    @staticmethod
    def indicator(r: Pair) -> "PWL":
        """0 on [r, inf), +inf elsewhere."""
        return PWL([r], [])

    # ------------------------------------------------------------ queries
    def slope_right_of(self, i: int) -> float:
        return self.s[i] if i < len(self.s) else 0.0

    def value(self, h: Pair) -> Pair:
        """f(h) as a pair (value, d value / d eta); (inf, 0) if infeasible."""
        p, s = self.p, self.s
        if lt(h, p[0]):
            return (INF, 0.0)
        va, vb = 0.0, 0.0
        for j in range(len(s) - 1, -1, -1):
            if not lt(h, p[j + 1]):          # h >= p[j+1]: segment entirely left of h
                break
            lo = h if lt(p[j], h) else p[j]
            # length of [max(p_j,h), p_{j+1}]
            a = p[j + 1][0] - lo[0]
            b = p[j + 1][1] - lo[1]
            va -= s[j] * a
            vb -= s[j] * b
        return (va, vb)

    def breakpoints(self) -> List[float]:
        return [q[0] for q in self.p]

    # ------------------------------------------------------------ operations
    def edge(self, alpha: float, xbar: float) -> "PWL":
        """g(h) = min_{0<=x<=xbar} alpha*x + f(h+x)  (three-piece formula)."""
        if xbar <= 0.0:
            return PWL(list(self.p), list(self.s))
        p, s = self.p, self.s
        k = len(p) - 1
        i = 0
        while i < k and s[i] < -alpha - 1e-12:
            i += 1
        # u* = p[i]
        newp = [(q[0] - xbar, q[1]) for q in p[: i + 1]]
        news = list(s[:i])
        newp.append(p[i])
        news.append(-alpha)
        newp.extend(p[i + 1:])
        news.extend(s[i:])
        return PWL(newp, news)

    @staticmethod
    def add(fs: List["PWL"]) -> "PWL":
        if len(fs) == 1:
            return fs[0]
        # domain start
        start = fs[0].p[0]
        for f in fs[1:]:
            if lt(start, f.p[0]):
                start = f.p[0]
        # merge breakpoints >= start
        pts: List[Pair] = [start]
        allp = sorted({q for f in fs for q in f.p if lt(start, q)}, key=cmp_to_key(_cmp))
        for q in allp:
            if not eq(q, pts[-1]):
                pts.append(q)
        # slope on [pts[j], pts[j+1]) = sum of slopes of the summands there
        slopes = []
        idx = [0] * len(fs)
        for j in range(len(pts) - 1):
            x = pts[j]
            tot = 0.0
            for m, f in enumerate(fs):
                fp = f.p
                t = idx[m]
                while t + 1 < len(fp) and not lt(x, fp[t + 1]):
                    t += 1
                idx[m] = t
                tot += f.slope_right_of(t) if not lt(x, fp[t]) else 0.0
            slopes.append(tot)
        # drop breakpoints between equal slopes and a trailing zero-slope part
        P, S = [pts[0]], []
        for j, sl in enumerate(slopes):
            if S and abs(S[-1] - sl) <= 1e-12:
                P[-1] = pts[j + 1]
            else:
                S.append(sl)
                P.append(pts[j + 1])
        while S and abs(S[-1]) <= 1e-12:
            S.pop()
            P.pop()
        return PWL(P, S)


def covering_dp(inst, root_edges, leaf_req, alpha, xbar):
    """Solve COV(r) on the subtree spanned by `root_edges` (edges leaving v*)
    by the dynamic programme of Lemma lem:treecover.

    leaf_req[t] : requirement r_t as a pair; returns the PWL phi_{v*}."""
    ch = inst.children
    funcs = {}

    def build(v):
        # iterative post-order to avoid recursion limits
        stack = [(v, False)]
        while stack:
            u, done = stack.pop()
            if not ch[u]:
                funcs[u] = PWL.indicator(leaf_req[u])
                continue
            if not done:
                stack.append((u, True))
                for w in ch[u]:
                    stack.append((w, False))
            else:
                funcs[u] = PWL.add([funcs.pop(w).edge(alpha[w], xbar[w]) for w in ch[u]])
        return funcs.pop(v)

    parts = [build(e).edge(alpha[e], xbar[e]) for e in root_edges]
    return PWL.add(parts)
