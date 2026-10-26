"""Tree instances of the IQCP and instance generators.

Conventions
-----------
* Vertices are 0..n-1, the prespecified vertex is v* = 0 (the root).
* Every vertex v >= 1 owns the edge (parent[v], v); all edge data are stored
  in arrays indexed by that child vertex v (index 0 is unused).
* Reduction data:    xm = x-bar^-, ym = y-bar^-, am = alpha^-, bm = beta^-
* Augmentation data: xp = x-bar^+, yp = y-bar^+, ap = alpha^+, bp = beta^+
  (yp is forced to 0 on T_2, restriction (eq:noT2cap) of the paper).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

INF = float("inf")


@dataclass
class Instance:
    n: int
    parent: List[int]
    sigma: float
    l: List[float]
    c: List[float]
    xm: List[float]
    ym: List[float]
    am: List[float]
    bm: List[float]
    xp: List[float]
    yp: List[float]
    ap: List[float]
    bp: List[float]
    name: str = ""
    # derived (filled by prepare())
    children: List[List[int]] = field(default_factory=list)
    leaves: List[int] = field(default_factory=list)
    branch: List[int] = field(default_factory=list)      # child of root on the path
    order: List[int] = field(default_factory=list)       # BFS order from the root
    lpath: List[float] = field(default_factory=list)     # l(P_{v* v})
    bott: List[float] = field(default_factory=list)      # min capacity on P_{v* v}
    Q: List[float] = field(default_factory=list)         # quickest distance from v*
    primary: int = -1                                   # child of root starting T_1

    def prepare(self) -> "Instance":
        n = self.n
        # own copies, so that setting ybar+ = 0 on T2 below never changes
        # a list that the caller also passed for another parameter
        for k in ("l", "c", "xm", "ym", "am", "bm", "xp", "yp", "ap", "bp"):
            setattr(self, k, list(getattr(self, k)))
        self.children = [[] for _ in range(n)]
        for v in range(1, n):
            self.children[self.parent[v]].append(v)
        self.order = [0]
        for v in self.order:
            self.order.extend(self.children[v])
        assert len(self.order) == n, "parent array is not a tree rooted at 0"
        self.leaves = [v for v in range(1, n) if not self.children[v]]
        self.branch = [0] * n
        self.lpath = [0.0] * n
        self.bott = [INF] * n
        for v in self.order[1:]:
            p = self.parent[v]
            self.branch[v] = v if p == 0 else self.branch[p]
            self.lpath[v] = self.lpath[p] + self.l[v]
            self.bott[v] = min(self.bott[p], self.c[v])
        self.Q = [0.0] * n
        for v in range(1, n):
            self.Q[v] = self.lpath[v] + self.sigma / self.bott[v]
        tstar = max(self.leaves, key=lambda t: self.Q[t])
        self.primary = self.branch[tstar]
        # restriction (eq:noT2cap): no capacity augmentation in T_2
        for v in range(1, n):
            if self.branch[v] != self.primary:
                self.yp[v] = 0.0
        return self

    # ------------------------------------------------------------------ helpers
    def in_T1(self, v: int) -> bool:
        return v != 0 and self.branch[v] == self.primary

    @property
    def L1(self) -> List[int]:
        return [t for t in self.leaves if self.in_T1(t)]

    @property
    def L2(self) -> List[int]:
        return [t for t in self.leaves if not self.in_T1(t)]

    def path(self, v: int) -> List[int]:
        """Edges (child vertices) of P_{v* v}, ordered from v*."""
        out = []
        while v != 0:
            out.append(v)
            v = self.parent[v]
        return out[::-1]

    def quickest(self, l, c, t: int) -> float:
        p = self.path(t)
        return sum(l[e] for e in p) + self.sigma / min(c[e] for e in p)

    def Qmax(self, l, c, leaves) -> float:
        return max(self.quickest(l, c, t) for t in leaves)

    @property
    def Q1(self) -> float:
        return max(self.Q[t] for t in self.L1)

    @property
    def Q2(self) -> float:
        return max(self.Q[t] for t in self.L2)

    def leaves_below(self) -> List[List[int]]:
        """For every vertex v, the leaves of the subtree rooted at v."""
        below: List[List[int]] = [[] for _ in range(self.n)]
        for v in reversed(self.order):
            if not self.children[v]:
                below[v] = [v]
            else:
                acc = []
                for w in self.children[v]:
                    acc.extend(below[w])
                below[v] = acc
        return below

    def is_center_already(self) -> bool:
        """Theorem 1 on the unmodified instance (A1 or A2 already holds)."""
        if abs(self.Q1 - self.Q2) <= 1e-12:
            return True
        ell0 = max(self.lpath[t] for t in self.L2)
        return self.Q1 <= ell0 + self.sigma / self.c[self.primary] + 1e-12


# ====================================================================== generators
def _random_data(inst_parent, rng: random.Random, sigma=60.0):
    n = len(inst_parent)
    U = rng.uniform
    l = [0.0] + [float(rng.randint(5, 40)) for _ in range(1, n)]
    c = [0.0] + [float(rng.randint(1, 15)) for _ in range(1, n)]
    xm = [0.0] + [l[v] * U(0.5, 1.0) for v in range(1, n)]
    ym = [0.0] + [c[v] * U(0.3, 0.9) for v in range(1, n)]
    am = [0.0] + [float(rng.randint(5, 40)) for _ in range(1, n)]
    bm = [0.0] + [float(rng.randint(50, 300)) for _ in range(1, n)]
    xp = [0.0] + [l[v] * U(0.5, 2.0) for v in range(1, n)]
    yp = [0.0] + [c[v] * U(1.0, 4.0) for v in range(1, n)]
    ap = [0.0] + [float(rng.randint(5, 40)) for _ in range(1, n)]
    bp = [0.0] + [float(rng.randint(50, 300)) for _ in range(1, n)]
    return dict(n=n, parent=list(inst_parent), sigma=sigma, l=l, c=c, xm=xm, ym=ym,
                am=am, bm=bm, xp=xp, yp=yp, ap=ap, bp=bp)


def parent_rrt(n: int, rng: random.Random) -> List[int]:
    """Random recursive tree: vertices 1,2 attached to v*, vertex v>=3 to a
    uniformly random earlier vertex."""
    par = [-1, 0, 0]
    for v in range(3, n):
        par.append(rng.randrange(0, v))
    return par[:n]


def parent_cat(n: int) -> List[int]:
    """Caterpillar: two spines of length m=(n-1)/4 leave v*, every spine vertex
    carries one pendant leaf.  Requires n = 4m+1."""
    assert (n - 1) % 4 == 0, "caterpillar sizes must satisfy n = 4m+1"
    m = (n - 1) // 4
    par = [-1]
    for _side in range(2):
        prev = 0
        for _ in range(m):
            s = len(par)
            par.append(prev)          # spine vertex
            par.append(s)             # pendant leaf
            prev = s
    return par


def parent_star_of_paths(n: int, plen: int = 4) -> List[int]:
    """ceil((n-1)/plen) disjoint paths of length plen hanging from v*."""
    par = [-1]
    while len(par) < n:
        prev = 0
        for _ in range(plen):
            if len(par) >= n:
                break
            par.append(prev)
            prev = len(par) - 1
    return par


def _make(par, rng, name, boost_T2=False):
    d = _random_data(par, rng)
    inst = Instance(name=name, **d).prepare()
    if boost_T2:
        # capacities of T_2 raised to a common value above every capacity of T_1
        top = max(inst.c[v] for v in range(1, inst.n) if inst.in_T1(v)) + 1.0
        for v in range(1, inst.n):
            if not inst.in_T1(v):
                inst.c[v] = top
                inst.ym[v] = top * rng.uniform(0.3, 0.9)
        inst.prepare()
    return inst


def generate(family: str, n: int, seed: int, max_tries: int = 1000) -> Instance:
    """Generate an instance with Q_1 > Q_2 strictly and v* not yet a center.

    family in {"rrt", "cat", "rrt-hc", "cat-hc", "catA2", "star"}; the suffix
    "-hc" raises the capacities of T_2 (configuration in which Assumption 2 is
    attainable)."""
    rng = random.Random(f"{family}-{n}-{seed}")
    for _ in range(max_tries):
        if family.startswith("rrt"):
            par = parent_rrt(n, rng)
        elif family.startswith("cat") and family != "catA2":
            par = parent_cat(n)
        elif family == "star":
            par = parent_star_of_paths(n)
        elif family == "catA2":
            return _make_catA2(n, rng)
        else:
            raise ValueError(family)
        inst = _make(par, rng, f"{family}-{n}-{seed}", boost_T2=family.endswith("-hc"))
        if inst.Q1 > inst.Q2 + 1e-9 and not inst.is_center_already():
            return inst
    raise RuntimeError(f"could not generate a valid {family} instance of size {n}")


def _make_catA2(n: int, rng: random.Random) -> Instance:
    """Worst case of Algorithm 2: T_1 is a caterpillar whose spine capacities
    decrease strictly from c(e*) down to 2 (so the path bottlenecks take
    Theta(n) distinct values), pendant capacities are large, T_2 has large
    capacities, and y-bar^-(e*) allows c(e*)-y-bar^-(e*) < c_1^min."""
    for _ in range(1000):
        par = parent_cat(n)
        d = _random_data(par, rng)
        inst = Instance(name=f"catA2-{n}", **d)
        inst.prepare()
        m = (n - 1) // 4
        # first spine = vertices 1,3,5,... (pendants 2,4,6,...) -> make it T_1
        spine1 = [1 + 2 * k for k in range(m)]
        pend1 = [2 + 2 * k for k in range(m)]
        top = float(2 * m + 2)
        for k, v in enumerate(spine1):          # strictly decreasing from top to 2
            inst.c[v] = top - k * (top - 2.0) / max(m - 1, 1)
        for v in pend1:
            inst.c[v] = 10.0 * top
        for v in range(1, n):
            if v not in spine1 and v not in pend1:
                inst.c[v] = 10.0 * top
        e_star = spine1[0]
        inst.ym[e_star] = inst.c[e_star] - 1.0          # c(e*) - ybar < c_1^min = 2
        for v in range(1, n):
            if v != e_star:
                inst.ym[v] = inst.c[v] * rng.uniform(0.3, 0.9)
        # lead times: longer in T_1 so that it is primary and A2 is violated
        for v in spine1 + pend1:
            inst.l[v] = float(rng.randint(20, 40))
            inst.xm[v] = inst.l[v] * rng.uniform(0.5, 1.0)
        for v in range(1, n):
            if v not in spine1 and v not in pend1:
                inst.l[v] = float(rng.randint(5, 25))
                inst.xm[v] = inst.l[v] * rng.uniform(0.5, 1.0)
        inst.prepare()
        if inst.primary == 1 and inst.Q1 > inst.Q2 + 1e-9 and not inst.is_center_already():
            return inst
    raise RuntimeError("catA2 generation failed")


# ====================================================================== the paper's example
def paper_example(setting: str = "reduction") -> Instance:
    """Instance of Figure 1 / Table 2 of the manuscript (sigma = 60).

    Edges e1..e14; leaves v1..v7 with
      v1: e1 e2 e3   v2: e1 e2 e4   v3: e1 e5 e6
      v4: e7 e8 e9   v5: e7 e8 e10  v6: e11 e12 e13   v7: e11 e14
    """
    l = [28, 37, 25, 28, 15, 8, 18, 15, 14, 8, 29, 37, 18, 38]
    c = [10, 3, 6, 2, 10, 3, 3, 5, 2, 4, 15, 4, 5, 8]
    xb = [23, 26, 24, 14, 10, 5, 17, 9, 10, 7, 17, 16, 16, 27]
    yb = [9, 1, 3, 1, 8, 2, 1, 3, 1, 3, 11, 3, 4, 6]
    al = [27, 28, 15, 10, 21, 8, 9, 18, 38, 16, 35, 21, 15, 13]
    be = [150, 110, 230, 100, 250, 90, 150, 300, 210, 220, 180, 170, 190, 80]
    # vertex numbering: edge e_k is owned by vertex k (1..14)
    #   e1:(0,1) e2:(1,2) e3:(2,3)=v1 e4:(2,4)=v2 e5:(1,5) e6:(5,6)=v3
    #   e7:(0,7) e8:(7,8) e9:(8,9)=v4 e10:(8,10)=v5
    #   e11:(0,11) e12:(11,12) e13:(12,13)=v6 e14:(11,14)=v7
    parent = [-1, 0, 1, 2, 2, 1, 5, 0, 7, 8, 8, 0, 11, 12, 11]
    z = lambda a: [0.0] + [float(v) for v in a]
    ap = list(al)
    ap[11], ap[12] = 8, 5            # Example 3: alpha^+(e12)=8, alpha^+(e13)=5
    inst = Instance(n=15, parent=parent, sigma=60.0, l=z(l), c=z(c), xm=z(xb), ym=z(yb),
                    am=z(al), bm=z(be), xp=z(xb), yp=z(yb), ap=z(ap), bp=z(be),
                    name=f"paper-{setting}")
    return inst.prepare()


LEAF_NAMES = {3: "v1", 4: "v2", 6: "v3", 9: "v4", 10: "v5", 13: "v6", 14: "v7"}
