"""Exact mathematical programming model of the full IQCP, solved by Gurobi.

The relation w = sigma / c~ is imposed exactly by the bilinear equality
c~ * w = sigma, so the model is a nonconvex MIQCP (Gurobi, NonConvex=2) and
Gurobi's optimum is the exact IQCP optimum.

The model does NOT use any structural result of the manuscript: lead times may
be modified on every edge, capacities on every edge allowed by the setting
(reduction: all edges; augmentation: edges of T_1 only, restriction
(eq:noT2cap)), and any branch of v* may become the primary subtree.  The
optimality of v* is imposed through Theorem 1 (Assumption 1 OR Assumption 2).

Variables (per edge e = (parent(v), v), indexed by v):
  x_v     lead-time modification,   l~_v = l_v -/+ x_v
  ct_v    modified capacity,        bounds [c - ybar^-, c] or [c, c + ybar^+]
  w_v     = sigma / ct_v            via the bilinear equality ct_v * w_v = sigma
  D_v     = l~(P_{v* v})            (linear recursion)
  W_v     = max_{e in P_{v* v}} w_e (exact, one binary s_v per vertex)
  Q_t     = D_t + W_t for leaves t  (quickest distance after modification)
  Z       = max_t Q_t               (exact, binaries a_t)
Theorem 1:
  A1 (binary m1): at least two branches contain a leaf attaining Z;
  A2 (binary m2): some branch b attains Z (binary p_b) and some leaf t outside
                  b satisfies Z <= D_t + w_{e*_b}  (binary g_{b,t}),
                  where e*_b is the edge from v* into branch b.
"""
from __future__ import annotations

import time
from typing import Dict


def build_model(inst, setting: str, gp, GRB, name: str = "iqcp"):
    red = setting == "reduction"
    n, sig = inst.n, inst.sigma
    par, ch = inst.parent, inst.children
    m = gp.Model(name)
    E = list(range(1, n))
    x, ct, w = {}, {}, {}
    caplo, caphi = {}, {}
    for v in E:
        xb = inst.xm[v] if red else inst.xp[v]
        x[v] = m.addVar(lb=0.0, ub=xb, name=f"x[{v}]")
        if red:
            lo, hi = inst.c[v] - inst.ym[v], inst.c[v]
        else:
            lo, hi = inst.c[v], inst.c[v] + inst.yp[v]
        caplo[v], caphi[v] = lo, hi
        if hi > lo + 1e-12:                       # modifiable capacity
            ct[v] = m.addVar(lb=lo, ub=hi, name=f"ct[{v}]")
            w[v] = m.addVar(lb=sig / hi, ub=sig / lo, name=f"w[{v}]")
            m.addQConstr(ct[v] * w[v] == sig, name=f"bil[{v}]")
        else:                                       # fixed capacity
            ct[v] = None
            w[v] = sig / inst.c[v]
    wmin = min(sig / caphi[v] for v in E)
    wmax = max(sig / caplo[v] for v in E)
    Mw = wmax - wmin
    # path lengths and bottleneck terms
    D, W, s = {0: 0.0}, {}, {}
    sign = -1.0 if red else 1.0
    Dub = {0: 0.0}
    for v in inst.order[1:]:
        p = par[v]
        Dub[v] = Dub[p] + inst.l[v] + (0.0 if red else inst.xp[v])
        D[v] = m.addVar(lb=0.0, ub=Dub[v], name=f"D[{v}]")
        m.addConstr(D[v] == D[p] + inst.l[v] + sign * x[v], name=f"Ddef[{v}]")
        if p == 0:
            W[v] = w[v]
        else:
            W[v] = m.addVar(lb=wmin, ub=wmax, name=f"W[{v}]")
            s[v] = m.addVar(vtype=GRB.BINARY, name=f"s[{v}]")
            m.addConstr(W[v] >= W[p], name=f"Wge1[{v}]")
            m.addConstr(W[v] >= w[v], name=f"Wge2[{v}]")
            m.addConstr(W[v] <= W[p] + Mw * (1 - s[v]), name=f"Wle1[{v}]")
            m.addConstr(W[v] <= w[v] + Mw * s[v], name=f"Wle2[{v}]")
    L = inst.leaves
    Zub = max(Dub[t] for t in L) + wmax
    MZ = Zub + wmax
    Z = m.addVar(lb=0.0, ub=Zub, name="Z")
    a = {}
    for t in L:
        m.addConstr(Z >= D[t] + W[t], name=f"Zge[{t}]")
        a[t] = m.addVar(vtype=GRB.BINARY, name=f"a[{t}]")
        m.addConstr(D[t] + W[t] >= Z - MZ * (1 - a[t]), name=f"att[{t}]")
    B = ch[0]
    leaves_of = {b: [t for t in L if inst.branch[t] == b] for b in B}
    u = {b: m.addVar(vtype=GRB.BINARY, name=f"u[{b}]") for b in B}
    for b in B:
        m.addConstr(u[b] <= gp.quicksum(a[t] for t in leaves_of[b]), name=f"ub[{b}]")
    m1 = m.addVar(vtype=GRB.BINARY, name="mode_A1")
    m2 = m.addVar(vtype=GRB.BINARY, name="mode_A2")
    m.addConstr(m1 + m2 >= 1, name="mode")
    m.addConstr(gp.quicksum(u[b] for b in B) >= 2 * m1, name="A1")
    pb, g = {}, {}
    for b in B:
        pb[b] = m.addVar(vtype=GRB.BINARY, name=f"p[{b}]")
        m.addConstr(pb[b] <= u[b], name=f"pb[{b}]")
        others = [t for t in L if inst.branch[t] != b]
        for t in others:
            g[b, t] = m.addVar(vtype=GRB.BINARY, name=f"g[{b},{t}]")
            m.addConstr(Z <= D[t] + w[b] + MZ * (1 - g[b, t]), name=f"A2ii[{b},{t}]")
        m.addConstr(gp.quicksum(g[b, t] for t in others) >= pb[b], name=f"A2t[{b}]")
    m.addConstr(gp.quicksum(pb[b] for b in B) >= m2, name="A2")
    # objective
    obj = gp.quicksum((inst.am[v] if red else inst.ap[v]) * x[v] for v in E)
    for v in E:
        if ct[v] is not None:
            if red:
                obj += inst.bm[v] * (inst.c[v] - ct[v])
            else:
                obj += inst.bp[v] * (ct[v] - inst.c[v])
    m.setObjective(obj, GRB.MINIMIZE)
    vars_ = dict(x=x, ct=ct, w=w, D=D, W=W, s=s, Z=Z, a=a, u=u, m1=m1, m2=m2, p=pb, g=g)
    return m, vars_


def solve_gurobi(inst, setting: str, time_limit: float = 3600.0, threads: int = 1,
                 mip_gap: float = 1e-6, verbose: bool = False) -> Dict:
    import gurobipy as gp
    from gurobipy import GRB

    t0 = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    with env:
        m, _ = build_model(inst, setting, _EnvGp(gp, env), GRB)
        m.Params.TimeLimit = time_limit
        m.Params.Threads = threads
        m.Params.MIPGap = mip_gap
        m.Params.NonConvex = 2                 # needed for c~ * w = sigma
        m.Params.FeasibilityTol = 1e-8
        m.Params.IntFeasTol = 1e-8
        m.Params.NumericFocus = 1
        build = time.perf_counter() - t0
        m.optimize()
        st = m.Status
        res = dict(
            status={GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
                    GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "infeasible"}.get(st, str(st)),
            obj=m.ObjVal if m.SolCount > 0 else float("inf"),
            bound=m.ObjBound if st in (GRB.OPTIMAL, GRB.TIME_LIMIT) else float("nan"),
            gap=m.MIPGap if m.SolCount > 0 else float("inf"),
            time=m.Runtime, build_time=build, nodes=m.NodeCount,
            nvars=m.NumVars, ncons=m.NumConstrs + m.NumQConstrs, nbin=m.NumBinVars)
        m.dispose()
    return res


class _EnvGp:
    """Thin wrapper so that build_model creates the model in a given Env."""

    def __init__(self, gp, env):
        self._gp, self._env = gp, env

    def Model(self, name):
        return self._gp.Model(name, env=self._env)

    def quicksum(self, it):
        return self._gp.quicksum(it)
