#!/usr/bin/env python3
"""Turn results/*.jsonl into the LaTeX material of the computational section:

  latex/results_macros.tex   \\newcommand's with every number quoted in the text
  latex/table_runtimes.tex   Table tab:runtimes
  latex/table_solvers.tex    Table tab:solvers (ours vs Gurobi)
  latex/table_eps.tex        Table tab:eps
  latex/iqcp_scaling.pdf     Figure fig:scaling (three panels)

Missing data are rendered as a bold "??" so that nothing is silently invented.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from statistics import mean, median, pstdev

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "latex")
MISSING = r"\textbf{??}"

ALGNAME = {"alg1": "Algorithm~\\ref{alg1}", "alg2": "Algorithm~\\ref{alg:A2}",
           "alg3": "Algorithm~\\ref{alg3}", "alg4": "Algorithm~\\ref{alg4}"}
ALGSHORT = {"alg1": "Alg.~\\ref{alg1}", "alg2": "Alg.~\\ref{alg:A2}",
            "alg3": "Alg.~\\ref{alg3}", "alg4": "Alg.~\\ref{alg4}"}
# Numbers of the algorithms in the compiled manuscript (Algorithm 1 is RLT), used in
# the figure legend, where \ref cannot be used.  Adjust if the numbering changes.
ALGNUM = {"alg1": 2, "alg2": 3, "alg3": 4, "alg4": 5}
FAMTEX = {"rrt": "\\emph{rrt}", "cat": "\\emph{cat}", "rrt-hc": "\\emph{rrt-hc}",
          "cat-hc": "\\emph{cat-hc}", "catA2": "\\emph{catA2}"}
SCALING_ROWS = [("alg1", "rrt"), ("alg1", "cat"), ("alg2", "rrt-hc"), ("alg2", "cat-hc"),
                ("alg2", "catA2"), ("alg3", "rrt"), ("alg3", "cat"), ("alg4", "rrt-hc"), ("alg4", "cat-hc")]
FIT_MIN_N = {"catA2": 161}
FIT_MIN_N_DEFAULT = 401


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return MISSING
    return f"{x:.{nd}f}"


def fmt_time(t):
    if t is None:
        return "--"
    if t < 0.01:
        return f"{t * 1000:.2f}\\,ms"
    if t < 100:
        return f"{t:.2f}"
    return f"{t:.0f}"


def fmt_ms(t):
    if t is None:
        return "--"
    v = 1000.0 * t
    if v >= 100:
        return f"{v:,.0f}".replace(",", "\\,")
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def fmt_sci(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return MISSING
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.1f}\\cdot 10^{{{e}}}"


def fit_exponent(pairs):
    pairs = [(n, t) for n, t in pairs if t > 0]
    if len(pairs) < 2:
        return None
    x = np.log([p[0] for p in pairs])
    y = np.log([p[1] for p in pairs])
    return float(np.polyfit(x, y, 1)[0])


# ======================================================================= scaling
def scaling_tables(macros):
    recs = load("scaling.jsonl")
    by = defaultdict(list)
    for r in recs:
        by[(r["alg"], r["family"], r["n"])].append(r)
    reps = {r["seed"] for r in recs}
    macros["RReps"] = str(len(reps)) if reps else MISSING
    sizes = sorted({r["n"] for r in recs if r["family"] != "catA2"})
    sizesA2 = sorted({r["n"] for r in recs if r["family"] == "catA2"})
    lines = []
    lines.append("\\begin{table}[htbp]\n\\centering\n\\small")
    lines.append("\\caption{Mean running time in milliseconds (wall clock, mean over "
                 f"{macros.get('RReps', MISSING)} instances per entry) and exponent $p$ of a "
                 "least-squares fit $t\\approx C n^{p}$. Columns give the number of vertices $n$; for "
                 "the family \\emph{catA2} they are " + ", ".join(map(str, sizesA2)) + ".}")
    lines.append("\\label{tab:runtimes}")
    ncol = max(len(sizes), len(sizesA2))
    lines.append("\\begin{tabular}{ll" + "r" * ncol + "r}\n\\hline")
    lines.append("Algorithm & Family & " + " & ".join(f"$n_{{{i + 1}}}$" for i in range(ncol)) + " & $p$ \\\\")
    lines.append("& & " + " & ".join(map(str, sizes)) + " & \\\\\n\\hline")
    curves = {}
    exps = []
    for alg, fam in SCALING_ROWS:
        ns = sizesA2 if fam == "catA2" else sizes
        row, pairs = [], []
        for n in ns:
            rs = by.get((alg, fam, n), [])
            if rs:
                t = mean(r["time"] for r in rs)
                row.append(fmt_ms(t))
                pairs.append((n, t))
            else:
                row.append("--")
        row += [""] * (ncol - len(ns))
        minn = FIT_MIN_N.get(fam, FIT_MIN_N_DEFAULT)
        p = fit_exponent([(n, t) for n, t in pairs if n >= minn])
        curves[(alg, fam)] = pairs
        key = "Rexp" + {"alg1": "Aone", "alg2": "Atwo", "alg3": "Athree", "alg4": "Afour"}[alg] + \
              {"rrt": "Rrt", "cat": "Cat", "rrt-hc": "RrtHc", "cat-hc": "CatHc", "catA2": "CatAtwo"}[fam]
        macros[key] = fmt(p)
        lines.append(f"{ALGSHORT[alg]} & {FAMTEX[fam]} & " + " & ".join(row) + f" & {fmt(p)} \\\\")
        if p is not None and fam != "catA2":
            exps.append(p)
    macros["RexpMin"] = fmt(min(exps)) if exps else MISSING
    macros["RexpMax"] = fmt(max(exps)) if exps else MISSING
    lines.append("\\hline\n\\end{tabular}\n\\end{table}")
    write("table_runtimes.tex", "\n".join(lines) + "\n")

    # quantities quoted in the text
    if sizes:
        nmax = sizes[-1]
        macros["RmaxN"] = f"{nmax:,}".replace(",", "\\,")
        for alg, key in (("alg1", "RtimeAoneMaxN"), ("alg3", "RtimeAthreeMaxN"), ("alg4", "RtimeAfourMaxN")):
            ts = [mean(r["time"] for r in by[(alg, f, nmax)]) for f in ("rrt", "cat", "rrt-hc", "cat-hc")
                  if by.get((alg, f, nmax))]
            macros[key] = fmt(max(ts)) if ts else MISSING
    if sizesA2:
        nA = sizesA2[-1]
        macros["RmaxNAtwo"] = f"{nA:,}".replace(",", "\\,")
        rs = by.get(("alg2", "catA2", nA), [])
        macros["RtimeAtwoMaxN"] = fmt(mean(r["time"] for r in rs)) if rs else MISSING
        allA2 = [r for r in recs if r["family"] == "catA2"]
        macros["RKmax"] = str(max(r["stats"]["K"] for r in allA2)) if allA2 else MISSING
        macros["RKmean"] = fmt(mean(r["stats"]["K"] for r in allA2)) if allA2 else MISSING
        ratios = [r["stats"]["n_D1_evals"] / r["n"] for r in allA2 if r["n"] == nA]
        macros["RDevalsPerN"] = fmt(mean(ratios)) if ratios else MISSING
        ratiosB = [r["stats"]["n_B0"] / r["n"] for r in allA2 if r["n"] == nA]
        macros["RBzeroPerN"] = fmt(mean(ratiosB)) if ratiosB else MISSING
    hc = [r for r in recs if r["alg"] == "alg2" and r["family"] in ("rrt-hc", "cat-hc")]
    macros["RDevalsHcMax"] = str(max(r["stats"]["n_D1_evals"] for r in hc)) if hc else MISSING
    bp = [r["stats"]["n_breakpoints_C1"] for r in recs if r["alg"] == "alg1"]
    macros["RbpCOneMax"] = str(max(bp)) if bp else MISSING
    return curves


# ======================================================================= solver comparison
SOLVED = ("optimal", "infeasible")


def solvers_table(macros):
    recs = load("solvers.jsonl")
    keys = ("RTimeLimit", "RRepsSolv", "RsolLargestN", "RgrbAtwoWins",
            "RgrbRuns", "RgrbOptimal", "RgrbMaxDev", "RgrbInconsistent", "RgrbAllSolvedRed",
            "RgrbAllSolvedAug", "RspeedupRed", "RspeedupAug", "RgrbSolvedLargest", "RgrbRunsLargest",
            "RgrbMaxGap")
    for k in keys:
        macros[k] = MISSING
    if not recs:
        write("table_solvers.tex", "% no results yet: run  python run_bench.py solvers\n")
        return None
    opt = {(r["family"], r["n"], r["seed"], r["setting"]): r for r in recs if r["solver"] == "alg"}
    A_all = [r for r in recs if r["solver"] == "alg"]
    Gr = [r for r in recs if r["solver"] == "gurobi"]
    tl_recs = [r for r in recs if r.get("time_limit")]
    if tl_recs:
        macros["RTimeLimit"] = f"{tl_recs[0]['time_limit']:.0f}"
    macros["RRepsSolv"] = str(len({r["seed"] for r in recs}))
    sizes = sorted({r["n"] for r in recs})
    macros["RsolLargestN"] = f"{sizes[-1]:,}".replace(",", "\\,")
    macros["RgrbAtwoWins"] = str(sum(1 for r in A_all if r.get("winner") == "A2" and r["obj"] is not None))

    def opt_of(r):
        a = opt.get((r["family"], r["n"], r["seed"], r["setting"]))
        return None if a is None else a["obj"]

    def dev(r):                           # relative deviation ours vs Gurobi (solved runs)
        o = opt_of(r)
        if r["status"] != "optimal" or o is None or r["obj"] is None:
            return None
        return abs((o - r["obj"]) / max(1.0, abs(r["obj"])))

    if Gr:
        gs = [r for r in Gr if r["status"] in SOLVED]
        bad = sum(1 for r in gs if (r["status"] == "infeasible") != (opt_of(r) is None))
        dv = [dev(r) for r in gs if dev(r) is not None]
        gaps = [r["gap"] for r in Gr if r["status"] == "time_limit" and r.get("gap") is not None]
        macros["RgrbRuns"] = str(len(Gr))
        macros["RgrbOptimal"] = str(len(gs))
        macros["RgrbMaxDev"] = fmt_sci(max(dv)) if dv else MISSING
        macros["RgrbInconsistent"] = str(bad)
        macros["RgrbMaxGap"] = fmt(100 * max(gaps), 1) if gaps else "--"

    lines = ["\\begin{table}[htbp]\n\\centering\n\\small",
             "\\caption{Our algorithms against Gurobi on the exact MIQCP (time limit "
             f"{macros['RTimeLimit']}\\,s, all families pooled). Times are means; \\#opt: runs Gurobi "
             "solved to proven optimality, and its time is the mean over these runs; ratio: median of "
             "Gurobi's time divided by ours over these runs; gap: largest final MIP gap of the runs that "
             "reached the time limit; $\\Delta$: largest relative deviation between Gurobi's optimum and "
             "ours.}",
             "\\label{tab:solvers}",
             "\\begin{tabular}{lrrr|rrrrr}\n\\hline",
             " & & & Ours & \\multicolumn{5}{c}{Gurobi (MIQCP)} \\\\",
             "Setting & $n$ & runs & [ms] & \\#opt & [s] & ratio & gap & $\\Delta$ \\\\\n\\hline"]
    for setting in ("reduction", "augmentation"):
        k = "Red" if setting == "reduction" else "Aug"
        last_all, speed, first = None, None, True
        for n in sizes:
            A = [r for r in A_all if r["setting"] == setting and r["n"] == n]
            if not A:
                continue
            cells = [setting.capitalize() if first else "", str(n), str(len(A)),
                     fmt_ms(mean(r["time"] for r in A))]
            first = False
            R = [r for r in Gr if r["setting"] == setting and r["n"] == n]
            if not R:
                cells += ["--"] * 5
            else:
                sol = [r for r in R if r["status"] in SOLVED]
                ratios = []
                for r in sol:
                    a = [x for x in A if (x["family"], x["seed"]) == (r["family"], r["seed"])]
                    if a:
                        ratios.append(r["time"] / max(a[0]["time"], 1e-9))
                gaps = [r["gap"] for r in R if r["status"] == "time_limit" and r.get("gap") is not None]
                dv = [dev(r) for r in sol if dev(r) is not None]
                cells += [f"{len(sol)}/{len(R)}",
                          fmt_time(mean(r["time"] for r in sol)) if sol else "--",
                          (f"{median(ratios):,.0f}".replace(",", "\\,")) if ratios else "--",
                          (f"{100 * max(gaps):.0f}\\%") if gaps else "--",
                          ("$" + fmt_sci(max(dv)) + "$") if dv else "--"]
                if sol and len(sol) == len(R):
                    last_all = n
                    speed = median(ratios) if ratios else None
                if n == sizes[-1]:
                    macros["RgrbSolvedLargest"] = str((0 if macros["RgrbSolvedLargest"] == MISSING
                                                       else int(macros["RgrbSolvedLargest"])) + len(sol))
                    macros["RgrbRunsLargest"] = str((0 if macros["RgrbRunsLargest"] == MISSING
                                                     else int(macros["RgrbRunsLargest"])) + len(R))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")
        macros[f"RgrbAllSolved{k}"] = str(last_all) if last_all else MISSING
        macros[f"Rspeedup{k}"] = f"{speed:,.0f}".replace(",", "\\,") if speed is not None else MISSING
    lines.append("\\end{tabular}\n\\end{table}")
    write("table_solvers.tex", "\n".join(lines) + "\n")
    return recs


# ======================================================================= eps
def eps_table(macros):
    recs = load("eps.jsonl")
    if not recs:
        write("table_eps.tex", "% no eps results yet: run  python run_bench.py eps\n")
        for k in ("RepsN", "RevalsPerDecade", "RepsTimeRatio"):
            macros[k] = MISSING
        return None
    by = defaultdict(list)
    for r in recs:
        by[r["eps"]].append(r)
    epss = sorted(by, reverse=True)
    n = recs[0]["n"]
    macros["RepsN"] = str(n)
    lines = ["\\begin{table}[htbp]\n\\centering\n\\small",
             f"\\caption{{Dependence of Algorithm~\\ref{{alg3}} on the accuracy $\\varepsilon$ "
             f"(star of paths, $n={n}$, mean over {len({r['seed'] for r in recs})} instances).}}",
             "\\label{tab:eps}",
             "\\begin{tabular}{rrrr}\n\\hline",
             "$\\varepsilon$ & leaves searched & evaluations of $F_i^{+}$ & time [s] \\\\\n\\hline"]
    xs, ys = [], []
    for e in epss:
        rs = by[e]
        ev = mean(r["n_F_evals"] for r in rs)
        xs.append(math.log10(1 / e))
        ys.append(ev)
        lines.append(f"$10^{{{int(round(math.log10(e)))}}}$ & {mean(r['n_leaves'] for r in rs):.1f} & "
                     f"{ev:,.0f} & {fmt_time(mean(r['time'] for r in rs))} \\\\".replace(",", "\\,"))
    lines.append("\\hline\n\\end{tabular}\n\\end{table}")
    write("table_eps.tex", "\n".join(lines) + "\n")
    slope = np.polyfit(xs, ys, 1)[0] if len(xs) > 1 else None
    macros["RevalsPerDecade"] = f"{slope:,.0f}".replace(",", "\\,") if slope is not None else MISSING
    t = {e: mean(r["time"] for r in by[e]) for e in epss}
    if 1e-2 in t and 1e-8 in t:
        macros["RepsTimeRatio"] = fmt(t[1e-8] / t[1e-2], 1)
    else:
        macros["RepsTimeRatio"] = MISSING
    return by


# ======================================================================= figure
def figure(curves, grb, epsby):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COL = {"alg1": "#2a78d6", "alg2": "#eb6834", "alg3": "#1baf7a", "alg4": "#eda100"}
    LS = {"rrt": ":", "rrt-hc": ":", "cat": "-", "cat-hc": "-", "catA2": "--"}
    MK = {"rrt": "o", "rrt-hc": "o", "cat": "s", "cat-hc": "s", "catA2": "D"}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#222222",
                         "xtick.color": "#444444", "ytick.color": "#444444", "lines.linewidth": 1.4})
    npan = 2 + (grb is not None)
    fig, axs = plt.subplots(1, npan, figsize=(7.2 if npan == 3 else 6.3, 3.1),
                            gridspec_kw=dict(width_ratios=[1.25] + [1.0] * (npan - 1)))
    ax = axs[0]
    for (alg, fam), pts in curves.items():
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], LS[fam], marker=MK[fam], ms=3.2,
                color=COL[alg], label=f"Alg. {ALGNUM[alg]} {fam}")
    allx = [p[0] for pts in curves.values() for p in pts]
    if allx:
        x0, x1 = min(allx), max(allx)
        ally = [p[1] for pts in curves.values() for p in pts if p[1] > 0]
        y0 = min(ally)
        for k in (2, 3):
            ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** k], color="#bbbbbb", lw=0.8, zorder=0)
            ax.text(x1, y0 * (x1 / x0) ** k, f" $n^{k}$", color="#777777", fontsize=7, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of vertices $n$"); ax.set_ylabel("time [s]")
    ax.set_title("(a) running time", loc="left", fontsize=8)
    handles_a, labels_a = ax.get_legend_handles_labels()
    k = 1
    if grb is not None:
        ax = axs[k]; k += 1
        SCOL = {"alg": "#2a78d6", "gurobi": "#1baf7a"}
        SNAME = {"alg": "ours", "gurobi": "Gurobi (MIQCP)"}
        for sv in ("alg", "gurobi"):
            for setting, ls, mk in (("reduction", "-", "o"), ("augmentation", "--", "s")):
                rs = [r for r in grb if r["solver"] == sv and r["setting"] == setting]
                ns = sorted({r["n"] for r in rs})
                if not ns:
                    continue
                ax.plot(ns, [mean(r["time"] for r in rs if r["n"] == n) for n in ns], ls, marker=mk, ms=3,
                        color=SCOL[sv], label=f"{SNAME[sv]}, {setting[:3]}.")
        tls = [r["time_limit"] for r in grb if r.get("time_limit")]
        if tls:
            ax.axhline(tls[0], color="#999999", lw=0.8, ls=":")
            ax.text(min(r["n"] for r in grb), tls[0] * 1.15, "time limit", fontsize=6.5, color="#777777")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("number of vertices $n$"); ax.set_ylabel("mean time [s]")
        ax.set_title("(b) ours vs Gurobi", loc="left", fontsize=8)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo / 30, hi)                     # free space for the legend
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    if epsby:
        ax = axs[k]
        es = sorted(epsby, reverse=True)
        xs = [math.log10(1 / e) for e in es]
        ax.plot(xs, [mean(r["n_F_evals"] for r in epsby[e]) for e in es], "-o", ms=3, color="#2a78d6")
        ax.set_xlabel(r"$\log_{10}(1/\varepsilon)$"); ax.set_ylabel("evaluations of $F_i^{+}$")
        ax2 = ax.twinx() if False else None
        ax.set_title(f"({'c' if grb is not None else 'b'}) accuracy, Alg. {ALGNUM['alg3']}", loc="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(handles_a, labels_a, loc="lower center", ncol=5, fontsize=6, frameon=False,
               title="panel (a): colour = algorithm, line style = family", title_fontsize=6)
    fig.savefig(os.path.join(OUT, "iqcp_scaling.pdf"))
    fig.savefig(os.path.join(OUT, "iqcp_scaling.png"), dpi=160)


def write(name, s):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(s)


def env_macros(macros):
    p = os.path.join(RES, "env.json")
    env = json.load(open(p)) if os.path.exists(p) else {}
    macros["RCpu"] = env.get("cpu") or env.get("processor") or MISSING
    macros["RRam"] = str(env.get("ram_gb", MISSING))
    macros["RPython"] = env.get("python", MISSING)
    macros["RGurobi"] = env.get("gurobi") or MISSING


def main():
    macros = {}
    env_macros(macros)
    curves = scaling_tables(macros)
    grb = solvers_table(macros)
    epsby = eps_table(macros)
    figure(curves, grb, epsby)
    body = ["% generated by make_tables.py -- do not edit by hand"]
    for k in sorted(macros):
        v = macros[k].replace("#", "\\#") if k == "RCpu" else macros[k]
        body.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    write("results_macros.tex", "\n".join(body) + "\n")
    print("written to", OUT)
    for k in sorted(macros):
        print(f"  {k:24s} {macros[k]}")


if __name__ == "__main__":
    main()
