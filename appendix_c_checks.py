#!/usr/bin/env python3
"""
appendix_c_checks.py

Robustness checks for the paper "Calibration and Conformal Prediction for Medical Question
Answering in Arabic and English: Evidence from Two Non-Parallel Benchmarks" (Appendix C).

It reads the saved option scores (.jsonl, one record per question) and prints:
  repro  reproduction check of Tables 2 and 3 (same layout)
  c1     Table C1: Jais, Latin-only vs union letter scoring, English and Arabic, with the paired
         bootstrap (2,000 resamples) and exact McNemar test for the Arabic differences
  c2     Table C2: bootstrap intervals (500 replicates, random 25/25/50 split, alpha = 0.10)
         for coverage, set size and set size / K of temperature-scaled conformal prediction
  c3     Table C3: AUROC (95% bootstrap CI, 500 resamples) of the top-two margin vs the maximum
         probability, plus AURC and risk at 20% coverage for both rankings

Usage:
    python appendix_c_checks.py <folder> [--seed 42] [--parts repro c1 c2 c3]

<folder> is the folder that holds results/ and ablation/ (medqa_results in the original runs):
    <folder>/results/Qwen2.5-7B-Instruct_{en,ar}_no_prefill_seed42_n{2816,4927}.jsonl
    <folder>/results/jais-adapted-7b-chat_{en,ar}_no_prefill_union_seed42_n{2816,4927}.jsonl
    <folder>/ablation/jais-adapted-7b-chat_{en,ar}_no_prefill_seed42_n{2816,4927}.jsonl   (Latin only)

Each record needs: id, K, answer, probs, scores, mass (other fields are ignored).
Requires numpy and scipy only.

Notes
- The random splits and bootstrap draws here are not the ones of the main analysis notebook
  (see Appendix C of the paper), so scaled ECE, T and set sizes can differ slightly from
  Tables 3 and 4; point estimates in Table C1 and C3 do not depend on the random numbers.
- Confidence = maximum of `probs`; margin = top-1 minus top-2 of `scores` (log-scores);
  ECE uses 15 equal-width bins with (lower, upper] edges; ties in AURC keep file order.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import binomtest, rankdata

PAD = -1e30  # stands in for -inf in padded (non-existent) options
SETTINGS = [("Qwen", "en"), ("Qwen", "ar"), ("Jais", "en"), ("Jais", "ar")]
MODEL_STEM = {"Qwen": "Qwen2.5-7B-Instruct", "Jais": "jais-adapted-7b-chat"}
N_QUESTIONS = {"en": 2816, "ar": 4927}
LANG_NAME = {"en": "English", "ar": "Arabic"}
AR_LETTERS = {"أ": 0, "ا": 0, "ب": 1, "ج": 2, "د": 3, "ه": 4, "و": 5}

# Values printed in Tables 2 and 3 of the paper (used only for the reproduction check).
# keys: acc, mass, conf, auroc, ece, ece20, ece_sc20, T, brier; tolerance is set in TOL.
PAPER = {
    ("Qwen", "en"): dict(acc=0.561, mass=1.000, conf=0.948, auroc=0.734, ece=0.386, ece20=0.388,
                         ece_sc20=0.068, T=8.71, brier=0.806),
    ("Qwen", "ar"): dict(acc=0.430, mass=1.000, conf=0.916, auroc=0.646, ece=0.486, ece20=0.487,
                         ece_sc20=0.035, T=10.90, brier=1.025),
    ("Jais", "en"): dict(acc=0.369, mass=0.899, conf=0.611, auroc=0.600, ece=0.242, ece20=0.244,
                         ece_sc20=0.042, T=3.40, brier=0.815),
    ("Jais", "ar"): dict(acc=0.308, mass=0.926, conf=0.618, auroc=0.579, ece=0.309, ece20=0.308,
                         ece_sc20=0.042, T=5.70, brier=0.899),
}
# "to three decimals" for point estimates; Appendix C tolerances for split-dependent quantities
TOL = dict(acc=0.0006, mass=0.0006, conf=0.0006, auroc=0.0006, ece=0.0006, brier=0.0006,
           ece20=0.006, ece_sc20=0.006, T=0.25)


def label(model, lang):
    return f"{model}, {LANG_NAME[lang]}"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def answer_to_index(a):
    """Option label -> 0-based index. Accepts ints, 'A'..'F' (any case) and Arabic letters."""
    if isinstance(a, (int, np.integer)):
        return int(a)
    s = str(a).strip()
    if s.isdigit():
        return int(s)
    if s[:1] in AR_LETTERS:
        return AR_LETTERS[s[:1]]
    return ord(s[:1].upper()) - ord("A")


class Run:
    """One .jsonl results file, as padded arrays (questions x max options)."""

    def __init__(self, path):
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"{path} is empty")

        def sort_key(r):
            try:
                return (0, int(r["id"]), "")
            except (TypeError, ValueError):
                return (1, 0, str(r["id"]))

        records.sort(key=sort_key)
        n = len(records)
        self.path = path
        self.n = n
        self.ids = [str(r["id"]) for r in records]
        self.K = np.array([int(r["K"]) for r in records])
        kmax = int(self.K.max())
        self.probs = np.zeros((n, kmax))
        self.scores = np.full((n, kmax), PAD)
        for i, r in enumerate(records):
            k = int(self.K[i])
            p, s = np.asarray(r["probs"], float), np.asarray(r["scores"], float)
            if len(p) != k or len(s) != k:
                raise ValueError(f"{path}: id {r['id']} has K={k} but {len(p)} probs / {len(s)} scores")
            self.probs[i, :k] = p
            self.scores[i, :k] = np.maximum(s, PAD)  # -inf -> PAD
        self.valid = np.arange(kmax)[None, :] < self.K[:, None]
        self.y = np.array([answer_to_index(r["answer"]) for r in records])
        if (self.y < 0).any() or (self.y >= self.K).any():
            raise ValueError(f"{path}: some answers fall outside the options")
        self.mass = np.array([float(r["mass"]) for r in records])
        self.pred = self.probs.argmax(1)
        self.conf = self.probs.max(1)
        self.correct = (self.pred == self.y).astype(float)
        top2 = np.sort(self.scores, axis=1)
        self.margin = top2[:, -1] - top2[:, -2]
        # consistency with the stored fields (warning only)
        agree = np.mean([self._same_pred(r, self.pred[i]) for i, r in enumerate(records)])
        conf_ok = np.mean(np.abs(np.array([float(r.get("conf", c)) for r, c in zip(records, self.conf)])
                                 - self.conf) < 1e-4)
        self.warning = None
        if agree < 0.999 or conf_ok < 0.999:
            self.warning = (f"stored pred agrees with argmax(probs) for {100 * agree:.1f}% and stored conf "
                            f"with max(probs) for {100 * conf_ok:.1f}% of questions")

    @staticmethod
    def _same_pred(record, idx):
        try:
            return answer_to_index(record["pred"]) == idx
        except (KeyError, IndexError, ValueError):
            return True  # field missing or unreadable: skip the check


def find_file(folder, sub, model, lang, union):
    stem = MODEL_STEM[model]
    tag = "_union" if union else ""
    pattern = os.path.join(folder, sub, f"{stem}_{lang}_no_prefill{tag}_seed42_n{N_QUESTIONS[lang]}.jsonl")
    hits = glob.glob(pattern)
    if not hits and union and sub == "results":  # fall back to a name without the union marker
        alt = os.path.join(folder, sub, f"{stem}_{lang}_no_prefill_seed42_n{N_QUESTIONS[lang]}.jsonl")
        hits = glob.glob(alt)
        if hits:
            print(f"note: no 'union' file for {label(model, lang)}; using {os.path.basename(hits[0])}")
    if not hits:
        sys.exit(f"error: no file matches {pattern}")
    return hits[0]


def load_runs(folder, need_ablation):
    main, latin = {}, {}
    for model, lang in SETTINGS:
        main[(model, lang)] = Run(find_file(folder, "results", model, lang, union=(model == "Jais")))
    if need_ablation:
        for lang in ("en", "ar"):
            latin[lang] = Run(find_file(folder, "ablation", "Jais", lang, union=False))
    for run in list(main.values()) + list(latin.values()):
        if run.warning:
            print(f"warning: {os.path.basename(run.path)}: {run.warning}")
    return main, latin


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def ece(conf, correct, bins=15):
    """Expected calibration error, equal-width bins with (lower, upper] edges."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1)
    conf_sum = np.bincount(idx, weights=conf, minlength=bins)
    acc_sum = np.bincount(idx, weights=correct, minlength=bins)
    return float(np.abs(acc_sum - conf_sum).sum() / len(conf))


def auroc(score, positive):
    """AUROC of `score` for predicting positive == 1 (rank formula, ties get average ranks)."""
    positive = positive.astype(bool)
    n1 = positive.sum()
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def softmax(scores, T=1.0):
    z = scores / T
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def fit_temperature(scores, y):
    """Temperature T minimising the negative log-likelihood of softmax(scores / T)."""
    true_score = scores[np.arange(len(y)), y]

    def nll(beta):  # beta = 1 / T; the NLL is convex in beta
        return float(np.mean(logsumexp(scores * beta, axis=1) - true_score * beta))

    res = minimize_scalar(nll, bounds=(1e-3, 20.0), method="bounded", options={"xatol": 1e-7})
    return 1.0 / res.x


def risk_curve(confidence, correct):
    """Risk (error rate) when answering the top-k most confident questions, k = 1..n."""
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    return np.cumsum(errors) / np.arange(1, len(errors) + 1)


def risk_at(curve, fraction):
    return float(curve[max(int(round(fraction * len(curve))), 1) - 1])


def pct(x, nd=3):
    return f"{x:.{nd}f}"


def ci_text(lo, hi, nd=3):
    return f"{lo:.{nd}f} to {hi:.{nd}f}"


def percentile_ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def print_table(title, header, rows, caption=None):
    print(f"\n{title}")
    table = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(str(r[j])) for r in table) for j in range(len(header))]
    for i, r in enumerate(table):
        print("| " + " | ".join(str(c).ljust(widths[j]) for j, c in enumerate(r)) + " |")
        if i == 0:
            print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    if caption:
        print(caption)


# --------------------------------------------------------------------------------------
# (a) Reproduction of Tables 2 and 3
# --------------------------------------------------------------------------------------
def part_repro(main, seed):
    rng_seed = [seed, 0]
    rows2, rows3, misses, n_checked = [], [], [], 0
    for model, lang in SETTINGS:
        r = main[(model, lang)]
        n = r.n
        brier = float(np.mean((r.probs ** 2).sum(1) - 2 * r.probs[np.arange(n), r.y] + 1))
        uniform = float(np.mean(1 - 1 / r.K))
        rng = np.random.default_rng(rng_seed)
        temps, raw20, sc20 = [], [], []
        for _ in range(20):  # 20 random 50/50 splits: T fitted on one half, ECE on the other
            perm = rng.permutation(n)
            fit, test = perm[: n // 2], perm[n // 2:]
            T = fit_temperature(r.scores[fit], r.y[fit])
            P = softmax(r.scores[test], T)
            temps.append(T)
            raw20.append(ece(r.conf[test], r.correct[test]))
            sc20.append(ece(P.max(1), (P.argmax(1) == r.y[test]).astype(float)))
        got = dict(acc=r.correct.mean(), mass=r.mass.mean(), conf=r.conf.mean(),
                   auroc=auroc(r.margin, r.correct), ece=ece(r.conf, r.correct), ece20=np.mean(raw20),
                   ece_sc20=np.mean(sc20), T=np.mean(temps), brier=brier)
        lab = label(model, lang)
        rows2.append([lab, f"{n:,}", pct(np.mean(1 / r.K)), pct(got["acc"]), pct(got["mass"]),
                      pct(got["conf"]), pct(got["auroc"])])
        rows3.append([lab, pct(got["ece"]), pct(got["ece20"]), pct(got["ece_sc20"]),
                      f"{np.mean(temps):.2f}, {np.std(temps, ddof=1):.2f}", f"{brier:.3f} ({uniform:.3f})"])
        for key, want in PAPER[(model, lang)].items():
            n_checked += 1
            if abs(got[key] - want) > TOL[key] + 1e-9:
                misses.append(f"{lab}: {key} = {got[key]:.4f}, paper {want} (tolerance {TOL[key]})")
    print_table("Table 2 (reproduction). Accuracy, format compliance, mean confidence and AUROC of the margin.",
                ["Model, language", "n", "Chance", "Accuracy", "Mass on letters", "Mean conf.", "AUROC"], rows2)
    print_table("Table 3 (reproduction). Calibration before and after temperature scaling (20 random 50/50 splits).",
                ["Model, language", "ECE raw", "ECE raw, 20 splits", "ECE scaled, 20 splits", "T (mean, SD)",
                 "Brier (uniform guess)"], rows3)
    print(f"\nReproduction check: {n_checked - len(misses)} of {n_checked} values agree with the paper "
          f"(accuracy, mass, confidence, AUROC, raw ECE, Brier to three decimals; split-dependent "
          f"scaled ECE within 0.006 and T within 0.25).")
    for m in misses:
        print("  DIFFERS:", m)


# --------------------------------------------------------------------------------------
# (b) Table C1: Latin-only vs union scoring
# --------------------------------------------------------------------------------------
def summarize(run):
    return dict(acc=run.correct.mean(), conf=run.conf.mean(), mass=run.mass.mean(),
                ece=ece(run.conf, run.correct), auroc=auroc(run.margin, run.correct),
                share_c=float(np.mean(run.pred == 2)))


def part_c1(main, latin, seed, n_boot):
    cols, summaries, differ = [], {}, {}
    for lang in ("en", "ar"):
        union, lat = main[("Jais", lang)], latin[lang]
        if lat.ids != union.ids:
            sys.exit(f"error: question ids differ between the Latin-only and union files ({lang})")
        summaries[(lang, "lat")], summaries[(lang, "uni")] = summarize(lat), summarize(union)
        differ[lang] = int(np.sum(lat.pred != union.pred))
    rows = []
    spec = [("Accuracy", "acc"), ("Mean confidence", "conf"), ("Mean mass on letters", "mass"),
            ("ECE, raw", "ece"), ("AUROC of top-two margin", "auroc"),
            ("Share of predictions that are option C", "share_c")]
    for name, key in spec:
        rows.append([name] + [pct(summaries[(lang, v)][key]) for lang in ("en", "ar") for v in ("lat", "uni")])
    rows.append(["Questions with a different predicted option",
                 f"{differ['en']:,} of {N_QUESTIONS['en']:,}", "(reference)",
                 f"{differ['ar']:,} of {N_QUESTIONS['ar']:,}", "(reference)"])
    print_table("Table C1. Jais on English and Arabic: Latin-only scoring versus union scoring "
                "(used for the main results).",
                ["Metric", "English, Latin only", "English, union", "Arabic, Latin only", "Arabic, union"], rows)

    # paired bootstrap and exact McNemar, Arabic (Latin only minus union)
    lat, uni = latin["ar"], main[("Jais", "ar")]
    n = uni.n
    point = dict(acc=lat.correct.mean() - uni.correct.mean(),
                 auroc=auroc(lat.margin, lat.correct) - auroc(uni.margin, uni.correct),
                 ece=ece(lat.conf, lat.correct) - ece(uni.conf, uni.correct))
    rng = np.random.default_rng([seed, 1])
    draws = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        draws["acc"][b] = lat.correct[i].mean() - uni.correct[i].mean()
        draws["auroc"][b] = auroc(lat.margin[i], lat.correct[i]) - auroc(uni.margin[i], uni.correct[i])
        draws["ece"][b] = ece(lat.conf[i], lat.correct[i]) - ece(uni.conf[i], uni.correct[i])
    only_lat = int(np.sum((lat.correct == 1) & (uni.correct == 0)))
    only_uni = int(np.sum((lat.correct == 0) & (uni.correct == 1)))
    p = binomtest(min(only_lat, only_uni), only_lat + only_uni, 0.5).pvalue if only_lat + only_uni else 1.0
    print(f"\nPaired bootstrap, Arabic, Latin only minus union ({n_boot:,} resamples over questions, "
          f"95% percentile interval):")
    for name, key in [("accuracy", "acc"), ("AUROC of the margin", "auroc"), ("raw ECE", "ece")]:
        lo, hi = percentile_ci(draws[key])
        print(f"  {name}: {point[key]:+.3f} ({lo:+.3f} to {hi:+.3f})")
    print(f"Exact McNemar test on accuracy (Arabic): p = {p:.3f} "
          f"({only_lat:,} questions correct only with Latin letters, {only_uni:,} only with union scoring)")
    en_diff = differ["en"]
    print(f"English: the two scorings differ in the predicted option for {en_diff:,} of {N_QUESTIONS['en']:,} questions.")


# --------------------------------------------------------------------------------------
# (c) Table C2: bootstrap intervals for conformal coverage and set size
# --------------------------------------------------------------------------------------
def conformal_replicate(r, rng, alpha):
    n = r.n
    n_fit = n_cal = int(0.25 * n)
    perm = rng.permutation(n)
    parts = [perm[:n_fit], perm[n_fit:n_fit + n_cal], perm[n_fit + n_cal:]]
    fit, cal, test = [rng.choice(p, size=len(p), replace=True) for p in parts]  # resample within each part
    T = fit_temperature(r.scores[fit], r.y[fit])
    p_cal, p_test = softmax(r.scores[cal], T), softmax(r.scores[test], T)
    s = np.sort(1.0 - p_cal[np.arange(len(cal)), r.y[cal]])
    k = math.ceil((len(cal) + 1) * (1 - alpha))
    q = s[k - 1] if k <= len(s) else np.inf
    in_set = ((1.0 - p_test) <= q) & r.valid[test]
    size = in_set.sum(1)
    coverage = in_set[np.arange(len(test)), r.y[test]].mean()
    return coverage, size.mean(), (size / r.K[test]).mean()


def part_c2(main, seed, n_boot, alpha=0.10):
    rows = []
    for j, (model, lang) in enumerate(SETTINGS):
        r = main[(model, lang)]
        rng = np.random.default_rng([seed, 2, j])
        out = np.array([conformal_replicate(r, rng, alpha) for _ in range(n_boot)])
        rows.append([label(model, lang)] + [ci_text(*percentile_ci(out[:, c]), nd=nd)
                                            for c, nd in ((0, 3), (1, 2), (2, 3))])
    print_table(f"Table C2. 95% bootstrap intervals ({n_boot} replicates) for coverage and set size of split "
                f"conformal prediction with temperature-scaled probabilities at alpha = {alpha:.2f}.",
                ["Model, language", f"Coverage (target {1 - alpha:.2f})", "Set size, options", "Set size / K"], rows)


# --------------------------------------------------------------------------------------
# (d) Table C3: AUROC and AURC of margin vs maximum probability
# --------------------------------------------------------------------------------------
def part_c3(main, seed, n_boot):
    rows_auroc, rows_risk = [], []
    for j, (model, lang) in enumerate(SETTINGS):
        r = main[(model, lang)]
        rng = np.random.default_rng([seed, 3, j])
        d_margin, d_conf = np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            i = rng.integers(0, r.n, r.n)
            d_margin[b] = auroc(r.margin[i], r.correct[i])
            d_conf[b] = auroc(r.conf[i], r.correct[i])
        a_margin, a_conf = auroc(r.margin, r.correct), auroc(r.conf, r.correct)
        rows_auroc.append([label(model, lang),
                           f"{a_margin:.3f} ({ci_text(*percentile_ci(d_margin))})",
                           f"{a_conf:.3f} ({ci_text(*percentile_ci(d_conf))})"])
        c_margin, c_conf = risk_curve(r.margin, r.correct), risk_curve(r.conf, r.correct)
        rows_risk.append([label(model, lang), f"{c_margin.mean():.4f}", f"{c_conf.mean():.4f}",
                          f"{risk_at(c_margin, 0.2):.3f}", f"{risk_at(c_conf, 0.2):.3f}"])
    print_table("Table C3. AUROC for predicting correctness from the top-two margin and from the maximum option "
                f"probability; 95% percentile bootstrap intervals over questions ({n_boot} resamples).",
                ["Model, language", "Margin (95% CI)", "Maximum probability (95% CI)"], rows_auroc)
    print_table("AURC (mean risk over all coverage levels) and risk when answering the 20% most confident "
                "questions, ranked by the margin or by the maximum probability.",
                ["Model, language", "AURC, margin", "AURC, max prob.", "Risk at 20%, margin",
                 "Risk at 20%, max prob."], rows_risk)


# --------------------------------------------------------------------------------------
def main_cli():
    ap = argparse.ArgumentParser(description="Appendix C robustness checks (see the module docstring).")
    ap.add_argument("folder", help="folder containing results/ and ablation/ (e.g. medqa_results)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parts", nargs="+", choices=["repro", "c1", "c2", "c3"],
                    default=["repro", "c1", "c2", "c3"])
    ap.add_argument("--n-boot-paired", type=int, default=2000, help="C.1 paired bootstrap resamples")
    ap.add_argument("--n-boot-conformal", type=int, default=500, help="C.2 bootstrap replicates")
    ap.add_argument("--n-boot-auroc", type=int, default=500, help="C.3 bootstrap resamples")
    args = ap.parse_args()
    if not os.path.isdir(args.folder):
        sys.exit(f"error: {args.folder} is not a folder")

    main, latin = load_runs(args.folder, need_ablation="c1" in args.parts)
    if "repro" in args.parts:
        part_repro(main, args.seed)
    if "c1" in args.parts:
        part_c1(main, latin, args.seed, args.n_boot_paired)
    if "c2" in args.parts:
        part_c2(main, args.seed, args.n_boot_conformal)
    if "c3" in args.parts:
        part_c3(main, args.seed, args.n_boot_auroc)


if __name__ == "__main__":
    main_cli()
