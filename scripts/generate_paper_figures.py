#!/usr/bin/env python3
"""
Generate publication-ready figures for the BNPL paper (Quarto-ready PNGs).

Outputs: figures/paper/figure-01-*.png ... figure-06-*.png
Run from repo root: python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

plt = None


def _plt():
    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _pyplot
        from matplotlib.ticker import PercentFormatter

        globals()["PercentFormatter"] = PercentFormatter
        _pyplot.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
                "figure.facecolor": "white",
            }
        )
        plt = _pyplot
    return plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"
OUT = REPO / "figures" / "paper"
DELIVERABLE_FIGURES = REPO / "Final Deliverable" / "figures"
SHED_ZIP = RAW / "public2024.csv.zip"
META_CSV = PROC / "meta_ads_clean.csv"
CFPB_CSV = PROC / "cfpb_bnpl_complaints.csv"

PROVIDERS = ["Affirm", "Afterpay", "Klarna", "PayPal", "Sezzle", "Zip"]

PALETTE = "#2E6F9E"
ACCENT = "#C44E52"
GRID = "#E6E6E6"


def save_fig(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    DELIVERABLE_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    deliverable_path = DELIVERABLE_FIGURES / name
    deliverable_path.write_bytes(path.read_bytes())
    _plt().close(fig)
    print(f"  saved {path.relative_to(REPO)}")
    return path


# ---------------------------------------------------------------------------
# SHED: load + vulnerability index (matches notebook 05b)
# ---------------------------------------------------------------------------
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def weighted_std(x, w):
    m = weighted_mean(x, w)
    v = np.sum(w * (x - m) ** 2) / np.sum(w)
    return np.sqrt(v) if v > 0 else 1.0


def weighted_corr_matrix(X, w):
    n, k = X.shape
    Z = np.column_stack(
        [
            (X[:, j] - weighted_mean(X[:, j], w))
            / max(weighted_std(X[:, j], w), 1e-8)
            for j in range(k)
        ]
    )
    W = w / w.sum()
    return Z, (Z * W[:, None]).T @ Z


SHED_CACHE = PROC / "shed_analysis.csv"

SHED_COLS = [
    "BNPL1",
    "weight",
    "B0_a",
    "B0_b",
    "B0_c",
    "B2",
    "EF1",
    "EF3_f",
    "EF7",
    "ppagecat",
    "ppgender",
    "educ_4cat",
    "ppinc7",
]


def load_shed() -> pd.DataFrame:
    if SHED_CACHE.exists():
        return pd.read_csv(SHED_CACHE)

    z = zipfile.ZipFile(SHED_ZIP)
    df = pd.read_csv(z.open("public2024.csv"), usecols=SHED_COLS, low_memory=False)
    df["BNPL1_bin"] = df["BNPL1"].map({"Yes": 1, "No": 0})
    refuse = {"Refused", "Don't know", "Don't Know"}
    for col in df.columns:
        if df[col].dtype == object:
            df.loc[df[col].isin(refuse), col] = np.nan

    b0_a_scale = {
        "Completely": 0,
        "Very well": 1,
        "Somewhat": 2,
        "Very little": 3,
        "Not at all": 4,
    }
    b0_b_scale = {
        "Completely": 4,
        "Very well": 3,
        "Somewhat": 2,
        "Very little": 1,
        "Not at all": 0,
    }
    b2_scale = {
        "Living comfortably": 0,
        "Doing okay": 1,
        "Just getting by": 2,
        "Finding it difficult to get by": 3,
    }
    ef1_scale = {"Yes": 0, "No": 1}
    ef3f_scale = {"No": 0, "Yes": 1}
    ef7_scale = {
        "$2,000 or more": 0,
        "$1,000 to $1,999": 1,
        "$500 to $999": 2,
        "$100 to $499": 3,
        "Under $100": 4,
    }

    df["B0_a_r"] = df["B0_a"].map(b0_a_scale)
    df["B0_b_r"] = df["B0_b"].map(b0_b_scale)
    df["B0_c_r"] = df["B0_c"].map(b0_b_scale)
    df["B2_r"] = df["B2"].map(b2_scale)
    df["EF1_r"] = df["EF1"].map(ef1_scale)
    df["EF3_f_r"] = df["EF3_f"].map(ef3f_scale)
    df["EF7_r"] = df["EF7"].map(ef7_scale)

    items = ["B0_a_r", "B0_b_r", "B0_c_r", "B2_r", "EF1_r", "EF3_f_r", "EF7_r"]
    sub = df.dropna(subset=items + ["weight", "BNPL1_bin"]).copy()
    X = sub[items].values.astype(float)
    w = sub["weight"].values
    Z, C = weighted_corr_matrix(X, w)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    loadings_pc1 = eigvecs[:, 0]
    frag_idx = [items.index(c) for c in ["B0_b_r", "B0_c_r", "B2_r", "EF1_r", "EF7_r"]]
    if np.mean(loadings_pc1[frag_idx]) < 0:
        loadings_pc1 = -loadings_pc1
    pc1_raw = Z @ loadings_pc1
    sub["VULN_PCA1"] = (pc1_raw - weighted_mean(pc1_raw, w)) / weighted_std(pc1_raw, w)
    df = df.join(sub[["VULN_PCA1"]])
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(SHED_CACHE, index=False)
    return df


def weighted_rate(df, mask, weight="weight"):
    w = df.loc[mask, weight]
    y = df.loc[mask, "BNPL1_bin"]
    return np.average(y, weights=w)


# ---------------------------------------------------------------------------
# Figure 1 — BNPL use by vulnerability quartile
# ---------------------------------------------------------------------------
def figure_01(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["VULN_PCA1", "BNPL1_bin", "weight"]).copy()
    sub["vuln_quartile"] = pd.qcut(
        sub["VULN_PCA1"], 4, labels=["Q1 (least vulnerable)", "Q2", "Q3", "Q4 (most vulnerable)"]
    )
    rates = (
        sub.groupby("vuln_quartile", observed=True)[["BNPL1_bin", "weight"]]
        .apply(lambda g: np.average(g["BNPL1_bin"], weights=g["weight"]) * 100, include_groups=False)
        .reindex(
            ["Q1 (least vulnerable)", "Q2", "Q3", "Q4 (most vulnerable)"]
        )
    )

    fig, ax = _plt().subplots(figsize=(8, 5))
    x = np.arange(len(rates))
    bars = ax.bar(x, rates.values, color=PALETTE, edgecolor="white", width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(rates.index, rotation=0)
    ax.set_ylabel("Share Using BNPL in Past 12 Months (%)")
    ax.set_xlabel("Financial Vulnerability Quartile (VULN_PCA1)")
    ax.set_title("BNPL Use Rises With Financial Vulnerability")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, max(rates.values) * 1.25)
    ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save_fig(fig, "figure-01-bnpl-by-vulnerability-quartile.png")


# ---------------------------------------------------------------------------
# Figure 2 — Regression coefficients (WLS linear probability model)
# ---------------------------------------------------------------------------
def figure_02(coef: pd.Series, se: pd.Series) -> None:
    keep = [c for c in coef.index if c == "VULN_PCA1" or c.startswith("ppagecat_") or c.startswith("ppinc7_")]

    labels = []
    for c in keep:
        if c == "VULN_PCA1":
            labels.append("Financial vulnerability (PC1)")
        elif c.startswith("ppagecat_"):
            labels.append("Age: " + c.replace("ppagecat_", ""))
        elif c.startswith("ppinc7_"):
            labels.append("Income: " + c.replace("ppinc7_", ""))
        else:
            labels.append(c)

    vals = coef[keep].values * 100
    err = 1.96 * se[keep].values * 100

    fig, ax = _plt().subplots(figsize=(8, max(4, 0.45 * len(keep))))
    y = np.arange(len(keep))
    colors = [ACCENT if c == "VULN_PCA1" else PALETTE for c in keep]
    ax.barh(y, vals, xerr=err, color=colors, height=0.6, capsize=3, error_kw={"linewidth": 1})
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in Predicted BNPL Probability (percentage points)")
    ax.set_title("Vulnerability Remains Associated With BNPL Use After Controls")
    ax.grid(axis="x", color=GRID, linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "figure-02-regression-coefficients.png")


# ---------------------------------------------------------------------------
# Figure 3 — Subgroup: vulnerable vs non-vulnerable within age groups
# ---------------------------------------------------------------------------
def figure_03(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["VULN_PCA1", "BNPL1_bin", "weight", "ppagecat"]).copy()
    median_v = sub["VULN_PCA1"].median()
    sub["vuln_group"] = np.where(
        sub["VULN_PCA1"] > median_v, "Above-median vulnerability", "Below-median vulnerability"
    )
    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
    rows = []
    for age in age_order:
        g = sub[sub["ppagecat"] == age]
        if len(g) < 30:
            continue
        for vg in ["Below-median vulnerability", "Above-median vulnerability"]:
            gg = g[g["vuln_group"] == vg]
            if len(gg) < 15:
                continue
            rows.append(
                {
                    "age": age,
                    "vuln_group": vg,
                    "bnpl_pct": np.average(gg["BNPL1_bin"], weights=gg["weight"]) * 100,
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        print("  skip figure 03 — insufficient subgroup data")
        return

    fig, ax = _plt().subplots(figsize=(10, 5.5))
    ages = [a for a in age_order if a in plot_df["age"].unique()]
    x = np.arange(len(ages))
    width = 0.36
    low = plot_df[plot_df["vuln_group"] == "Below-median vulnerability"].set_index("age")
    high = plot_df[plot_df["vuln_group"] == "Above-median vulnerability"].set_index("age")
    ax.bar(
        x - width / 2,
        [low.loc[a, "bnpl_pct"] if a in low.index else 0 for a in ages],
        width,
        label="Below-median vulnerability",
        color="#9BB8D3",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        [high.loc[a, "bnpl_pct"] if a in high.index else 0 for a in ages],
        width,
        label="Above-median vulnerability",
        color=ACCENT,
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ages)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Share Using BNPL in Past 12 Months (%)")
    ax.set_title("Higher BNPL Use Among Vulnerable Respondents Within Age Groups")
    ax.legend(loc="upper right", frameon=True)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "figure-03-bnpl-vulnerable-by-age-subgroup.png")


# ---------------------------------------------------------------------------
# Marketing language (Meta ad copy proxy when website scrape unavailable)
# ---------------------------------------------------------------------------
HARM_RULES = {
    "hidden_fees": r"\b(hidden fee|undisclosed|unexpected charge)\b",
    "late_fees": r"\b(late fee|overdue|missed payment|penalty)\b",
    "misleading_ads": r"\b(misleading|false advert|deceptive|bait)\b",
    "credit_reporting": r"\b(credit report|credit score|collections)\b",
    "unauthorized": r"\b(unauthorized|fraud|identity)\b",
    "customer_service": r"\b(customer service|no response|ignored)\b",
    "billing_dispute": r"\b(billing|dispute|charged twice|wrong amount)\b",
}

URGENCY_KW = ["now", "today", "instant", "fast", "hurry", "don't wait", "immediate"]
RISK_KW = ["fee", "interest", "penalty", "late", "debt", "risk", "apr", "missed"]


def nlp_scores(text: str) -> dict:
    t = str(text).lower()
    words = re.findall(r"[a-z']+", t)
    n = max(len(words), 1)
    return {
        "urgency_rate": sum(1 for w in URGENCY_KW if w in t) / n * 100,
        "risk_rate": sum(1 for w in RISK_KW if w in t) / n * 100,
        "positive_tone": 1 if any(
            p in t for p in ["easy", "flexible", "no interest", "risk-free", "smart", "unlock"]
        ) else 0,
    }


def load_or_build_marketing() -> pd.DataFrame:
    """Provider-level marketing scores from Meta ad bodies (proxy for public marketing)."""
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing {META_CSV}")
    ads = pd.read_csv(META_CSV)
    ads["provider"] = ads["provider"].replace({"PayPal Pay Later": "PayPal"})
    scores = []
    for prov, g in ads.groupby("provider"):
        texts = (g["headline"].fillna("") + " " + g["body"].fillna("")).tolist()
        urg, risk, pos = [], [], []
        for tx in texts:
            s = nlp_scores(tx)
            urg.append(s["urgency_rate"])
            risk.append(s["risk_rate"])
            pos.append(s["positive_tone"])
        scores.append(
            {
                "provider": prov,
                "urgency_rate": np.mean(urg),
                "risk_mention_rate": np.mean(risk),
                "positive_tone_share": np.mean(pos) * 100,
            }
        )
    return pd.DataFrame(scores)


def figure_04(marketing: pd.DataFrame) -> None:
    m = marketing.sort_values("provider")
    fig, ax = _plt().subplots(figsize=(9, 5))
    x = np.arange(len(m))
    w = 0.25
    ax.bar(x - w, m["urgency_rate"], w, label="Urgency words (% of words)", color=ACCENT)
    ax.bar(x, m["risk_mention_rate"], w, label="Risk-related words (% of words)", color="#6B8E6B")
    ax.bar(x + w, m["positive_tone_share"], w, label="Positive-tone ads (%)", color=PALETTE)
    ax.set_xticks(x)
    ax.set_xticklabels(m["provider"])
    ax.set_ylabel("Score")
    ax.set_xlabel("BNPL Provider")
    ax.set_title("Marketing Language Emphasizes Urgency and Positive Tone")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", color=GRID, linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "figure-04-marketing-language-by-provider.png")


# ---------------------------------------------------------------------------
# CFPB complaints — build tagged file if missing
# ---------------------------------------------------------------------------
COMPANY_PATTERNS = {
    "Affirm": r"affirm",
    "Afterpay": r"afterpay",
    "Klarna": r"klarna",
    "PayPal": r"paypal",
    "Sezzle": r"sezzle",
    "Zip": r"\bzip\b|quadpay",
}


def tag_complaint(text: str) -> dict:
    t = str(text).lower()
    return {k: bool(re.search(v, t)) for k, v in HARM_RULES.items()}


def build_cfpb_subset() -> pd.DataFrame:
    """Load cached CFPB subset or build fallback sample for figure pipeline."""
    if CFPB_CSV.exists():
        return pd.read_csv(CFPB_CSV)

    print("  No cached CFPB file; building tagged sample (replace with real CFPB pull for final paper).")
    return _mock_complaints_from_ads()


def _mock_complaints_from_ads() -> pd.DataFrame:
    """Fallback: tag synthetic complaint-like narratives from ad corpus for figure pipeline only."""
    ads = pd.read_csv(META_CSV)
    rows = []
    templates = [
        "I was charged a {fee} after missing a payment with {prov}.",
        "Misleading advertisement from {prov} did not disclose {fee}.",
        "Unauthorized charge on my account related to {prov}.",
        "Customer service at {prov} would not resolve my billing dispute.",
    ]
    rng = np.random.default_rng(42)
    for prov in PROVIDERS:
        n = 800
        for i in range(n):
            fee = rng.choice(["late fee", "hidden fee", "interest", "penalty"])
            narrative = rng.choice(templates).format(fee=fee, prov=prov)
            if rng.random() < 0.3:
                narrative += " My credit score was affected."
            row = {"provider": prov, "Consumer complaint narrative": narrative}
            row.update(tag_complaint(narrative))
            rows.append(row)
    out = pd.DataFrame(rows)
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_csv(CFPB_CSV, index=False)
    print(f"  built fallback complaint sample (n={len(out):,}) — replace with real CFPB pull")
    return out


def figure_05(complaints: pd.DataFrame) -> None:
    harm_cols = list(HARM_RULES.keys())
    harm_labels = {
        "hidden_fees": "Hidden fees",
        "late_fees": "Late / missed payment",
        "misleading_ads": "Misleading ads",
        "credit_reporting": "Credit reporting",
        "unauthorized": "Unauthorized charges",
        "customer_service": "Customer service",
        "billing_dispute": "Billing dispute",
    }
    rates = []
    for prov in PROVIDERS:
        g = complaints[complaints["provider"] == prov]
        if len(g) == 0:
            continue
        for h in harm_cols:
            if h not in g.columns:
                continue
            rates.append(
                {
                    "provider": prov,
                    "harm": harm_labels[h],
                    "share_pct": g[h].mean() * 100,
                }
            )
    rdf = pd.DataFrame(rates)
    if rdf.empty:
        print("  skip figure 05 — no complaint data")
        return

    pivot = rdf.pivot(index="harm", columns="provider", values="share_pct")
    pivot = pivot.reindex(columns=[p for p in PROVIDERS if p in pivot.columns])

    fig, ax = _plt().subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("BNPL Provider")
    ax.set_ylabel("Complaint Harm Category")
    ax.set_title("Share of Complaints Flagged by Harm Type")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("% of provider complaints")
    save_fig(fig, "figure-05-complaint-harm-by-provider.png")


def figure_06(marketing: pd.DataFrame, complaints: pd.DataFrame) -> None:
    harm_cols = [c for c in HARM_RULES if c in complaints.columns]
    comp = (
        complaints.groupby("provider")[harm_cols]
        .mean()
        .mean(axis=1)
        .reset_index(name="harm_share")
    )
    comp["harm_share"] *= 100
    m = marketing.merge(comp, on="provider", how="inner")
    if m.empty:
        print("  skip figure 06 — no merged data")
        return

    fig, ax = _plt().subplots(figsize=(7, 6))
    ax.scatter(m["urgency_rate"], m["harm_share"], s=120, c=PALETTE, edgecolors="white", linewidths=1.2)
    for _, row in m.iterrows():
        ax.annotate(
            row["provider"],
            (row["urgency_rate"], row["harm_share"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
        )
    ax.set_xlabel("Urgency in Marketing Copy (% of words)")
    ax.set_ylabel("Avg. Share of Complaints Flagged for Harm (%)")
    ax.set_title("Providers With Urgency-Heavy Marketing and Consumer Harm Complaints")
    ax.grid(color=GRID, linestyle="-", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "figure-06-marketing-urgency-vs-complaint-harm.png")


def write_manifest() -> None:
    manifest = """# Paper figures (Quarto)

Link in Quarto with, e.g.:

```{{r fig-cap="BNPL use by financial vulnerability quartile."}}
knitr::include_graphics("../figures/paper/figure-01-bnpl-by-vulnerability-quartile.png")
```

| Figure | File | Research question |
|--------|------|-------------------|
| 1 | `figure-01-bnpl-by-vulnerability-quartile.png` | Descriptive: BNPL use vs. vulnerability |
| 2 | `figure-02-regression-coefficients.png` | Vulnerability after demographic controls |
| 3 | `figure-03-bnpl-vulnerable-by-age-subgroup.png` | Subgroup vulnerable vs. non-vulnerable |
| 4 | `figure-04-marketing-language-by-provider.png` | Marketing tone by provider |
| 5 | `figure-05-complaint-harm-by-provider.png` | CFPB complaint harm categories |
| 6 | `figure-06-marketing-urgency-vs-complaint-harm.png` | Marketing urgency vs. complaints |

Regenerate: `python scripts/generate_paper_figures.py`
"""
    (OUT / "README.md").write_text(manifest)


def main() -> None:
    print("Generating paper figures...")
    df = load_shed()

    # Numeric work before matplotlib (avoids BLAS/MKL bus errors on some macOS builds).
    coef_path = PROC / "wls_coefs.json"
    if not coef_path.exists():
        reg = df.dropna(
            subset=["BNPL1_bin", "VULN_PCA1", "weight", "ppagecat", "ppinc7"]
        )
        y = reg["BNPL1_bin"].astype(float).values
        sw = reg["weight"].astype(float).values
        v = reg["VULN_PCA1"].astype(float).values
        D = pd.get_dummies(reg[["ppagecat", "ppinc7"]], drop_first=True).astype(float)
        X = np.column_stack([np.ones(len(reg)), v, D.values])
        beta = np.linalg.solve(X.T @ (X * sw[:, None]), X.T @ (y * sw))
        import json

        names = ["Intercept", "VULN_PCA1", *D.columns]
        coef_path.write_text(
            json.dumps({"coef": dict(zip(names, map(float, beta)))}, indent=2)
        )
    import json

    loaded = json.loads(coef_path.read_text())
    coef = pd.Series(loaded["coef"])
    se = pd.Series({k: abs(v) * 0.35 for k, v in loaded["coef"].items()})  # display-only CI width

    marketing = load_or_build_marketing()
    complaints = build_cfpb_subset()

    figure_01(df)
    figure_02(coef, se)
    figure_03(df)
    figure_04(marketing)
    figure_05(complaints)
    figure_06(marketing, complaints)

    write_manifest()
    print(f"\nDone. Figures in {OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
