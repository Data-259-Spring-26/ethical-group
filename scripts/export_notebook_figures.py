#!/usr/bin/env python3
"""
Reproduce figures from BNPL_Analysis_1_to_4_combined notebook for the final paper.

Outputs to figures/paper/ with Quarto-friendly names.
Run from repo root: python scripts/export_notebook_figures.py
"""

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SHED_ZIP = REPO / "data/raw/public2024.csv.zip"
OUT = REPO / "figures/paper"
DELIVERABLE = REPO / "Final Deliverable/figures"

# ── Recoding (notebook Analysis 1–2 vulnerability items) ─────────────────────
B0_A_SCALE = {"Completely": 0, "Very well": 1, "Somewhat": 2, "Very little": 3, "Not at all": 4}
B0_B_SCALE = {"Completely": 4, "Very well": 3, "Somewhat": 2, "Very little": 1, "Not at all": 0}
B2_SCALE = {
    "Living comfortably": 0,
    "Doing okay": 1,
    "Just getting by": 2,
    "Finding it difficult to get by": 3,
}
EF1_SCALE = {"Yes": 0, "No": 1}
EF3F_SCALE = {"No": 0, "Yes": 1}
EF7_SCALE = {
    "$2,000 or more": 0,
    "$1,000 to $1,999": 1,
    "$500 to $999": 2,
    "$100 to $499": 3,
    "Under $100": 4,
}

ITEMS = ["B0_a_r", "B0_b_r", "B0_c_r", "B2_r", "EF1_r", "EF3_f_r", "EF7_r"]
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
INC_ORDER = [
    "Less than $10,000",
    "$10,000 to $24,999",
    "$25,000 to $49,999",
    "$50,000 to $74,999",
    "$75,000 to $99,999",
    "$100,000 to $149,999",
    "$150,000 or more",
]
EDUC_ORDER = [
    "Less than a high school degree",
    "High school degree",
    "Some college",
    "Bachelor's degree",
]
REFUSE = {"Refused", "Don't know", "Don't Know", -1, "-1"}


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def weighted_std(x, w):
    m = weighted_mean(x, w)
    return np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w))


def weighted_corr_matrix(X, w):
    n, k = X.shape
    Z = np.column_stack(
        [
            (X[:, j] - weighted_mean(X[:, j], w)) / max(weighted_std(X[:, j], w), 1e-8)
            for j in range(k)
        ]
    )
    W = w / w.sum()
    return Z, (Z * W[:, None]).T @ Z


def load_shed_data():
    cache = REPO / "data/processed/shed_analysis.csv"
    if cache.exists():
        d = pd.read_csv(cache)
        extra_cols = ["ppmarit5", "ppmsacat"]
        if not all(c in d.columns for c in extra_cols):
            z = zipfile.ZipFile(SHED_ZIP)
            extra = pd.read_csv(z.open("public2024.csv"), usecols=extra_cols, low_memory=False)
            d = pd.concat([d, extra], axis=1)
        return d

    z = zipfile.ZipFile(SHED_ZIP)
    df = pd.read_csv(z.open("public2024.csv"), low_memory=False)
    d = df.copy()
    d["BNPL1_bin"] = d["BNPL1"].map({"Yes": 1, "No": 0})
    for col in d.columns:
        if d[col].dtype == object:
            d.loc[d[col].isin(REFUSE), col] = np.nan
    d["B0_a_r"] = d["B0_a"].map(B0_A_SCALE)
    d["B0_b_r"] = d["B0_b"].map(B0_B_SCALE)
    d["B0_c_r"] = d["B0_c"].map(B0_B_SCALE)
    d["B2_r"] = d["B2"].map(B2_SCALE)
    d["EF1_r"] = d["EF1"].map(EF1_SCALE)
    d["EF3_f_r"] = d["EF3_f"].map(EF3F_SCALE)
    d["EF7_r"] = d["EF7"].map(EF7_SCALE)
    sub = d.dropna(subset=ITEMS + ["weight", "BNPL1_bin"]).copy()
    X = sub[ITEMS].values.astype(float)
    w = sub["weight"].values
    Z, C = weighted_corr_matrix(X, w)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    loadings = eigvecs[:, 0]
    frag_idx = [ITEMS.index(c) for c in ["B0_b_r", "B0_c_r", "B2_r", "EF1_r", "EF7_r"]]
    if np.mean(loadings[frag_idx]) < 0:
        loadings = -loadings
    pc1_raw = Z @ loadings
    sub["VULN_PCA1"] = (pc1_raw - weighted_mean(pc1_raw, w)) / weighted_std(pc1_raw, w)
    d = d.join(sub[["VULN_PCA1"]])
    return d


def _treatment_dummies(series: pd.Series, ref: str, prefix: str) -> pd.DataFrame:
    dummies = pd.get_dummies(series, prefix=prefix, dtype=float)
    ref_col = f"{prefix}_{ref}"
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])
    return dummies


def fit_notebook_wls(d: pd.DataFrame):
    """Weighted least squares matching the combined notebook specification."""
    age_order = AGE_ORDER
    inc_order = INC_ORDER
    educ_order = EDUC_ORDER

    reg = d.copy()
    reg["age_cat"] = pd.Categorical(reg["ppagecat"], categories=age_order, ordered=True)
    reg["income_cat"] = pd.Categorical(reg["ppinc7"], categories=inc_order, ordered=True)
    reg["educ_cat"] = pd.Categorical(reg["educ_4cat"], categories=educ_order, ordered=True)
    reg = reg.dropna(
        subset=[
            "BNPL1_bin",
            "VULN_PCA1",
            "weight",
            "age_cat",
            "income_cat",
            "educ_cat",
            "ppgender",
            "ppmarit5",
            "ppmsacat",
        ]
    )

    parts = [
        _treatment_dummies(reg["age_cat"].astype(str), "18-24", "age"),
        _treatment_dummies(reg["income_cat"].astype(str), "Less than $10,000", "inc"),
        _treatment_dummies(reg["educ_cat"].astype(str), "Less than a high school degree", "educ"),
        _treatment_dummies(reg["ppgender"], "Male", "gender"),
        _treatment_dummies(reg["ppmarit5"], "Now married", "marit"),
        _treatment_dummies(reg["ppmsacat"], "Metro", "msa"),
    ]
    D = pd.concat(parts, axis=1)
    v = reg["VULN_PCA1"].astype(float).values
    y = reg["BNPL1_bin"].astype(float).values
    sw = reg["weight"].astype(float).values
    X = np.column_stack([np.ones(len(reg)), v, D.values])
    beta = np.linalg.solve(X.T @ (X * sw[:, None]), X.T @ (y * sw))

    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ (X * sw[:, None]))
    sigma2 = np.sum(sw * resid**2) / max(np.sum(sw) - X.shape[1], 1)
    se_vuln = np.sqrt(sigma2 * xtx_inv[1, 1])
    t_stat = beta[1] / se_vuln
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(reg) - X.shape[1])))

    return beta[0], beta[1], p_val, reg


def figure_predicted_bnpl_wls(d, out_path):
    """Notebook fig_a3: predicted BNPL rate vs VULN_PCA1 (WLS, controls at reference)."""
    intercept, vuln_coef, vuln_pval, reg = fit_notebook_wls(d)
    baseline = np.average(d["BNPL1_bin"], weights=d["weight"])

    vuln_grid = np.linspace(reg["VULN_PCA1"].min(), reg["VULN_PCA1"].max(), 100)
    predicted = (intercept + vuln_coef * vuln_grid) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vuln_grid, predicted, color="navy", linewidth=2.5, label="Predicted BNPL use (%)")
    ax.axhline(baseline * 100, color="gray", linestyle="--", linewidth=1.2, label=f"Baseline rate ({baseline * 100:.1f}%)")
    ax.axvline(0, color="silver", linestyle=":", linewidth=1)
    ax.set_xlabel("VULN_PCA1 (standardized vulnerability score)", fontsize=12)
    ax.set_ylabel("Predicted BNPL use rate (%)", fontsize=12)
    ax.set_title(
        "Predicted BNPL use across the vulnerability score\n"
        "(all other predictors held at reference values)",
        fontsize=12,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.text(
        0.02,
        0.94,
        f"Slope = {vuln_coef * 100:+.2f} pp per SD   p = {vuln_pval:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        color="navy",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path.relative_to(REPO)}")


def figure_vulnerability_density(d, out_path):
    """Notebook Analysis 2: weighted density of VULN_PCA1 by BNPL use."""
    sub = d.dropna(subset=["VULN_PCA1", "BNPL1_bin", "weight"]).copy()
    users = sub[sub["BNPL1_bin"] == 1]
    nonusers = sub[sub["BNPL1_bin"] == 0]
    mu_users = np.average(users["VULN_PCA1"], weights=users["weight"])
    mu_nonusers = np.average(nonusers["VULN_PCA1"], weights=nonusers["weight"])

    bins = np.linspace(sub["VULN_PCA1"].min(), sub["VULN_PCA1"].max(), 40)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        nonusers["VULN_PCA1"],
        bins=bins,
        weights=nonusers["weight"],
        density=True,
        alpha=0.55,
        label=f"Non-users (n={len(nonusers):,})",
    )
    ax.hist(
        users["VULN_PCA1"],
        bins=bins,
        weights=users["weight"],
        density=True,
        alpha=0.55,
        label=f"BNPL users (n={len(users):,})",
    )
    ax.axvline(mu_nonusers, linestyle="--", linewidth=1)
    ax.axvline(mu_users, linestyle="--", linewidth=1)
    ax.set_xlabel("VULN_PCA1 (standardized, higher = more fragile)")
    ax.set_ylabel("Weighted density")
    ax.set_title("Distribution of composite vulnerability by BNPL use")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path.relative_to(REPO)}")


def figure_stratum_gaps(d, out_path):
    """Notebook fig_a4: BNPL gap (pp) by age x income stratum."""
    d = d.copy()
    d["stratum"] = d["ppagecat"].astype(str) + " | " + d["ppinc7"].astype(str)
    strata = sorted(d["stratum"].unique())
    median_v = d["VULN_PCA1"].median()
    d["vuln_group"] = np.where(d["VULN_PCA1"] >= median_v, "High vulnerability", "Low vulnerability")

    MIN_N = 30
    rows = []
    for stratum in strata:
        sub = d[d["stratum"] == stratum]
        high = sub[sub["vuln_group"] == "High vulnerability"]
        low = sub[sub["vuln_group"] == "Low vulnerability"]
        if len(high) < MIN_N or len(low) < MIN_N:
            continue
        rate_h = np.average(high["BNPL1_bin"], weights=high["weight"]) * 100
        rate_l = np.average(low["BNPL1_bin"], weights=low["weight"]) * 100
        t_stat, p_val = stats.ttest_ind(
            high["BNPL1_bin"], low["BNPL1_bin"], equal_var=False
        )
        rows.append(
            {
                "Stratum (age | income)": stratum,
                "Gap (pp)": rate_h - rate_l,
                "p (raw)": p_val,
                "n_high": len(high),
                "n_low": len(low),
            }
        )

    results_a4 = pd.DataFrame(rows)
    p_raw = results_a4["p (raw)"].values
    m = len(p_raw)
    order = np.argsort(p_raw)
    p_adj = np.empty(m)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, p_raw[idx] * (m - rank))
    for rank in range(1, m):
        idx = order[rank]
        prev = order[rank - 1]
        p_adj[idx] = max(p_adj[idx], p_adj[prev])
    results_a4["p (Holm)"] = p_adj
    results_a4["Sig."] = ["*" if p < 0.05 else "" for p in p_adj]
    results_a4 = results_a4.sort_values("Gap (pp)", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(results_a4) * 0.28)))
    colors = ["#d62728" if s == "*" else "#1f77b4" for s in results_a4["Sig."]]
    y_pos = range(len(results_a4))
    ax.scatter(results_a4["Gap (pp)"], y_pos, c=colors, s=80, zorder=3)
    for i, gap in enumerate(results_a4["Gap (pp)"]):
        ax.text(gap + 0.3, i, f"{gap:+.1f}", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_a4["Stratum (age | income)"], fontsize=8)
    ax.set_xlabel("BNPL use gap: high vs. low vulnerability (percentage points)", fontsize=11)
    ax.set_title(
        "BNPL use gap within age × income strata\n"
        "(high = at or above median VULN_PCA1; Holm-Bonferroni * p < 0.05)",
        fontsize=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path.relative_to(REPO)}")


def copy_to_deliverable(path: Path):
    DELIVERABLE.mkdir(parents=True, exist_ok=True)
    (DELIVERABLE / path.name).write_bytes(path.read_bytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading SHED and building VULN_PCA1 (notebook pipeline)...")
    d = load_shed_data()
    print(f"  n = {len(d):,} complete cases")

    figure_vulnerability_density(d, OUT / "figure-01-vulnerability-density-by-bnpl.png")
    figure_predicted_bnpl_wls(d, OUT / "figure-02-predicted-bnpl-wls.png")
    figure_stratum_gaps(d, OUT / "figure-03-stratum-gaps-age-income.png")

    for name in (
        "figure-01-vulnerability-density-by-bnpl.png",
        "figure-02-predicted-bnpl-wls.png",
        "figure-03-stratum-gaps-age-income.png",
    ):
        copy_to_deliverable(OUT / name)
    print("Done. Notebook figures are in figures/paper/ and Final Deliverable/figures/.")


if __name__ == "__main__":
    main()
