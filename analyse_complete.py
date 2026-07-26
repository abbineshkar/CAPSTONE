"""
analyse_complete.py
--------------------
Complete analysis script for federated phishing detection results.
Nazario + Enron only (SpamAssassin and TREC 2007 removed).

Produces:
  1. Full results summary table
  2. Convergence curves (F1 per round)
  3. F1 heatmap by condition
  4. DistilBERT vs TinyBERT comparison
  5. F1 stability analysis (std deviation per experiment)
  6. Communication cost analysis (MB transferred to convergence)
  7. Centralised vs federated comparison
  8. Convergence speed bar chart
  9. LaTeX tables for paper

Run:
  python analyse_complete.py
"""

import os
import sys
import json
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

PALETTE = {
    "fedavg":     "#2563EB",
    "fedprox":    "#DC2626",
    "distilbert": "#7C3AED",
    "tinybert":   "#D97706",
}

NON_IID_LABELS = {
    "label_skew_iid":      "IID\n(α=100)",
    "label_skew_moderate": "Label Skew\nModerate\n(α=0.5)",
    "label_skew_severe":   "Label Skew\nSevere\n(α=0.1)",
    "feature_skew":        "Feature\nSkew",
    "quantity":            "Quantity\nHetero",
}

MODEL_PARAMS = {
    "distilbert": 66_955_010,
    "tinybert":   14_350_874,
}

# 1. Load results

def load_results():
    # Rebuild summary from individual JSON files
    records = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "E*_final.json")):
        with open(path) as f:
            records.append(json.load(f))

    summary = pd.DataFrame(records).sort_values("experiment_id").reset_index(drop=True)
    for col in ["accuracy", "f1", "precision", "recall", "auc_roc"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    # Save rebuilt summary
    summary.to_csv(os.path.join(RESULTS_DIR, "all_experiments_summary.csv"), index=False)
    print(f"Loaded {len(summary)} experiments")

    # Load round data
    round_dfs = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "E*_rounds.csv")):
        exp_id = os.path.basename(path).replace("_rounds.csv", "")
        df = pd.read_csv(path)
        if "f1" in df.columns:
            df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
        round_dfs[exp_id] = df

    print(f"Round data loaded for: {sorted(round_dfs.keys())}")
    return summary, round_dfs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Print full summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(summary):
    print("\n" + "="*100)
    print("FULL RESULTS TABLE — Nazario + Enron (20 Experiments)")
    print("="*100)
    cols = ["experiment_id", "model", "aggregation", "non_iid_type",
            "accuracy", "f1", "auc_roc", "convergence_round", "runtime_sec"]
    cols = [c for c in cols if c in summary.columns]
    df = summary[cols].copy()
    for col in ["accuracy", "f1", "auc_roc"]:
        if col in df.columns:
            df[col] = df[col].map("{:.4f}".format)
    print(df.to_string(index=False))
    print("="*100)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergence curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(summary, round_dfs):
    non_iid_list = [
        ("label_skew_iid",      "IID Baseline (α=100)"),
        ("label_skew_moderate", "Label Skew Moderate (α=0.5)"),
        ("label_skew_severe",   "Label Skew Severe (α=0.1) ★"),
        ("feature_skew",        "Feature Skew"),
        ("quantity",            "Quantity Heterogeneity"),
    ]
    models = [("distilbert", "DistilBERT (66M)"), ("tinybert", "TinyBERT (14M)")]

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)

    for row, (model_key, model_name) in enumerate(models):
        for col, (nid_type, nid_label) in enumerate(non_iid_list):
            ax = axes[row, col]
            plotted = False

            for agg in ["fedavg", "fedprox"]:
                match = summary[
                    (summary["model"] == model_key) &
                    (summary["aggregation"] == agg) &
                    (summary["non_iid_type"] == nid_type)
                ]
                if match.empty:
                    continue
                exp_id = match.iloc[0]["experiment_id"]
                if exp_id not in round_dfs:
                    continue

                rdf = round_dfs[exp_id]
                ax.plot(
                    rdf["round"], rdf["f1"],
                    color=PALETTE[agg],
                    label=agg.upper(),
                    linewidth=2.5,
                    marker="o", markersize=4,
                )
                plotted = True

            if row == 0:
                ax.set_title(nid_label, fontsize=9, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(f"{model_name}\nF1 Score", fontsize=9)
            ax.set_xlabel("Round", fontsize=8)
            ax.set_ylim(0.5, 1.02)
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

            # Highlight severe skew
            if nid_type == "label_skew_severe":
                ax.set_facecolor("#fff8f8")

            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower right")

    plt.suptitle(
        "F1 Score per Communication Round — FedAvg vs FedProx\n"
        "DistilBERT (top) | TinyBERT (bottom) | ★ = Key finding",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "convergence_curves")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    plt.close()
    print("Saved: convergence_curves.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. F1 heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(summary):
    df = summary.copy()
    df["config"] = df["model"].str.upper() + "+" + df["aggregation"].str.upper()

    pivot = df.pivot_table(
        index="config", columns="non_iid_type", values="f1", aggfunc="mean"
    )
    col_order = ["label_skew_iid", "label_skew_moderate", "label_skew_severe",
                 "feature_skew", "quantity"]
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.columns = [NON_IID_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="RdYlGn",
        vmin=0.88, vmax=1.0, linewidths=0.8, linecolor="white",
        ax=ax, cbar_kws={"label": "F1 Score (macro)", "shrink": 0.8}
    )
    ax.set_title(
        "F1 Score by Model Configuration and Non-IID Condition\n"
        "(Nazario + Enron datasets)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Non-IID Condition", fontsize=10)
    ax.set_ylabel("Model + Aggregation", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "heatmap_f1")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    plt.close()
    print("Saved: heatmap_f1.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. F1 Stability Analysis (std deviation across rounds)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_stability(summary, round_dfs):
    print("\n" + "="*80)
    print("F1 STABILITY ANALYSIS — Standard Deviation Across Rounds")
    print("="*80)
    print(f"{'Experiment':<8} {'Model':<12} {'Aggregation':<10} {'Non-IID Type':<25} "
          f"{'Mean F1':>8} {'Std F1':>8} {'Min F1':>8} {'Max F1':>8} {'Range':>8}")
    print("-"*95)

    stability_records = []
    for _, row in summary.iterrows():
        exp_id = row["experiment_id"]
        if exp_id not in round_dfs:
            continue
        rdf = round_dfs[exp_id]
        f1s = rdf["f1"].dropna().values

        if len(f1s) == 0:
            continue

        rec = {
            "experiment_id": exp_id,
            "model":         row["model"],
            "aggregation":   row["aggregation"],
            "non_iid_type":  row["non_iid_type"],
            "mean_f1":       float(np.mean(f1s)),
            "std_f1":        float(np.std(f1s)),
            "min_f1":        float(np.min(f1s)),
            "max_f1":        float(np.max(f1s)),
            "range_f1":      float(np.max(f1s) - np.min(f1s)),
            "rounds":        len(f1s),
        }
        stability_records.append(rec)

        print(
            f"{exp_id:<8} {row['model']:<12} {row['aggregation']:<10} "
            f"{row['non_iid_type']:<25} "
            f"{rec['mean_f1']:>8.4f} {rec['std_f1']:>8.4f} "
            f"{rec['min_f1']:>8.4f} {rec['max_f1']:>8.4f} {rec['range_f1']:>8.4f}"
        )

    stab_df = pd.DataFrame(stability_records)

    # Save
    stab_path = os.path.join(RESULTS_DIR, "stability_analysis.csv")
    stab_df.to_csv(stab_path, index=False)
    print(f"\nStability data saved to {stab_path}")

    # Key insight
    print("\n── Key stability comparison (Severe Label Skew) ──")
    severe = stab_df[stab_df["non_iid_type"] == "label_skew_severe"]
    for _, r in severe.iterrows():
        stability_label = "UNSTABLE" if r["std_f1"] > 0.02 else "Stable"
        print(
            f"  {r['experiment_id']} {r['model'].upper()} {r['aggregation'].upper()}: "
            f"std={r['std_f1']:.4f} range={r['range_f1']:.4f} → {stability_label}"
        )

    # Plot stability heatmap
    if not stab_df.empty:
        stab_df["config"] = stab_df["model"].str.upper() + "+" + stab_df["aggregation"].str.upper()
        pivot_std = stab_df.pivot_table(
            index="config", columns="non_iid_type", values="std_f1", aggfunc="mean"
        )
        col_order = ["label_skew_iid", "label_skew_moderate", "label_skew_severe",
                     "feature_skew", "quantity"]
        col_order = [c for c in col_order if c in pivot_std.columns]
        pivot_std = pivot_std[col_order]
        pivot_std.columns = [NON_IID_LABELS.get(c, c) for c in pivot_std.columns]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Std dev heatmap
        sns.heatmap(
            pivot_std, annot=True, fmt=".4f", cmap="YlOrRd",
            vmin=0, vmax=0.05, linewidths=0.8, linecolor="white",
            ax=axes[0], cbar_kws={"label": "Std Dev of F1", "shrink": 0.8}
        )
        axes[0].set_title("F1 Instability (Std Dev across Rounds)\nHigher = More Unstable",
                          fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")

        # Range heatmap
        pivot_range = stab_df.pivot_table(
            index="config", columns="non_iid_type", values="range_f1", aggfunc="mean"
        )
        col_order2 = [c for c in ["label_skew_iid", "label_skew_moderate",
                                    "label_skew_severe", "feature_skew", "quantity"]
                      if c in pivot_range.columns]
        pivot_range = pivot_range[col_order2]
        pivot_range.columns = [NON_IID_LABELS.get(c, c) for c in pivot_range.columns]

        sns.heatmap(
            pivot_range, annot=True, fmt=".4f", cmap="YlOrRd",
            vmin=0, vmax=0.20, linewidths=0.8, linecolor="white",
            ax=axes[1], cbar_kws={"label": "F1 Range (max−min)", "shrink": 0.8}
        )
        axes[1].set_title("F1 Range across Rounds\nHigher = More Oscillation",
                          fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")

        plt.suptitle("Convergence Stability Analysis — FedAvg vs FedProx",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "stability_analysis")
        fig.savefig(path + ".pdf", bbox_inches="tight")
        fig.savefig(path + ".png", bbox_inches="tight")
        plt.close()
        print("Saved: stability_analysis.pdf / .png")

    return stab_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Communication cost analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_communication_cost(summary, round_dfs):
    print("\n" + "="*90)
    print("COMMUNICATION COST ANALYSIS")
    print("Total data transferred = rounds × model_params × 4 bytes × clients × 2 (upload+download)")
    print("="*90)
    print(f"{'Experiment':<8} {'Model':<12} {'Aggregation':<10} {'Non-IID Type':<25} "
          f"{'Rounds':>7} {'Total MB':>10} {'MB/Client':>10}")
    print("-"*85)

    cost_records = []
    for _, row in summary.iterrows():
        exp_id  = row["experiment_id"]
        model   = row["model"]
        agg     = row["aggregation"]
        nid     = row["non_iid_type"]
        rounds  = pd.to_numeric(row.get("convergence_round", row.get("total_rounds", 10)),
                                errors="coerce")
        if pd.isna(rounds):
            rounds = 10

        params      = MODEL_PARAMS.get(model, 66_955_010)
        clients     = 5
        bytes_total = params * 4 * clients * int(rounds) * 2
        mb_total    = bytes_total / (1024 ** 2)
        mb_client   = mb_total / clients

        rec = {
            "experiment_id": exp_id,
            "model":         model,
            "aggregation":   agg,
            "non_iid_type":  nid,
            "convergence_rounds": int(rounds),
            "total_mb":      round(mb_total, 1),
            "mb_per_client": round(mb_client, 1),
        }
        cost_records.append(rec)

        print(
            f"{exp_id:<8} {model:<12} {agg:<10} {nid:<25} "
            f"{int(rounds):>7} {mb_total:>10.1f} {mb_client:>10.1f}"
        )

    cost_df = pd.DataFrame(cost_records)
    cost_path = os.path.join(RESULTS_DIR, "communication_cost.csv")
    cost_df.to_csv(cost_path, index=False)
    print(f"\nCommunication cost data saved to {cost_path}")

    # Key insight
    print("\n── Communication cost: FedAvg vs FedProx under severe skew ──")
    severe = cost_df[cost_df["non_iid_type"] == "label_skew_severe"]
    for _, r in severe.iterrows():
        print(
            f"  {r['experiment_id']} {r['model'].upper()} {r['aggregation'].upper()}: "
            f"{r['convergence_rounds']} rounds = {r['total_mb']:.1f} MB total"
        )

    print("\n── TinyBERT vs DistilBERT communication savings ──")
    for nid in ["label_skew_iid", "label_skew_severe"]:
        sub = cost_df[cost_df["non_iid_type"] == nid]
        for agg in ["fedavg", "fedprox"]:
            dist = sub[(sub["model"]=="distilbert") & (sub["aggregation"]==agg)]["total_mb"].mean()
            tiny = sub[(sub["model"]=="tinybert")   & (sub["aggregation"]==agg)]["total_mb"].mean()
            if dist > 0 and tiny > 0:
                saving = (1 - tiny/dist) * 100
                print(f"  {nid} {agg.upper()}: DistilBERT={dist:.1f}MB TinyBERT={tiny:.1f}MB "
                      f"(TinyBERT saves {saving:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart — total MB by condition and model
    nid_order  = ["label_skew_iid", "label_skew_moderate", "label_skew_severe",
                  "feature_skew", "quantity"]
    nid_labels = ["IID", "Moderate\nSkew", "Severe\nSkew ★", "Feature\nSkew", "Quantity"]

    ax = axes[0]
    x  = np.arange(len(nid_order))
    w  = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    cfgs = [
        ("distilbert", "fedavg",  "DistilBERT\nFedAvg",  PALETTE["fedavg"]),
        ("distilbert", "fedprox", "DistilBERT\nFedProx", PALETTE["fedprox"]),
        ("tinybert",   "fedavg",  "TinyBERT\nFedAvg",    PALETTE["distilbert"]),
        ("tinybert",   "fedprox", "TinyBERT\nFedProx",   PALETTE["tinybert"]),
    ]
    for i, (model_key, agg, label, color) in enumerate(cfgs):
        vals = []
        for nid in nid_order:
            sub = cost_df[(cost_df["model"]==model_key) & (cost_df["aggregation"]==agg) &
                          (cost_df["non_iid_type"]==nid)]
            vals.append(sub["total_mb"].mean() if not sub.empty else 0)
        ax.bar(x + offsets[i]*w, vals, width=w, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(nid_labels, fontsize=8)
    ax.set_ylabel("Total Data Transferred (MB)")
    ax.set_title("Communication Cost to Convergence\nby Non-IID Condition", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # Pie chart — TinyBERT vs DistilBERT total cost
    ax = axes[1]
    dist_total = cost_df[cost_df["model"]=="distilbert"]["total_mb"].sum()
    tiny_total = cost_df[cost_df["model"]=="tinybert"]["total_mb"].sum()
    ax.pie(
        [dist_total, tiny_total],
        labels=[f"DistilBERT\n{dist_total:.0f} MB", f"TinyBERT\n{tiny_total:.0f} MB"],
        colors=[PALETTE["distilbert"], PALETTE["tinybert"]],
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title("Total Communication Cost\nDistilBERT vs TinyBERT (all 20 experiments)",
                 fontweight="bold")

    plt.suptitle("Communication Efficiency Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "communication_cost")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    plt.close()
    print("Saved: communication_cost.pdf / .png")

    return cost_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Centralised vs Federated comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_centralised_comparison(summary):
    cent_path = os.path.join(RESULTS_DIR, "centralised_baseline.json")
    if not os.path.exists(cent_path):
        print("Centralised baseline not found — skipping comparison plot")
        return

    with open(cent_path) as f:
        cent_data = json.load(f)
    cent_df = pd.DataFrame(cent_data)

    print("\n" + "="*80)
    print("FEDERATED vs CENTRALISED COMPARISON")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (model_key, model_name) in enumerate(
        [("distilbert", "DistilBERT"), ("tinybert", "TinyBERT")]
    ):
        ax = axes[ax_idx]

        cent_f1 = float(cent_df[cent_df["model"]==model_key]["f1"].values[0])

        sub = summary[summary["model"]==model_key].copy()
        conditions = [
            ("label_skew_iid",      "IID\nBaseline"),
            ("label_skew_moderate", "Label Skew\nModerate"),
            ("label_skew_severe",   "Label Skew\nSevere ★"),
            ("feature_skew",        "Feature\nSkew"),
            ("quantity",            "Quantity\nHetero"),
        ]

        x = np.arange(len(conditions))
        w = 0.3

        fedavg_f1  = []
        fedprox_f1 = []

        for nid, _ in conditions:
            fa = sub[(sub["non_iid_type"]==nid) & (sub["aggregation"]=="fedavg")]["f1"].mean()
            fp = sub[(sub["non_iid_type"]==nid) & (sub["aggregation"]=="fedprox")]["f1"].mean()
            fedavg_f1.append(fa if not np.isnan(fa) else 0)
            fedprox_f1.append(fp if not np.isnan(fp) else 0)

        ax.bar(x - w/2, fedavg_f1,  width=w, label="FedAvg",  color=PALETTE["fedavg"],  alpha=0.85)
        ax.bar(x + w/2, fedprox_f1, width=w, label="FedProx", color=PALETTE["fedprox"], alpha=0.85)
        ax.axhline(y=cent_f1, color="black", linestyle="--", linewidth=2,
                   label=f"Centralised ({cent_f1:.4f})")

        ax.set_xticks(x)
        ax.set_xticklabels([c[1] for c in conditions], fontsize=8)
        ax.set_ylim(0.88, 1.005)
        ax.set_ylabel("F1 Score (macro)")
        ax.set_title(f"{model_name}\nFederated vs Centralised Baseline", fontweight="bold")
        ax.legend(fontsize=8)

        # Print numbers
        print(f"\n{model_name} (Centralised F1 = {cent_f1:.4f})")
        for i, (nid, label) in enumerate(conditions):
            fa_gap = cent_f1 - fedavg_f1[i]
            fp_gap = cent_f1 - fedprox_f1[i]
            print(f"  {label.replace(chr(10),' '):<25} "
                  f"FedAvg={fedavg_f1[i]:.4f} (gap={fa_gap:+.4f})  "
                  f"FedProx={fedprox_f1[i]:.4f} (gap={fp_gap:+.4f})")

    plt.suptitle("Federated Learning vs Centralised Baseline\n"
                 "Dashed line = centralised performance ceiling",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "centralised_comparison")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight")
    plt.close()
    print(f"\nSaved: centralised_comparison.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def run_stats(summary):
    print("\n" + "="*70)
    print("STATISTICAL TESTS — FedAvg vs FedProx (Non-IID conditions only)")
    print("="*70)

    non_iid_mask = summary["non_iid_type"] != "label_skew_iid"

    for model_key in ["distilbert", "tinybert"]:
        fa = summary[(summary["model"]==model_key) & (summary["aggregation"]=="fedavg") &
                     non_iid_mask]["f1"].dropna().values
        fp = summary[(summary["model"]==model_key) & (summary["aggregation"]=="fedprox") &
                     non_iid_mask]["f1"].dropna().values

        n = min(len(fa), len(fp))
        if n < 2:
            print(f"{model_key.upper()}: insufficient data (n={n})")
            continue

        fa, fp = fa[:n], fp[:n]
        delta  = np.mean(fp) - np.mean(fa)

        try:
            _, p_w = wilcoxon(fa, fp)
            _, p_t = ttest_rel(fa, fp)
        except Exception:
            p_w = p_t = float("nan")

        direction = "FedProx better" if delta > 0 else "FedAvg better"
        print(f"\n{model_key.upper()} (n={n} Non-IID experiments)")
        print(f"  FedAvg  mean F1: {np.mean(fa):.4f} ± {np.std(fa):.4f}")
        print(f"  FedProx mean F1: {np.mean(fp):.4f} ± {np.std(fp):.4f}")
        print(f"  Δ F1 (FedProx − FedAvg): {delta:+.4f}  → {direction}")
        print(f"  Wilcoxon signed-rank: p={p_w:.4f} {'★ significant' if p_w<0.05 else 'ns'}")
        print(f"  Paired t-test:        p={p_t:.4f} {'★ significant' if p_t<0.05 else 'ns'}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Export LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────

def export_latex(summary, stab_df, cost_df):
    # Main results table
    df = summary[["experiment_id", "model", "aggregation", "non_iid_type",
                  "accuracy", "f1", "auc_roc", "convergence_round"]].copy()
    for col in ["accuracy", "f1", "auc_roc"]:
        df[col] = df[col].map("{:.4f}".format)
    df.columns = ["Exp.", "Model", "Strategy", "Non-IID Type",
                  "Accuracy", "F1", "AUC-ROC", "Conv. Round"]
    latex1 = df.to_latex(
        index=False, escape=True,
        caption="Full results for all 20 federated phishing detection experiments (Nazario + Enron).",
        label="tab:full_results",
        column_format="llllcccr",
    )
    with open(os.path.join(FIGURES_DIR, "results_table.tex"), "w") as f:
        f.write(latex1)

    # Stability table
    if stab_df is not None and not stab_df.empty:
        sdf = stab_df[["experiment_id", "model", "aggregation", "non_iid_type",
                        "mean_f1", "std_f1", "range_f1", "rounds"]].copy()
        for col in ["mean_f1", "std_f1", "range_f1"]:
            sdf[col] = sdf[col].map("{:.4f}".format)
        sdf.columns = ["Exp.", "Model", "Strategy", "Non-IID Type",
                       "Mean F1", "Std F1", "Range F1", "Rounds"]
        latex2 = sdf.to_latex(
            index=False, escape=True,
            caption="Convergence stability analysis: F1 standard deviation and range across rounds.",
            label="tab:stability",
            column_format="llllcccr",
        )
        with open(os.path.join(FIGURES_DIR, "stability_table.tex"), "w") as f:
            f.write(latex2)

    # Communication cost table
    if cost_df is not None and not cost_df.empty:
        cdf = cost_df[["experiment_id", "model", "aggregation", "non_iid_type",
                        "convergence_rounds", "total_mb", "mb_per_client"]].copy()
        cdf["total_mb"]    = cdf["total_mb"].map("{:.1f}".format)
        cdf["mb_per_client"] = cdf["mb_per_client"].map("{:.1f}".format)
        cdf.columns = ["Exp.", "Model", "Strategy", "Non-IID Type",
                       "Conv. Rounds", "Total MB", "MB/Client"]
        latex3 = cdf.to_latex(
            index=False, escape=True,
            caption="Communication cost analysis: total data transferred to convergence.",
            label="tab:communication",
            column_format="llllcrr",
        )
        with open(os.path.join(FIGURES_DIR, "communication_table.tex"), "w") as f:
            f.write(latex3)

    print("\nLaTeX tables saved:")
    print("  results_table.tex")
    print("  stability_table.tex")
    print("  communication_table.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...")
    summary, round_dfs = load_results()

    print_summary(summary)

    print("\nGenerating figures...")
    plot_convergence(summary, round_dfs)
    plot_heatmap(summary)

    stab_df = analyse_stability(summary, round_dfs)
    cost_df = analyse_communication_cost(summary, round_dfs)

    plot_centralised_comparison(summary)
    run_stats(summary)
    export_latex(summary, stab_df, cost_df)

    print(f"\nAll outputs saved to: {FIGURES_DIR}")
    print("\nFiles generated:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"  {f}  ({size//1024} KB)")
