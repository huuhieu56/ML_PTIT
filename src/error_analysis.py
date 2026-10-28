import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SENTIMENT_LABELS = ["Negative (1-2★)", "Neutral (3★)", "Positive (4-5★)"]
RATING_LABELS = ["1★", "2★", "3★", "4★", "5★"]

def plot_confusion_matrix(cm, labels, title: str, save_path: str):
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
    annot = np.array([[f"{c}\n({p:.1%})" for c, p in zip(row_c, row_p)] for row_c, row_p in zip(cm_arr, cm_norm)])

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()

def extract_error_cases(texts, y_true, y_pred, labels, top_n: int = 5):
    df = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred})
    errs = df[df["true"] != df["pred"]].copy()
    errs["error_type"] = [f"{labels[t]} -> {labels[p]}" for t, p in zip(errs["true"], errs["pred"])]

    top_pairs = errs["error_type"].value_counts().head(top_n).to_dict()
    cases = []
    for pair in top_pairs:
        for text in errs[errs["error_type"] == pair]["text"].head(3):
            cases.append({"type": pair, "text": text})

    return {
        "total": len(df),
        "errors": len(errs),
        "error_rate": round(len(errs) / len(df), 4),
        "top_pairs": top_pairs,
        "cases": cases
    }
