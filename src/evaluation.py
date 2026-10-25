"""Test seti değerlendirmesi, karşılaştırma tabloları ve grafikleri."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import plot_tree

from . import config as C
from .visualization import BLUE, GRAY, ORANGE, SEQUENTIAL, TEXT, save, tr

MODEL_COLORS = {"Random Forest": BLUE, "Decision Tree": ORANGE, "Baseline (çoğunluk)": GRAY}


def test_metrics(name: str, pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def plot_confusion(models: dict, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.9))
    for ax, (name, pipe) in zip(np.atleast_1d(axes), models.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        ax.imshow(cm, cmap=SEQUENTIAL)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            share = v / cm.sum() * 100
            ax.text(
                j,
                i,
                f"{v:,}\n%{share:.1f}",
                ha="center",
                va="center",
                color="white" if v > cm.max() / 2 else TEXT,
                fontsize=10,
            )
        ax.set_xticks([0, 1], ["Nötr/Değil", "Memnun"])
        ax.set_yticks([0, 1], ["Nötr/Değil", "Memnun"])
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        ax.set_title(name)
    fig.suptitle("Karmaşıklık matrisleri (test seti)", fontweight="bold", y=1.03)
    return save(fig, "08_karmasiklik_matrisi")


def plot_roc(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for name, pipe in models.items():
        p = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(
            fpr,
            tpr,
            lw=2,
            color=MODEL_COLORS.get(name, GRAY),
            label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRAY)
    ax.set_xlabel("Yanlış pozitif oranı")
    ax.set_ylabel("Doğru pozitif oranı")
    ax.set_title("ROC eğrileri (test seti)")
    ax.legend(loc="lower right", fontsize=8)
    return save(fig, "09_roc")


def plot_metric_comparison(df: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    d = df.set_index("model")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / len(d)
    for k, (model, row) in enumerate(d.iterrows()):
        x = np.arange(len(metrics)) + k * w
        bars = ax.bar(
            x, row[metrics].values, w * 0.92, label=model, color=MODEL_COLORS.get(model, GRAY)
        )
        if model != "Baseline (çoğunluk)":
            for b, v in zip(bars, row[metrics].values):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    fontsize=7,
                    rotation=90,
                )
    ax.set_xticks(np.arange(len(metrics)) + w * (len(d) - 1) / 2, names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Test seti metrik karşılaştırması")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    return save(fig, "10_metrik_karsilastirma")


def plot_depth_curve(curve: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(curve["max_depth"], curve["train"], lw=2, color=ORANGE, label="Eğitim")
    ax.plot(curve["max_depth"], curve["validation"], lw=2, color=BLUE, label="Doğrulama (CV)")
    best = curve.loc[curve["validation"].idxmax()]
    ax.scatter([best["max_depth"]], [best["validation"]], s=60, color=BLUE, zorder=3)
    ax.annotate(
        f"en iyi derinlik = {int(best['max_depth'])}\ndoğrulama = {best['validation']:.3f}",
        (best["max_depth"], best["validation"]),
        xytext=(10, -40),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color=GRAY),
    )
    ax.set_xlabel("Karar ağacı max_depth")
    ax.set_ylabel("Doğruluk")
    ax.set_title("Model karmaşıklığı: eksik öğrenme → aşırı öğrenme")
    ax.legend()
    return save(fig, "07_derinlik_egrisi")


def plot_tree_top(pipe, max_depth=3):
    model, names = pipe.named_steps["model"], pipe.named_steps["prep"].feature_names_
    fig, ax = plt.subplots(figsize=(16, 6.5))
    plot_tree(
        model,
        max_depth=max_depth,
        feature_names=[tr(n) for n in names],
        class_names=["Nötr/Değil", "Memnun"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Karar ağacının ilk {max_depth} seviyesi (karar kuralları)")
    return save(fig, "11_karar_agaci")


def literature_table(our_rows: pd.DataFrame) -> pd.DataFrame:
    lit = pd.DataFrame(C.LITERATURE)
    ours = pd.DataFrame(
        [
            {
                "study": f"Bu çalışma - {r['model']}",
                "methods": "DT, RF + SHAP",
                "best": r["model"],
                "accuracy": r["accuracy"],
            }
            for _, r in our_rows.iterrows()
            if "Baseline" not in r["model"]
        ]
    )
    out = pd.concat([lit, ours], ignore_index=True)
    out["accuracy"] = out["accuracy"].round(4)
    return out
