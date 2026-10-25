"""Keşifsel veri analizi (EDA) grafikleri ve ortak grafik stili."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

# --------------------------------------------------------------- Stil
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SAT_COLORS = {1: BLUE, 0: ORANGE}  # 1 = memnun, 0 = nötr/memnun değil
SAT_NAMES = {1: "Memnun", 0: "Nötr / Memnun değil"}
DIVERGING = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ee", BLUE])
VALUE_TR = {
    "Business travel": "İş",
    "Personal Travel": "Kişisel",
    "Loyal Customer": "Sadık",
    "disloyal Customer": "Sadık değil",
    "Eco": "Economy",
    "Eco Plus": "Economy Plus",
}
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", BLUE, "#123d73"])


def set_style():
    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "text.color": TEXT,
        }
    )


def save(fig, name: str):
    fig.savefig(C.FIG_DIR / f"{name}.png")
    return fig


def tr(col: str) -> str:
    return C.TR_LABELS.get(col, col)


# --------------------------------------------------------------- EDA
def plot_target(y: pd.Series):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(
        [SAT_NAMES[i] for i in counts.index],
        counts.values,
        color=[SAT_COLORS[i] for i in counts.index],
        width=0.55,
    )
    for b, v in zip(bars, counts.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:,}\n(%{v / counts.sum() * 100:.1f})",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Hedef değişken dağılımı")
    ax.set_ylabel("Yolcu sayısı")
    return save(fig, "01_hedef_dagilimi")


def plot_segments(df: pd.DataFrame, y: pd.Series):
    """Kabin sınıfı, seyahat tipi ve müşteri tipine göre memnuniyet oranı."""
    cols = [
        ("travel_class", "Kabin sınıfı"),
        ("type_of_travel", "Seyahat tipi"),
        ("customer_type", "Müşteri tipi"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
    for ax, (col, title) in zip(axes, cols):
        rate = y.groupby(df[col].replace(VALUE_TR)).mean().sort_values()
        pos = np.arange(len(rate))
        ax.barh(pos, rate.values, color=BLUE, height=0.55)
        ax.set_yticks(pos, rate.index.astype(str))
        for i, v in enumerate(rate.values):
            ax.text(v + 0.01, i, f"%{v * 100:.1f}", va="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Memnun oranı")
    fig.suptitle("Segmentlere göre memnuniyet oranı", fontweight="bold", y=1.04)
    return save(fig, "02_segment_memnuniyet")


def plot_service_means(df: pd.DataFrame, y: pd.Series):
    """Memnun ve memnun olmayan yolcuların hizmet puanı ortalamaları."""
    m = df[C.SERVICE_COLS].groupby(y).mean().T
    m["fark"] = m[1] - m[0]
    m = m.sort_values("fark")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    idx = np.arange(len(m))
    ax.hlines(idx, m[0], m[1], color=GRID, linewidth=3)
    ax.scatter(m[0], idx, color=ORANGE, s=45, label=SAT_NAMES[0], zorder=3)
    ax.scatter(m[1], idx, color=BLUE, s=45, label=SAT_NAMES[1], zorder=3)
    ax.set_yticks(idx, [tr(c) for c in m.index])
    ax.set_xlabel("Ortalama puan (0-5)")
    ax.set_title("Hizmet puanları: memnun vs memnun değil")
    ax.legend(loc="lower right")
    return save(fig, "03_hizmet_puan_farki")


def plot_correlation(df_num: pd.DataFrame):
    corr = df_num.corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [tr(c) if c != "target" else "MEMNUNİYET" for c in corr.columns]
    ax.set_xticks(range(len(corr)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), labels, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title("Korelasyon matrisi")
    return save(fig, "04_korelasyon")


def plot_delays(df: pd.DataFrame, y: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.scatter(df["departure_delay"], df["arrival_delay"], s=2, alpha=0.15, color=BLUE)
    ax.set_xlabel(tr("departure_delay"))
    ax.set_ylabel(tr("arrival_delay"))
    r = df["departure_delay"].corr(df["arrival_delay"])
    ax.set_title(f"Kalkış vs varış gecikmesi (r = {r:.2f})")
    ax = axes[1]
    bins = pd.cut(
        df["arrival_delay"].fillna(df["departure_delay"]),
        [-1, 0, 15, 30, 60, 120, 10_000],
        labels=["0", "1-15", "16-30", "31-60", "61-120", "120+"],
    )
    rate = y.groupby(bins, observed=True).mean()
    ax.bar(rate.index.astype(str), rate.values, color=BLUE, width=0.6)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.01, f"%{v * 100:.0f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(rate.values) * 1.2)
    ax.set_xlabel("Varış gecikmesi (dk)")
    ax.set_ylabel("Memnun oranı")
    ax.set_title("Gecikmeye göre memnuniyet")
    return save(fig, "05_gecikme_analizi")


def plot_zero_ratings(df: pd.DataFrame):
    z = (df[C.SERVICE_COLS] == 0).sum().sort_values()
    z = z[z > 0]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh([tr(c) for c in z.index], z.values, color=GRAY, height=0.6)
    for i, v in enumerate(z.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set_title("'0' puanlar (uygulanamaz / puanlanmamış)")
    ax.set_xlabel("Satır sayısı")
    return save(fig, "06_sifir_puanlar")
