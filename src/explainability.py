"""Açıklanabilirlik: özellik önemleri ve SHAP analizi.

Üç farklı önem ölçüsü karşılaştırılır:
- Impurity (Gini) önemi: RF'nin kendi ölçüsü; hızlı ama yüksek kardinaliteli
  sayısal özellikleri kayırma eğilimi vardır.
- Permutation önemi: bir özellik karıştırılınca test başarısı ne kadar düşüyor.
- SHAP: her tahmini özelliklerin katkılarına ayırır; yön (+/-) bilgisi de verir.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from . import config as C
from .visualization import BLUE, GRAY, ORANGE, save, tr


def importance_table(pipe, X_test, y_test, n_sample=5000) -> pd.DataFrame:
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xs = X_test.sample(min(n_sample, len(X_test)), random_state=C.RANDOM_STATE)
    perm = permutation_importance(
        pipe,
        Xs,
        y_test.loc[Xs.index],
        scoring="f1",
        n_repeats=5,
        random_state=C.RANDOM_STATE,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": prep.feature_names_,
            "impurity": model.feature_importances_,
            "permutation": perm.importances_mean,
        }
    )
    return df.sort_values("permutation", ascending=False).reset_index(drop=True)


def plot_importances(imp: pd.DataFrame, top=12):
    d = imp.head(top).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    axes[0].barh([tr(f) for f in d.feature], d.impurity, color=GRAY, height=0.6)
    axes[0].set_title("Impurity (Gini) önemi")
    axes[1].barh([tr(f) for f in d.feature], d.permutation, color=BLUE, height=0.6)
    axes[1].set_title("Permutation önemi (F1 düşüşü)")
    fig.suptitle("Random Forest özellik önemleri", fontweight="bold", y=1.02)
    return save(fig, "12_ozellik_onemleri")


def shap_values(pipe, X_test):
    """Random Forest için SHAP değerleri (pozitif sınıf = memnun)."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xs = prep.transform(X_test.sample(min(C.SHAP_SAMPLE, len(X_test)), random_state=C.RANDOM_STATE))
    explainer = shap.TreeExplainer(model)
    sv = explainer(Xs, check_additivity=False)
    if sv.values.ndim == 3:  # (n, özellik, sınıf) -> memnun sınıfı
        sv = sv[:, :, 1]
    sv.feature_names = [tr(f) for f in Xs.columns]
    return sv, Xs


def plot_shap_summary(sv):
    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.title("SHAP özet grafiği: hangi özellik memnuniyeti nasıl etkiliyor?")
    save(fig, "13_shap_beeswarm")
    plt.close(fig)
    plt.figure()
    shap.plots.bar(sv, max_display=14, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7, 5.5)
    plt.title("Ortalama |SHAP| değeri")
    save(fig, "14_shap_bar")
    plt.close(fig)


def shap_ranking(sv) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": np.abs(sv.values).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def segment_shap(sv, Xs, col="travel_class", groups=None) -> pd.DataFrame:
    """Segment bazında ortalama |SHAP|: Business ve Economy yolcusunu ne memnun eder?"""
    groups = groups or {"Economy": Xs[col] == 0, "Business": Xs[col] == 2}
    out = pd.DataFrame(
        {g: np.abs(sv.values[mask.values]).mean(0) for g, mask in groups.items()},
        index=sv.feature_names,
    )
    return out


def plot_segment_shap(
    seg: pd.DataFrame, top=10, title="Kabin sınıfına göre SHAP önemi", name="15_shap_segment"
):
    d = seg.loc[seg.max(axis=1).sort_values(ascending=False).index[:top]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    idx = np.arange(len(d))
    cols = list(d.columns)
    ax.barh(idx - 0.2, d[cols[0]], 0.38, color=ORANGE, label=cols[0])
    ax.barh(idx + 0.2, d[cols[1]], 0.38, color=BLUE, label=cols[1])
    ax.set_yticks(idx, d.index)
    ax.set_xlabel("Ortalama |SHAP|")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return save(fig, name)


def plot_waterfall(sv, i=0):
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.title("Tek bir yolcunun tahmin açıklaması")
    save(fig, "16_shap_tek_yolcu")
    plt.close(fig)
