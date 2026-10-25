"""Skytrax gerçek yolcu yorumları ile bulguların dış doğrulaması.

Amaç: Kaggle veri setinde bulunan 'memnuniyeti belirleyen faktörler' bulgusunun,
kaynağı belli gerçek yorumlarda (airlinequality.com) da geçerli olup olmadığını
görmek ve Turkish Airlines yorumlarını ayrıca incelemek.

Tasarım:
- Hedef: 'recommended' (yolcu havayolunu tavsiye ediyor mu?) -> memnuniyet vekili.
- Özellikler: 5 alt puan (0-5) + Business/First kabin bayrağı.
- 'overall_rating' KULLANILMAZ: tavsiye kararıyla neredeyse aynı bilgidir
  (hedef sızıntısı / target leakage).
- Turkish Airlines yorumları eğitimden tamamen çıkarılır; model diğer
  havayollarında eğitilip THY üzerinde test edilir (dış örneklem testi).
"""

import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .evaluation import test_metrics
from .visualization import BLUE, GRAY, ORANGE, save

THY = "turkish-airlines"
FEATURES = C.SKY_RATINGS + ["cabin_business"]


def download():
    if not C.SKYTRAX_FILE.exists():
        print("Skytrax verisi indiriliyor...")
        urllib.request.urlretrieve(C.SKYTRAX_URL, C.SKYTRAX_FILE)
    return C.SKYTRAX_FILE


def load_clean():
    raw = pd.read_csv(download())
    info = {"ham_yorum": len(raw)}
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].dt.year >= 2000]  # 1970 gibi hatalı tarihler
    df = df.dropna(subset=C.SKY_RATINGS)  # 5 alt puanın tamamı dolu
    df["cabin_business"] = df["cabin_flown"].isin(["Business Class", "First Class"]).astype(int)
    df["is_thy"] = (df["airline_name"] == THY).astype(int)
    info.update(
        {
            "temiz_yorum": len(df),
            "havayolu_sayisi": df["airline_name"].nunique(),
            "tarih_araligi": f"{df['date'].min():%Y-%m} / {df['date'].max():%Y-%m}",
            "thy_yorum": int(df["is_thy"].sum()),
            "tavsiye_orani": round(df["recommended"].mean(), 4),
            "overall_recommended_korelasyon": round(
                raw[["overall_rating", "recommended"]].corr().iloc[0, 1], 4
            ),
            "eksik_oranlari": raw[
                C.SKY_RATINGS + ["ground_service_rating", "wifi_connectivity_rating"]
            ]
            .isna()
            .mean()
            .round(3)
            .to_dict(),
        }
    )
    return df, info


def train_models(df):
    other, thy = df[df.is_thy == 0], df[df.is_thy == 1]
    Xtr, Xte, ytr, yte = train_test_split(
        other[FEATURES],
        other["recommended"],
        test_size=0.2,
        stratify=other["recommended"],
        random_state=C.RANDOM_STATE,
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=C.RANDOM_STATE)
    dt = GridSearchCV(
        DecisionTreeClassifier(random_state=C.RANDOM_STATE),
        {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_leaf": [1, 10, 30]},
        scoring="f1",
        cv=cv,
        n_jobs=C.N_JOBS,
    ).fit(Xtr, ytr)
    rf = GridSearchCV(
        RandomForestClassifier(n_estimators=200, random_state=C.RANDOM_STATE, n_jobs=C.N_JOBS),
        {"max_depth": [5, 10, None], "min_samples_leaf": [1, 10]},
        scoring="f1",
        cv=cv,
        n_jobs=1,
    ).fit(Xtr, ytr)
    models = {"Decision Tree": dt.best_estimator_, "Random Forest": rf.best_estimator_}
    rows = []
    for name, m in models.items():
        for split, X, y in (
            ("diğer havayolları (test)", Xte, yte),
            ("Turkish Airlines (dış örneklem)", thy[FEATURES], thy["recommended"]),
        ):
            r = test_metrics(name, m, X, y)
            r["veri"] = split
            rows.append(r)
    params = {"Decision Tree": dt.best_params_, "Random Forest": rf.best_params_}
    return models, pd.DataFrame(rows), params, (Xte, thy)


def shap_ranking(rf, X) -> pd.DataFrame:
    sv = shap.TreeExplainer(rf)(
        X.sample(min(2000, len(X)), random_state=C.RANDOM_STATE), check_additivity=False
    )
    vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(vals).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def compare_with_aps(sky_rank: pd.DataFrame, aps_rank: pd.DataFrame) -> pd.DataFrame:
    """Ortak 4 faktörün iki veri setindeki göreli önemi (toplamı 1'e normalize)."""
    aps = aps_rank.set_index("feature")["mean_abs_shap"]
    sky = sky_rank.set_index("feature")["mean_abs_shap"]
    rows = [
        {"faktor": C.SKY_TR[s], "skytrax": sky[s], "kaggle": aps[C.TR_LABELS[a]]}
        for s, a in C.SKY_TO_APS.items()
    ]
    d = pd.DataFrame(rows)
    for c in ("skytrax", "kaggle"):
        d[c] = d[c] / d[c].sum()
    d["skytrax_sira"] = d["skytrax"].rank(ascending=False).astype(int)
    d["kaggle_sira"] = d["kaggle"].rank(ascending=False).astype(int)
    return d.sort_values("kaggle", ascending=False)


def thy_profile(df) -> pd.DataFrame:
    g = df.groupby("is_thy")[C.SKY_RATINGS + ["recommended"]].mean().T
    g.columns = ["Diğer havayolları", "Turkish Airlines"]
    g.index = [C.SKY_TR.get(i, "Tavsiye oranı") for i in g.index]
    g["fark"] = g["Turkish Airlines"] - g["Diğer havayolları"]
    return g.round(3)


# -------------------------------------------------------------- Grafikler
def plot_factor_comparison(comp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    idx = np.arange(len(comp))
    ax.bar(idx - 0.2, comp["kaggle"], 0.38, color=BLUE, label="Kaggle (129.880 yolcu)")
    ax.bar(idx + 0.2, comp["skytrax"], 0.38, color=ORANGE, label="Skytrax (gerçek yorumlar)")
    ax.set_xticks(idx, comp["faktor"])
    ax.set_ylabel("Göreli önem (toplam = 1)")
    ax.set_title("Ortak 4 faktörün göreli SHAP önemi: Kaggle vs Skytrax")
    ax.legend(fontsize=8)
    return save(fig, "17_skytrax_faktor_karsilastirma")


def plot_thy_profile(prof: pd.DataFrame):
    d = prof.drop(index="Tavsiye oranı")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    idx = np.arange(len(d))
    ax.barh(idx - 0.2, d["Diğer havayolları"], 0.38, color=GRAY, label="Diğer havayolları")
    ax.barh(idx + 0.2, d["Turkish Airlines"], 0.38, color=BLUE, label="Turkish Airlines")
    for i, (o, t) in enumerate(zip(d["Diğer havayolları"], d["Turkish Airlines"])):
        ax.text(max(o, t) + 0.05, i, f"{t - o:+.2f}", va="center", fontsize=8)
    ax.set_yticks(idx, d.index)
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Ortalama puan (0-5)")
    r = prof.loc["Tavsiye oranı"]
    ax.set_title(
        f"Skytrax: THY vs diğerleri (tavsiye: %{r['Turkish Airlines'] * 100:.0f} "
        f"vs %{r['Diğer havayolları'] * 100:.0f})"
    )
    ax.legend(fontsize=8, loc="lower right")
    return save(fig, "18_skytrax_thy_profil")
