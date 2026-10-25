"""Ek analizler: senaryo (ne olur?) analizi, hata analizi, istatistiksel anlamlılık.

1) Senaryo analizi: Modelin "memnun değil" dediği yolcularda bir hizmetin puanı
   1 artırılırsa model kaç yolcuyu "memnun"a çeviriyor?
   ÖNEMLİ: Bu bir nedensellik iddiası DEĞİLDİR. Modelin öğrendiği ilişkiye göre
   bir öngörüdür; gerçek yatırım kararı için kontrollü deney (A/B testi) gerekir.
2) Hata analizi: Model kimi yanlış tahmin ediyor, bu yolcuların ortak profili ne?
3) McNemar testi: RF ile DT'nin aynı test yolcuları üzerindeki doğru/yanlış
   tahminleri karşılaştırılır; fark tesadüf olabilir mi?
   Ek olarak doğruluk farkı için bootstrap %95 güven aralığı hesaplanır.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from . import config as C
from .visualization import BLUE, GRAY, ORANGE, TEXT, VALUE_TR, save, tr


# ------------------------------------------------------------ 1) Senaryo
def what_if(pipe, X_test: pd.DataFrame, step: int = 1) -> pd.DataFrame:
    """Her hizmet için: puanı +step artırınca 'memnun'a dönen yolcu sayısı.

    Yalnızca puanı 1-4 olan yolcular değiştirilir (0 = uygulanamaz, 5 = zaten tavan).
    """
    base_pred = pipe.predict(X_test)
    base_proba = pipe.predict_proba(X_test)[:, 1]
    target = base_pred == 0  # model 'memnun değil' diyor
    n_target = int(target.sum())
    rows = []
    for col in C.SERVICE_COLS:
        Xw = X_test.copy()
        eligible = target & X_test[col].between(1, 5 - step).values
        Xw.loc[eligible, col] = Xw.loc[eligible, col] + step
        new_pred = pipe.predict(Xw)
        new_proba = pipe.predict_proba(Xw)[:, 1]
        flipped = int(((new_pred == 1) & target).sum())
        rows.append(
            {
                "hizmet": tr(col),
                "sutun": col,
                "uygun_yolcu": int(eligible.sum()),
                "memnuna_donen": flipped,
                "donusum_%": round(100 * flipped / n_target, 2),
                "uygunlar_icinde_%": round(100 * flipped / max(int(eligible.sum()), 1), 2),
                "ort_olasilik_artisi": round(float((new_proba - base_proba)[eligible].mean()), 4)
                if eligible.any()
                else 0.0,
            }
        )
    out = pd.DataFrame(rows).sort_values("memnuna_donen", ascending=False).reset_index(drop=True)
    out.attrs["memnun_degil_tahmini"] = n_target
    return out


def what_if_by_class(pipe, X_test: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Aynı senaryo, kabin sınıfına göre: Economy ve Business için ilk 'top' hizmet."""
    rows = []
    for cls in ("Eco", "Business"):
        mask = X_test["travel_class"] == cls
        res = what_if(pipe, X_test[mask])
        for r, (_, row) in enumerate(res.head(top).iterrows(), 1):
            rows.append(
                {
                    "sinif": VALUE_TR.get(cls, cls),
                    "sira": r,
                    "hizmet": row["hizmet"],
                    "donusum_%": row["donusum_%"],
                    "memnuna_donen": row["memnuna_donen"],
                }
            )
    return pd.DataFrame(rows)


def plot_what_if(res: pd.DataFrame, top: int = 10):
    d = res.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(d["hizmet"], d["donusum_%"], color=BLUE, height=0.6)
    for i, (p, n) in enumerate(zip(d["donusum_%"], d["memnuna_donen"])):
        ax.text(p, i, f"  %{p:.1f}  ({n:,} yolcu)".replace(",", "."), va="center", fontsize=8)
    ax.set_xlim(0, d["donusum_%"].max() * 1.35)
    ax.set_xlabel("'Memnun değil' tahmin edilen yolcular içinde memnuna dönenlerin oranı")
    ax.set_title("Senaryo: hizmet puanı +1 artarsa model kaç yolcuyu kazanıyor?")
    return save(fig, "19_senaryo_analizi")


# ------------------------------------------------------------ 2) Hata analizi
def error_analysis(pipe, X_test: pd.DataFrame, y_test: pd.Series):
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    y = y_test.values
    kind = np.select(
        [(pred == 1) & (y == 0), (pred == 0) & (y == 1)],
        ["Yanlış pozitif", "Yanlış negatif"],
        "Doğru",
    )
    df = X_test.assign(_tur=kind, _olasilik=proba, _y=y)

    # Segment bazında hata oranı
    seg_rows = []
    for col, name in (
        ("travel_class", "Kabin sınıfı"),
        ("type_of_travel", "Seyahat tipi"),
        ("customer_type", "Müşteri tipi"),
    ):
        g = df.groupby(df[col].replace(VALUE_TR))
        for val, part in g:
            seg_rows.append(
                {
                    "degisken": name,
                    "deger": val,
                    "yolcu": len(part),
                    "hata_%": round(100 * (part["_tur"] != "Doğru").mean(), 2),
                }
            )
    segments = pd.DataFrame(seg_rows)

    # Hatalı ve doğru tahminlerin hizmet puanı ortalamaları
    profile = df.groupby("_tur")[C.SERVICE_COLS + ["departure_delay"]].mean().T.round(2)
    profile.index = [tr(i) for i in profile.index]

    errors = df[df["_tur"] != "Doğru"]
    summary = {
        "test_yolcu": len(df),
        "hatali_tahmin": int(len(errors)),
        "hata_orani_%": round(100 * len(errors) / len(df), 2),
        "yanlis_pozitif": int((kind == "Yanlış pozitif").sum()),
        "yanlis_negatif": int((kind == "Yanlış negatif").sum()),
        # Model kararsız mıydı? Olasılık 0.35-0.65 arasındaysa 'sınırda' sayılır
        "hatalarin_sinirda_%": round(100 * errors["_olasilik"].between(0.35, 0.65).mean(), 2),
        "dogrularin_sinirda_%": round(
            100 * df.loc[df["_tur"] == "Doğru", "_olasilik"].between(0.35, 0.65).mean(), 2
        ),
    }
    return summary, segments, profile, df


def plot_errors(segments: pd.DataFrame, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    s = segments.copy()
    s["etiket"] = s["degisken"] + ": " + s["deger"].astype(str)
    s = s.sort_values("hata_%")
    ax.barh(s["etiket"], s["hata_%"], color=ORANGE, height=0.6)
    for i, v in enumerate(s["hata_%"]):
        ax.text(v, i, f" %{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Hata oranı (%)")
    ax.set_title("Segmentlere göre hata oranı")
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ok = df.loc[df["_tur"] == "Doğru", "_olasilik"]
    err = df.loc[df["_tur"] != "Doğru", "_olasilik"]
    ax.hist(ok, bins=bins, color=GRAY, alpha=0.6, density=True, label="Doğru tahminler")
    ax.hist(err, bins=bins, color=ORANGE, alpha=0.8, density=True, label="Hatalı tahminler")
    ax.axvline(0.5, color=TEXT, lw=1, ls="--")
    ax.set_xlabel("Modelin 'memnun' olasılığı")
    ax.set_ylabel("Yoğunluk")
    ax.set_title("Hatalar sınırda mı? (olasılık dağılımı)")
    ax.legend(fontsize=8)
    return save(fig, "20_hata_analizi")


# ------------------------------------------------------------ 3) McNemar
def mcnemar_test(
    pipe_a, pipe_b, X_test, y_test, names=("Random Forest", "Decision Tree"), n_boot: int = 1000
) -> dict:
    """McNemar testi + doğruluk farkı için bootstrap %95 güven aralığı."""
    y = y_test.values
    a_ok = pipe_a.predict(X_test) == y
    b_ok = pipe_b.predict(X_test) == y
    n01 = int((a_ok & ~b_ok).sum())  # A doğru, B yanlış
    n10 = int((~a_ok & b_ok).sum())  # A yanlış, B doğru
    n = n01 + n10
    if n == 0:  # iki model her yolcuda aynı sonucu verdi
        p, stat, method = 1.0, None, "no disagreements"
    elif n < 25:  # az uyuşmazlıkta kesin (binom) test
        p = binomtest(min(n01, n10), n, 0.5).pvalue
        stat, method = None, "exact binomial"
    else:  # süreklilik düzeltmeli ki-kare
        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = float(chi2.sf(stat, 1))
        method = "chi-square (continuity corrected)"
    rng = np.random.default_rng(C.RANDOM_STATE)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(a_ok[idx].mean() - b_ok[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "model_a": names[0],
        "model_b": names[1],
        "a_dogru_b_yanlis": n01,
        "a_yanlis_b_dogru": n10,
        "istatistik": None if stat is None else round(float(stat), 3),
        "p_degeri": float(p),
        "yontem": method,
        "dogruluk_farki": round(float(a_ok.mean() - b_ok.mean()), 4),
        "fark_%95_GA": [round(float(lo), 4), round(float(hi), 4)],
        "anlamli_mi_(p<0.05)": bool(p < 0.05),
    }
