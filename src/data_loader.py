"""Veri yükleme ve veri kalitesi raporu."""

import pandas as pd

from . import config as C


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Kimlik sütunlarını atar, sütun isimlerini snake_case yapar."""
    df = df.drop(columns=[c for c in C.DROP_COLS if c in df.columns])
    missing = set(C.COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(
            f"Beklenen sütunlar bulunamadı: {sorted(missing)}. "
            "Kaggle 'teejmahal20/airline-passenger-satisfaction' train.csv / test.csv "
            "dosyalarını data/raw/ klasörüne koyduğunuzdan emin olun."
        )
    return df.rename(columns=C.COLUMN_MAP)[list(C.COLUMN_MAP.values())]


def load_data():
    """Kaggle'ın hazır train/test bölünmesini yükler.

    Test seti modelleme boyunca hiç kullanılmaz; yalnızca son değerlendirmede açılır.
    """
    for f in (C.TRAIN_FILE, C.TEST_FILE):
        if not f.exists():
            raise FileNotFoundError(f"{f} yok. README'deki 'Veriyi indirme' adımını uygulayın.")
    train = _standardize(pd.read_csv(C.TRAIN_FILE))
    test = _standardize(pd.read_csv(C.TEST_FILE))
    return train, test


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Her sütun için tip, eksik değer, benzersiz değer ve '0 puan' sayısı."""
    rep = pd.DataFrame(
        {
            "tip": df.dtypes.astype(str),
            "eksik": df.isna().sum(),
            "eksik_%": (df.isna().mean() * 100).round(2),
            "benzersiz": df.nunique(),
        }
    )
    rep["sifir_puan"] = [
        int((df[c] == 0).sum()) if c in C.SERVICE_COLS else None for c in df.columns
    ]
    return rep


def summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Veri setinin kısa sayısal özeti."""
    full = pd.concat([train, test])
    return {
        "train_satir": len(train),
        "test_satir": len(test),
        "toplam": len(full),
        "ozellik_sayisi": train.shape[1] - 1,
        "tekrar_eden_satir": int(full.duplicated().sum()),
        "eksik_arrival_delay": int(full["arrival_delay"].isna().sum()),
        "memnun_orani": round((full[C.TARGET] == C.POSITIVE_LABEL).mean(), 4),
        "dep_arr_korelasyon": round(full["departure_delay"].corr(full["arrival_delay"]), 4),
    }
