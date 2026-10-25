"""Veri ön işleme.

Kararlar:
1. Eksik 'arrival_delay' -> aynı uçuşun 'departure_delay' değeri ile doldurulur
   (iki sütun arasında r ≈ 0.965; genel ortalamadan daha isabetli bir tahmin).
2. İkili kategorikler (cinsiyet, müşteri tipi, seyahat tipi) -> 0/1.
3. Kabin sınıfı -> sıralı kodlama Eco=0 < Eco Plus=1 < Business=2.
4. Hizmet puanlarındaki 0 değeri 'uygulanamaz' anlamına gelir; ağaç tabanlı
   modeller 0'ı ayrı bir dalda yakalayabildiği için ayrı bir değer olarak korunur.
5. Ölçekleme yapılmaz: karar ağaçları ve rastgele orman ölçekten bağımsızdır.

Dönüştürücü sklearn Pipeline içinde kullanılır; böylece fit yalnızca eğitim
verisinde yapılır ve test setinden bilgi sızmaz (data leakage).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C


class AirlinePreprocessor(BaseEstimator, TransformerMixin):
    """Ham Kaggle sütunlarını modele hazır sayısal özelliklere dönüştürür."""

    def fit(self, X: pd.DataFrame, y=None):
        # departure_delay da boşsa kullanılacak yedek değer (yalnızca eğitimden öğrenilir)
        self.arrival_median_ = float(X["arrival_delay"].median())
        self.feature_names_ = self._transform(X).columns.tolist()
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["arrival_delay"] = (
            X["arrival_delay"]
            .fillna(X["departure_delay"])
            .fillna(getattr(self, "arrival_median_", 0.0))
        )
        for col, pos in C.BINARY_COLS.items():
            X[col] = (X[col] == pos).astype(int)
        X["travel_class"] = (
            X["travel_class"].map({v: i for i, v in enumerate(C.CLASS_ORDER)}).astype(int)
        )
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def split_xy(df: pd.DataFrame):
    """Özellikler (X) ve hedef (y: 1 = memnun) ayrımı."""
    y = (df[C.TARGET] == C.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[C.TARGET])
    return X, y
