"""Model tanımları, hiperparametre araması ve eğitim."""

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .preprocessing import AirlinePreprocessor

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", AirlinePreprocessor()), ("model", estimator)])


# Arama uzayları: aşırı öğrenmeyi kontrol eden parametreler (derinlik, yaprak boyutu)
SEARCH_SPACES = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=C.RANDOM_STATE),
        {
            "model__max_depth": [5, 8, 10, 12, 15, 20, None],
            "model__min_samples_split": [2, 10, 20, 50],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__criterion": ["gini", "entropy"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=C.RANDOM_STATE, n_jobs=C.N_JOBS),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, 30, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
    ),
}


def baseline(X, y) -> Pipeline:
    """Referans model: her zaman çoğunluk sınıfını tahmin eder."""
    return make_pipeline(DummyClassifier(strategy="most_frequent")).fit(X, y)


def tune(name: str, X, y, verbose: int = 1):
    """RandomizedSearchCV ile en iyi hiperparametreleri bulur (F1'e göre)."""
    est, space = SEARCH_SPACES[name]
    search = RandomizedSearchCV(
        make_pipeline(est),
        space,
        n_iter=C.SEARCH_ITER,
        scoring="f1",
        cv=StratifiedKFold(C.SEARCH_CV, shuffle=True, random_state=C.RANDOM_STATE),
        random_state=C.RANDOM_STATE,
        n_jobs=C.N_JOBS if name != "Random Forest" else 1,
        refit=True,
        verbose=verbose,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    search.fit_seconds_ = time.perf_counter() - t0
    return search


def cv_report(pipe: Pipeline, X, y) -> dict:
    """En iyi modelin 5 katlı çapraz doğrulama sonuçları (ortalama ± std)."""
    res = cross_validate(
        pipe,
        X,
        y,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=1,
        cv=StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE),
    )
    out = {}
    for m in SCORING:
        out[f"cv_{m}_mean"] = float(np.mean(res[f"test_{m}"]))
        out[f"cv_{m}_std"] = float(np.std(res[f"test_{m}"]))
        out[f"train_{m}_mean"] = float(np.mean(res[f"train_{m}"]))
    out["fit_time_mean_s"] = float(np.mean(res["fit_time"]))
    return out


def depth_curve(X, y, depths=range(1, 31)) -> pd.DataFrame:
    """Karar ağacı derinliğine göre eğitim/doğrulama doğruluğu.

    Aşırı öğrenme (overfitting) ve eksik öğrenmeyi (underfitting) göstermek için.
    """
    rows = []
    cv = StratifiedKFold(C.SEARCH_CV, shuffle=True, random_state=C.RANDOM_STATE)
    for d in depths:
        pipe = make_pipeline(DecisionTreeClassifier(max_depth=d, random_state=C.RANDOM_STATE))
        r = cross_validate(
            pipe, X, y, cv=cv, scoring="accuracy", return_train_score=True, n_jobs=C.N_JOBS
        )
        rows.append(
            {"max_depth": d, "train": r["train_score"].mean(), "validation": r["test_score"].mean()}
        )
    return pd.DataFrame(rows)
