"""Ön işleme ve McNemar birim testleri:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.preprocessing import AirlinePreprocessor, split_xy


def _sample():
    row = {c: 3 for c in C.SERVICE_COLS}
    row.update(
        {
            "gender": "Female",
            "customer_type": "Loyal Customer",
            "age": 30,
            "type_of_travel": "Business travel",
            "travel_class": "Business",
            "flight_distance": 1000,
            "departure_delay": 10,
            "arrival_delay": np.nan,
            "satisfaction": "satisfied",
        }
    )
    other = dict(
        row,
        gender="Male",
        customer_type="disloyal Customer",
        type_of_travel="Personal Travel",
        travel_class="Eco",
        arrival_delay=5.0,
        satisfaction="neutral or dissatisfied",
    )
    return pd.DataFrame([row, other])


def test_target_encoding():
    _, y = split_xy(_sample())
    assert y.tolist() == [1, 0]


def test_arrival_delay_filled_from_departure():
    X, _ = split_xy(_sample())
    out = AirlinePreprocessor().fit_transform(X)
    assert out["arrival_delay"].tolist() == [10, 5]
    assert not out.isna().any().any()


def test_categorical_encoding():
    X, _ = split_xy(_sample())
    out = AirlinePreprocessor().fit_transform(X)
    assert out["gender"].tolist() == [1, 0]
    assert out["customer_type"].tolist() == [1, 0]
    assert out["type_of_travel"].tolist() == [1, 0]
    assert out["travel_class"].tolist() == [2, 0]  # Business=2, Eco=0


def test_feature_order_is_stable():
    X, _ = split_xy(_sample())
    prep = AirlinePreprocessor().fit(X)
    shuffled = X[X.columns[::-1]]
    assert list(prep.transform(shuffled).columns) == prep.feature_names_


def test_mcnemar_detects_identical_models():
    """Aynı model kendisiyle karşılaştırılınca fark anlamlı çıkmamalı."""
    from sklearn.dummy import DummyClassifier

    from src.analysis import mcnemar_test

    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0, 1] * 50)
    m = DummyClassifier(strategy="most_frequent").fit(X, y)
    res = mcnemar_test(m, m, X, y, n_boot=50)
    assert res["a_dogru_b_yanlis"] == 0 and res["a_yanlis_b_dogru"] == 0
    assert res["anlamli_mi_(p<0.05)"] is False
