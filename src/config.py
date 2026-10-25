"""Proje genelinde kullanılan sabitler: yollar, sütun grupları, model ayarları."""

import os
from pathlib import Path

# ---------------------------------------------------------------- Yollar
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"  # Kaggle train.csv / test.csv
DATA_EXT = ROOT / "data" / "external"  # Skytrax skytrax_airline.csv (otomatik iner)
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
MET_DIR = OUT / "metrics"
MODEL_DIR = OUT / "models"

TRAIN_FILE = DATA_RAW / "train.csv"
TEST_FILE = DATA_RAW / "test.csv"
SKYTRAX_FILE = DATA_EXT / "skytrax_airline.csv"
SKYTRAX_URL = (
    "https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv"
)

RANDOM_STATE = 42
CV_FOLDS = 5  # en iyi modelin raporlanan çapraz doğrulaması
N_JOBS = -1  # tüm CPU çekirdekleri
# Hızlı deneme için:  SEARCH_ITER=4 SEARCH_CV=2 python main.py
SEARCH_ITER = int(os.getenv("SEARCH_ITER", 15))  # RandomizedSearchCV deneme sayısı
SEARCH_CV = int(os.getenv("SEARCH_CV", 3))  # arama sırasındaki kat sayısı
SHAP_SAMPLE = int(os.getenv("SHAP_SAMPLE", 1000))  # SHAP hesaplanacak test örneği

# ------------------------------------------------- Kanonik sütun isimleri
# Kaggle sürümündeki orijinal isim -> projede kullanılan snake_case isim
COLUMN_MAP = {
    "Gender": "gender",
    "Customer Type": "customer_type",
    "Age": "age",
    "Type of Travel": "type_of_travel",
    "Class": "travel_class",
    "Flight Distance": "flight_distance",
    "Inflight wifi service": "inflight_wifi",
    "Departure/Arrival time convenient": "time_convenience",
    "Ease of Online booking": "online_booking",
    "Gate location": "gate_location",
    "Food and drink": "food_drink",
    "Online boarding": "online_boarding",
    "Seat comfort": "seat_comfort",
    "Inflight entertainment": "inflight_entertainment",
    "On-board service": "onboard_service",
    "Leg room service": "leg_room",
    "Baggage handling": "baggage_handling",
    "Checkin service": "checkin_service",
    "Inflight service": "inflight_service",
    "Cleanliness": "cleanliness",
    "Departure Delay in Minutes": "departure_delay",
    "Arrival Delay in Minutes": "arrival_delay",
    "satisfaction": "satisfaction",
}
DROP_COLS = ["Unnamed: 0", "id"]  # anlamsız indeks/kimlik sütunları

TARGET = "satisfaction"
POSITIVE_LABEL = "satisfied"  # 1 = memnun, 0 = nötr veya memnun değil

# 0-5 ölçekli hizmet puanları (0 = "uygulanamaz / puanlanmadı")
SERVICE_COLS = [
    "inflight_wifi",
    "time_convenience",
    "online_booking",
    "gate_location",
    "food_drink",
    "online_boarding",
    "seat_comfort",
    "inflight_entertainment",
    "onboard_service",
    "leg_room",
    "baggage_handling",
    "checkin_service",
    "inflight_service",
    "cleanliness",
]
NUMERIC_COLS = ["age", "flight_distance", "departure_delay", "arrival_delay"]
BINARY_COLS = {  # sütun -> 1 olarak kodlanacak değer
    "gender": "Female",
    "customer_type": "Loyal Customer",
    "type_of_travel": "Business travel",
}
CLASS_ORDER = ["Eco", "Eco Plus", "Business"]  # sıralı (ordinal) kodlama

# Grafiklerde kullanılacak Türkçe etiketler
TR_LABELS = {
    "gender": "Cinsiyet (K=1)",
    "customer_type": "Sadık müşteri",
    "age": "Yaş",
    "type_of_travel": "İş seyahati",
    "travel_class": "Kabin sınıfı",
    "flight_distance": "Uçuş mesafesi",
    "inflight_wifi": "Uçak içi Wi-Fi",
    "time_convenience": "Kalkış/varış saati uygunluğu",
    "online_booking": "Online rezervasyon kolaylığı",
    "gate_location": "Kapı konumu",
    "food_drink": "Yiyecek-içecek",
    "online_boarding": "Online boarding",
    "seat_comfort": "Koltuk konforu",
    "inflight_entertainment": "Uçak içi eğlence",
    "onboard_service": "Kabin hizmeti",
    "leg_room": "Diz mesafesi",
    "baggage_handling": "Bagaj hizmeti",
    "checkin_service": "Check-in hizmeti",
    "inflight_service": "Uçak içi servis",
    "cleanliness": "Temizlik",
    "departure_delay": "Kalkış gecikmesi (dk)",
    "arrival_delay": "Varış gecikmesi (dk)",
}

# ---------------------------------------------- Literatür karşılaştırması
# Aynı veri setini kullanan çalışmaların raporladığı en iyi test doğrulukları
LITERATURE = [
    {
        "study": "Javadi, Ergin & Yüksel (2026), J. Turkish Operations Management",
        "methods": "LR, DT, RF, SVM, KNN",
        "best": "RF",
        "accuracy": 0.9461,
    },
    {
        "study": "Fadri vd. (2025), IJATIS",
        "methods": "KNN, NB, DT, RF, SVM",
        "best": "RF",
        "accuracy": 0.9578,
    },
    {
        "study": "Zhang (2024), EMITI - SciTePress",
        "methods": "LR, RF, AdaBoost, XGBoost + SHAP",
        "best": "XGBoost",
        "accuracy": 0.951,
    },
    {
        "study": "Xue (2025), IAMPA - SciTePress",
        "methods": "PCA + LR, RF",
        "best": "RF",
        "accuracy": 0.92,
    },
    {
        "study": "Hong vd., DAAM (UMP)",
        "methods": "KNN, DT, LR, RF, NB, AdaBoost",
        "best": "RF",
        "accuracy": 0.892,
    },
]

# --------------------------------------------------------------- Skytrax
SKY_RATINGS = [
    "seat_comfort_rating",
    "cabin_staff_rating",
    "food_beverages_rating",
    "inflight_entertainment_rating",
    "value_money_rating",
]
SKY_TR = {
    "seat_comfort_rating": "Koltuk konforu",
    "cabin_staff_rating": "Kabin ekibi",
    "food_beverages_rating": "Yiyecek-içecek",
    "inflight_entertainment_rating": "Uçak içi eğlence",
    "value_money_rating": "Fiyat/performans",
    "cabin_business": "Business kabin",
}
# Skytrax faktörünün Kaggle veri setindeki karşılığı (bulgu teyidi için)
SKY_TO_APS = {
    "seat_comfort_rating": "seat_comfort",
    "cabin_staff_rating": "onboard_service",
    "food_beverages_rating": "food_drink",
    "inflight_entertainment_rating": "inflight_entertainment",
}

for _d in (FIG_DIR, MET_DIR, MODEL_DIR, DATA_RAW, DATA_EXT):
    _d.mkdir(parents=True, exist_ok=True)
