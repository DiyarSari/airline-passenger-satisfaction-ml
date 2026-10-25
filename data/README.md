# Veri

## data/raw/ — Ana veri seti (Kaggle)
**Airline Passenger Satisfaction** — https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

1. Kaggle'a giriş yapın, sayfadan **Download** ile zip'i indirin.
2. Zip içindeki `train.csv` ve `test.csv` dosyalarını bu klasöre (`data/raw/`) koyun.

Beklenen boyut: train 103.904 satır, test 25.976 satır, 25 sütun.

> Not: Veri setinin asıl kaynağı Kaggle'da belirtilmemiştir (başka bir Kaggle yüklemesinden
> türetilmiştir). Bu nedenle bulgular, kaynağı belli Skytrax yorumlarıyla ayrıca doğrulanır.

## data/external/ — Skytrax gerçek yorumları
`python main.py` çalışınca otomatik indirilir (`skytrax_airline.csv`).
Kaynak: https://github.com/quankiquanki/skytrax-reviews-dataset (airlinequality.com yorumları,
Ağustos 2015 anlık görüntüsü; depo CC0-1.0 lisanslı, yorumların özgün lisansı belirtilmemiş).
