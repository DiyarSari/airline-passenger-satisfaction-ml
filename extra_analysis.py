"""Ek analizler (senaryo, hata analizi, McNemar).

main.py daha önce çalıştırıldıysa yalnızca bu adım tekrarlanabilir (~1-2 dk):
    python extra_analysis.py
Kaydedilmiş modelleri outputs/models/ klasöründen okur; yeniden eğitim yapmaz.
"""

import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import analysis, data_loader, visualization
from src import config as C
from src.preprocessing import split_xy

warnings.filterwarnings("ignore")


def run(rf, dt, X_test, y_test) -> dict:
    """Senaryo, hata ve McNemar analizlerini çalıştırır; sonuçları outputs/ altına yazar."""
    visualization.set_style()
    out = {}

    print("\n--> Senaryo analizi: hizmet puanı +1 artarsa?")
    wi = analysis.what_if(rf, X_test)
    wi.to_csv(C.MET_DIR / "senaryo_analizi.csv", index=False)
    analysis.plot_what_if(wi)
    by_class = analysis.what_if_by_class(rf, X_test)
    by_class.to_csv(C.MET_DIR / "senaryo_kabin_sinifi.csv", index=False)
    print(f"Model 'memnun değil' dediği yolcu: {wi.attrs['memnun_degil_tahmini']:,}")
    cols = ["hizmet", "uygun_yolcu", "memnuna_donen", "donusum_%"]
    print(wi[cols].head(8).to_string(index=False))
    print(by_class.to_string(index=False))
    out["memnun_degil_tahmini"] = int(wi.attrs["memnun_degil_tahmini"])
    out["senaryo_ilk5"] = wi.head(5)[["hizmet", "memnuna_donen", "donusum_%"]].to_dict("records")

    print("\n--> Hata analizi")
    summ, segments, profile, df = analysis.error_analysis(rf, X_test, y_test)
    segments.to_csv(C.MET_DIR / "hata_segmentleri.csv", index=False)
    profile.to_csv(C.MET_DIR / "hata_profili.csv")
    analysis.plot_errors(segments, df)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    print(segments.to_string(index=False))
    out["hata_analizi"] = summ

    print("\n--> McNemar testi: Random Forest vs Decision Tree")
    mc = analysis.mcnemar_test(rf, dt, X_test, y_test)
    print(json.dumps(mc, indent=2, ensure_ascii=False))
    out["mcnemar"] = mc

    with open(C.MET_DIR / "ek_analizler.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    plt.close("all")
    return out


if __name__ == "__main__":
    for name in ("random_forest", "decision_tree"):
        if not (C.MODEL_DIR / f"{name}.joblib").exists():
            raise FileNotFoundError("Önce 'python main.py' çalıştırın (modeller kaydedilmemiş).")
    _, test = data_loader.load_data()
    X_test, y_test = split_xy(test)
    run(
        joblib.load(C.MODEL_DIR / "random_forest.joblib"),
        joblib.load(C.MODEL_DIR / "decision_tree.joblib"),
        X_test,
        y_test,
    )
    print("\nBitti. Yeni çıktılar: outputs/figures/19-20, outputs/metrics/ek_analizler.json")
