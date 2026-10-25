"""Uçtan uca proje akışı.

Çalıştırma:  python main.py
Hızlı deneme: SEARCH_ITER=4 SEARCH_CV=2 SHAP_SAMPLE=300 python main.py   (Windows: README'ye bakın)

Adımlar:
 1. Veri yükleme ve kalite raporu
 2. Keşifsel veri analizi (EDA) grafikleri
 3. Baseline + Decision Tree + Random Forest (hiperparametre araması + 5-katlı CV)
 4. Test seti değerlendirmesi ve literatür karşılaştırması
 5. Açıklanabilirlik (özellik önemleri + SHAP + segment analizi)
 6. Skytrax gerçek yorumlarla dış doğrulama (Turkish Airlines dahil)
 7. Ek analizler: senaryo (ne olur?), hata analizi, McNemar testi
Tüm çıktılar outputs/ klasörüne yazılır.
"""

import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import extra_analysis
from src import config as C
from src import data_loader, evaluation, explainability, models, skytrax, visualization
from src.preprocessing import AirlinePreprocessor, split_xy

warnings.filterwarnings("ignore")


def step(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def main():
    t_start = time.perf_counter()
    visualization.set_style()
    results = {}

    # 1 ----------------------------------------------------------- Veri
    step("1) Veri yükleniyor")
    train, test = data_loader.load_data()
    results["veri"] = data_loader.summary(train, test)
    data_loader.quality_report(pd.concat([train, test])).to_csv(C.MET_DIR / "veri_kalitesi.csv")
    print(json.dumps(results["veri"], indent=2, ensure_ascii=False))
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    # 2 ----------------------------------------------------------- EDA
    step("2) Keşifsel veri analizi")
    full = pd.concat([train, test])
    X_full, y_full = split_xy(full)
    visualization.plot_target(y_full)
    visualization.plot_segments(X_full, y_full)
    visualization.plot_service_means(X_full, y_full)
    num = AirlinePreprocessor().fit_transform(X_full).assign(target=y_full.values)
    visualization.plot_correlation(num)
    visualization.plot_delays(X_full, y_full)
    visualization.plot_zero_ratings(X_full)
    num.corr()["target"].drop("target").sort_values(ascending=False).to_csv(
        C.MET_DIR / "hedef_korelasyonlari.csv"
    )
    plt.close("all")

    # 3 ----------------------------------------------------------- Modelleme
    step("3) Modelleme (yalnızca eğitim seti kullanılır)")
    fitted = {"Baseline (çoğunluk)": models.baseline(X_train, y_train)}
    cv_rows, best_params = [], {}
    for name in ("Decision Tree", "Random Forest"):
        print(f"\n--> {name}: RandomizedSearchCV (n_iter={C.SEARCH_ITER}, cv={C.SEARCH_CV})")
        search = models.tune(name, X_train, y_train)
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"    en iyi parametreler: {best_params[name]}  ({search.fit_seconds_:.0f} sn)")
        pd.DataFrame(search.cv_results_).to_csv(
            C.MET_DIR / f"arama_{name.replace(' ', '_').lower()}.csv", index=False
        )
        print(f"--> {name}: {C.CV_FOLDS}-katlı çapraz doğrulama")
        rep = models.cv_report(search.best_estimator_, X_train, y_train)
        rep["model"], rep["search_seconds"] = name, search.fit_seconds_
        cv_rows.append(rep)
        fitted[name] = search.best_estimator_
        joblib.dump(
            search.best_estimator_, C.MODEL_DIR / f"{name.replace(' ', '_').lower()}.joblib"
        )
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(C.MET_DIR / "cv_sonuclari.csv", index=False)
    results["en_iyi_parametreler"] = best_params

    print("\n--> Karar ağacı derinlik eğrisi (aşırı öğrenme analizi)")
    curve = models.depth_curve(X_train, y_train)
    curve.to_csv(C.MET_DIR / "derinlik_egrisi.csv", index=False)
    evaluation.plot_depth_curve(curve)

    # 4 ----------------------------------------------------------- Değerlendirme
    step("4) Test seti değerlendirmesi")
    test_df = pd.DataFrame(
        [evaluation.test_metrics(n, p, X_test, y_test) for n, p in fitted.items()]
    )
    test_df.to_csv(C.MET_DIR / "test_sonuclari.csv", index=False)
    print(test_df.round(4).to_string(index=False))
    trained = {k: v for k, v in fitted.items() if "Baseline" not in k}
    evaluation.plot_confusion(trained, X_test, y_test)
    evaluation.plot_roc(fitted, X_test, y_test)
    evaluation.plot_metric_comparison(test_df)
    evaluation.plot_tree_top(fitted["Decision Tree"])
    lit = evaluation.literature_table(test_df)
    lit.to_csv(C.MET_DIR / "literatur_karsilastirma.csv", index=False)
    print("\n", lit.to_string(index=False))
    results["test"] = test_df.round(4).to_dict(orient="records")
    plt.close("all")

    # 5 ----------------------------------------------------------- Açıklanabilirlik
    step("5) Açıklanabilirlik: özellik önemleri + SHAP")
    rf = fitted["Random Forest"]
    imp = explainability.importance_table(rf, X_test, y_test)
    imp.to_csv(C.MET_DIR / "ozellik_onemleri.csv", index=False)
    explainability.plot_importances(imp)
    print(f"SHAP {C.SHAP_SAMPLE} test örneği için hesaplanıyor...")
    sv, Xs = explainability.shap_values(rf, X_test)
    explainability.plot_shap_summary(sv)
    explainability.plot_waterfall(sv, 0)
    aps_rank = explainability.shap_ranking(sv)
    aps_rank.to_csv(C.MET_DIR / "shap_siralama.csv", index=False)
    seg = explainability.segment_shap(sv, Xs, "travel_class")
    seg.to_csv(C.MET_DIR / "shap_kabin_sinifi.csv")
    explainability.plot_segment_shap(seg)
    seg2 = explainability.segment_shap(
        sv,
        Xs,
        "type_of_travel",
        {"Kişisel seyahat": Xs["type_of_travel"] == 0, "İş seyahati": Xs["type_of_travel"] == 1},
    )
    seg2.to_csv(C.MET_DIR / "shap_seyahat_tipi.csv")
    explainability.plot_segment_shap(
        seg2, title="Seyahat tipine göre SHAP önemi", name="15b_shap_seyahat_tipi"
    )
    print(aps_rank.head(8).to_string(index=False))
    results["shap_ilk5"] = aps_rank.head(5)["feature"].tolist()
    plt.close("all")

    # 6 ----------------------------------------------------------- Skytrax
    step("6) Skytrax gerçek yorumlarla dış doğrulama")
    sky, info = skytrax.load_clean()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    sky_models, sky_res, sky_params, (sky_Xte, thy) = skytrax.train_models(sky)
    sky_res.to_csv(C.MET_DIR / "skytrax_sonuclari.csv", index=False)
    print(sky_res.round(4).to_string(index=False))
    sky_rank = skytrax.shap_ranking(sky_models["Random Forest"], sky_Xte)
    sky_rank.to_csv(C.MET_DIR / "skytrax_shap.csv", index=False)
    comp = skytrax.compare_with_aps(sky_rank, aps_rank)
    comp.to_csv(C.MET_DIR / "skytrax_kaggle_faktor_karsilastirma.csv", index=False)
    skytrax.plot_factor_comparison(comp)
    prof = skytrax.thy_profile(sky)
    prof.to_csv(C.MET_DIR / "skytrax_thy_profil.csv")
    skytrax.plot_thy_profile(prof)
    print(comp.round(3).to_string(index=False))
    print(prof.to_string())
    results["skytrax"] = {
        "bilgi": info,
        "parametreler": sky_params,
        "sonuclar": sky_res.round(4).to_dict(orient="records"),
    }
    plt.close("all")

    # 7 ----------------------------------------------------------- Ek analizler
    step("7) Ek analizler: senaryo, hata analizi, McNemar")
    results["ek_analizler"] = extra_analysis.run(
        fitted["Random Forest"], fitted["Decision Tree"], X_test, y_test
    )

    results["toplam_sure_sn"] = round(time.perf_counter() - t_start, 1)
    with open(C.MET_DIR / "ozet.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    step(f"Bitti. Toplam süre: {results['toplam_sure_sn']} sn. Çıktılar: outputs/")


if __name__ == "__main__":
    main()
