"""
Run ONCE from the project root:   python export_notebook_plots.py
Extracts the plots already embedded in your notebooks (their saved outputs)
and writes them as PNG files into visualizations/<track>/.
"""
import base64, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "visualizations"

# (notebook file, cell index, image number inside that cell) -> (track folder, output file name)
PLOTS = {
    # ---------------- REGRESSION ----------------
    ("eda.ipynb", 13, 0): ("regression", "eda_cancellation_distribution.png"),
    ("eda.ipynb", 14, 0): ("regression", "eda_numeric_histograms.png"),
    ("eda.ipynb", 16, 0): ("regression", "eda_correlation_heatmap.png"),
    ("eda.ipynb", 18, 0): ("regression", "eda_leadtime_vs_adr.png"),
    ("eda.ipynb", 20, 0): ("regression", "eda_total_nights_vs_adr.png"),
    ("regression_linear_models.ipynb", 1, 0): ("regression", "lr_top_coefficients.png"),
    ("regression_linear_models.ipynb", 2, 0): ("regression", "ridge_alpha_vs_performance.png"),
    ("regression_linear_models.ipynb", 3, 0): ("regression", "lasso_coefficients.png"),
    ("regression_linear_models.ipynb", 4, 0): ("regression", "elasticnet_l1_ratio.png"),
    ("regression_linear_models.ipynb", 5, 0): ("regression", "polynomial_degree.png"),
    ("regression_linear_models.ipynb", 8, 0): ("regression", "linear_models_comparison.png"),
    ("regression_linear_models.ipynb", 9, 0): ("regression", "best_actual_vs_predicted.png"),
    ("Hotel_ADR_Regression_Models_6to10 (1).ipynb", 3, 0): ("regression", "dt_feature_importance.png"),
    ("Hotel_ADR_Regression_Models_6to10 (1).ipynb", 13, 0): ("regression", "tree_models_comparison.png"),
    ("Hotel_ADR_Regression_Models_6to10 (1).ipynb", 13, 1): ("regression", "tree_models_cv_spread.png"),

    # ---------------- CLASSIFICATION ----------------
    ("preprocessing_class.ipynb", 6, 0): ("classification", "eda_class_distribution.png"),
    ("preprocessing_class.ipynb", 8, 0): ("classification", "eda_speed_by_class.png"),
    ("preprocessing_class.ipynb", 9, 0): ("classification", "eda_queue_by_class.png"),
    ("preprocessing_class.ipynb", 10, 0): ("classification", "eda_speed_vs_density.png"),
    ("preprocessing_class.ipynb", 11, 0): ("classification", "eda_correlation_heatmap.png"),
    ("classification.ipynb", 5, 0): ("classification", "logreg_coefficients.png"),
    ("classification.ipynb", 6, 0): ("classification", "knn_accuracy_vs_k.png"),
    ("classification.ipynb", 11, 0): ("classification", "decision_tree.png"),
    ("classification.ipynb", 12, 0): ("classification", "dt_feature_importance.png"),
    ("classification.ipynb", 22, 0): ("classification", "rf_feature_importance.png"),
    ("classification.ipynb", 28, 0): ("classification", "confusion_matrices_all.png"),
    ("classification.ipynb", 29, 0): ("classification", "metrics_comparison_all.png"),

    # ---------------- CLUSTERING ----------------
    ("preprocessing_clustering.ipynb", 12, 0): ("clustering", "features_before_transform.png"),
    ("preprocessing_clustering.ipynb", 15, 0): ("clustering", "features_after_transform.png"),
    ("preprocessing_clustering.ipynb", 16, 0): ("clustering", "correlation_matrix.png"),
    ("clustering.ipynb", 7, 0): ("clustering", "kmeans_elbow.png"),
    ("clustering.ipynb", 24, 0): ("clustering", "kmeans_silhouette_vs_k.png"),
    ("clustering.ipynb", 13, 0): ("clustering", "ward_dendrogram.png"),
    ("clustering.ipynb", 17, 0): ("clustering", "pca_cluster_plots.png"),
    ("clustering.ipynb", 19, 0): ("clustering", "tsne_kmeans.png"),
    ("clustering.ipynb", 22, 0): ("clustering", "evaluation_metrics.png"),
    ("clustering.ipynb", 25, 0): ("clustering", "cluster_sizes.png"),
}


def main():
    saved, missed = 0, []
    cache = {}
    for (nb_name, cell_idx, img_no), (track, fname) in PLOTS.items():
        nb_path = NB_DIR / nb_name
        if not nb_path.exists():
            missed.append(f"{nb_name} (notebook not found)")
            continue
        if nb_name not in cache:
            cache[nb_name] = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = cache[nb_name]["cells"]
        try:
            outputs = cells[cell_idx].get("outputs", [])
            pngs = [o["data"]["image/png"] for o in outputs
                    if "data" in o and "image/png" in o["data"]]
            b64 = pngs[img_no]
            if isinstance(b64, list):
                b64 = "".join(b64)
            target = OUT_DIR / track
            target.mkdir(parents=True, exist_ok=True)
            (target / fname).write_bytes(base64.b64decode(b64))
            saved += 1
        except (IndexError, KeyError):
            missed.append(f"{nb_name} cell {cell_idx} image {img_no} (no saved plot - re-run that cell)")
    print(f"Saved {saved} plots into {OUT_DIR}")
    for m in missed:
        print("  MISSING:", m)


if __name__ == "__main__":
    main()