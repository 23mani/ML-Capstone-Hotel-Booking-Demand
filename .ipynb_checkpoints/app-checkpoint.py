from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ---------------------------------------------------------
# App setup and paths
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone | Three Tracks",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path)


@st.cache_resource
def load_pickle(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required model file not found: {path}. "
            "Run the corresponding notebook's model-saving cell first."
        )
    return joblib.load(path)


def numeric_input(label, value, min_value=None, max_value=None, step=None, key=None):
    """Create a sensible numeric input based on the selected dataset row."""
    value = float(value) if pd.notna(value) else 0.0
    if min_value is None:
        min_value = min(0.0, value)
    if max_value is None:
        max_value = max(100.0, value)
    if min_value == max_value:
        max_value = min_value + 1.0
    if step is None:
        step = 1.0 if float(value).is_integer() else 0.1
    if float(value).is_integer() and float(min_value).is_integer() and float(max_value).is_integer():
        return st.number_input(
            label, min_value=int(min_value), max_value=int(max_value),
            value=int(np.clip(round(value), min_value, max_value)),
            step=max(1, int(step)), key=key
        )
    return st.number_input(
        label, min_value=float(min_value), max_value=float(max_value),
        value=float(np.clip(value, min_value, max_value)),
        step=float(step), key=key, format="%.3f"
    )

VIS_DIR = ROOT / "visualizations"


def show_plots(track_folder, plots, columns=1):
    """Show saved notebook plots. plots = [(file_name, caption), ...]"""
    folder = VIS_DIR / track_folder
    cols = st.columns(columns) if columns > 1 else [st.container()]
    for i, (fname, caption) in enumerate(plots):
        path = folder / fname
        with cols[i % len(cols)]:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.warning(f"Plot not found: {path.name}. Run export_notebook_plots.py first.")


def plots_section(track_folder, tabs):
    """tabs = {"Tab title": [(file, caption), ...], ...}"""
    st.divider()
    st.subheader("📈 Notebook visualizations")
    for tab, (title, plots) in zip(st.tabs(list(tabs.keys())), tabs.items()):
        with tab:
            show_plots(track_folder, plots)


st.title("📊 Machine Learning Capstone")
st.caption("Hotel Booking ADR Regression • VANET Traffic Classification • Online Retail Customer Clustering")
st.info(
    "This app loads the models saved by the project notebooks. "
    "Run the model-saving cells and keep the data/ and models/ folders in the project root."
)

track = st.sidebar.radio(
    "Choose a track",
    ["🏨 Regression — Hotel ADR", "🚦 Classification — Traffic Congestion", "🛍️ Clustering — Retail Customers"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Models are loaded locally from this project's models/ folder.")

# ---------------------------------------------------------
# TRACK 1: REGRESSION
# ---------------------------------------------------------
if track == "🏨 Regression — Hotel ADR":
    st.header("🏨 Hotel Booking — Average Daily Rate (ADR)")
    st.write(
        "Estimate a booking's ADR using the trained regression models. "
        "Choose a record from the cleaned dataset as a starting point, then adjust common booking details."
    )

    try:
        data = load_csv("hotel_bookings_cleaned.csv")
        models_a = load_pickle("regression_linear_5.pkl")
        models_b = load_pickle("regression_tree_ensemble_6_to_10.pkl")
        models = {**models_a, **models_b}
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    target_col = "adr"
    excluded = [c for c in [target_col, "reservation_status", "reservation_status_date", "is_canceled"] if c in data.columns]
    feature_cols = [c for c in data.columns if c not in excluded]

    record_index = st.selectbox(
        "Starting booking record (row number)",
        options=list(range(min(len(data), 5000))),
        format_func=lambda x: f"Record {x + 1}",
        help="The model was trained on cleaned booking records. This selects a real row as a starting point.",
    )
    row = data.iloc[record_index]
    input_df = pd.DataFrame([{c: row[c] for c in feature_cols}])

    # Editable, high-impact fields. All other fields remain from the selected record.
    with st.expander("Adjust booking details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            if "hotel" in feature_cols:
                options = sorted(data["hotel"].dropna().astype(str).unique().tolist())
                current = str(row["hotel"])
                input_df.loc[0, "hotel"] = st.selectbox("Hotel type", options, index=options.index(current) if current in options else 0)
            if "lead_time" in feature_cols:
                input_df.loc[0, "lead_time"] = numeric_input("Lead time (days)", row["lead_time"], 0, max(3650, int(data["lead_time"].max())), 1, "reg_lead")
            if "adults" in feature_cols:
                input_df.loc[0, "adults"] = numeric_input("Adults", row["adults"], 0, 20, 1, "reg_adults")
            if "children" in feature_cols:
                input_df.loc[0, "children"] = numeric_input("Children", row["children"], 0, 20, 1, "reg_children")
        with col2:
            if "arrival_date_month" in feature_cols:
                opts = data["arrival_date_month"].dropna().astype(str).unique().tolist()
                cur = str(row["arrival_date_month"])
                input_df.loc[0, "arrival_date_month"] = st.selectbox("Arrival month", opts, index=opts.index(cur) if cur in opts else 0)
            if "stays_in_weekend_nights" in feature_cols:
                input_df.loc[0, "stays_in_weekend_nights"] = numeric_input("Weekend nights", row["stays_in_weekend_nights"], 0, 30, 1, "reg_weekend")
            if "stays_in_week_nights" in feature_cols:
                input_df.loc[0, "stays_in_week_nights"] = numeric_input("Week nights", row["stays_in_week_nights"], 0, 60, 1, "reg_weeknights")
            if "total_guests" in feature_cols:
                input_df.loc[0, "total_guests"] = numeric_input("Total guests", row["total_guests"], 0, 50, 1, "reg_guests")
        with col3:
            for col_name, label, max_val in [
                ("market_segment", "Market segment", None),
                ("deposit_type", "Deposit type", None),
                ("customer_type", "Customer type", None),
            ]:
                if col_name in feature_cols:
                    opts = data[col_name].dropna().astype(str).unique().tolist()
                    cur = str(row[col_name])
                    input_df.loc[0, col_name] = st.selectbox(
                        label, opts, index=opts.index(cur) if cur in opts else 0, key=f"reg_{col_name}"
                    )
            if "total_nights" in feature_cols:
                input_df.loc[0, "total_nights"] = numeric_input("Total nights", row["total_nights"], 0, 90, 1, "reg_total_nights")

    st.caption("Other model features retain the values from the selected starting record.")
    model_name = st.selectbox("Regression model", list(models.keys()))
    if st.button("Predict hotel ADR", type="primary"):
        try:
            model = models[model_name]
            # Preserve exact training feature order.
            prediction_input = input_df.reindex(columns=feature_cols)
            prediction = float(model.predict(prediction_input)[0])
            st.success(f"Predicted ADR: {prediction:,.2f}")
            st.caption("ADR is shown in the same currency/unit used by the source dataset.")
            with st.expander("View the features sent to the model"):
                st.dataframe(prediction_input.T.rename(columns={0: "Input value"}), use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.write("Check that both regression model files were saved from the current notebooks and that the dataset columns match the training data.")

    plots_section("regression", {
        "EDA": [
            ("eda_cancellation_distribution.png", "Booking cancellation distribution"),
            ("eda_numeric_histograms.png", "Numeric feature distributions"),
            ("eda_correlation_heatmap.png", "Correlation heatmap"),
            ("eda_leadtime_vs_adr.png", "Lead time vs ADR"),
            ("eda_total_nights_vs_adr.png", "Total nights vs ADR"),
        ],
        "Linear models (1-5)": [
            ("lr_top_coefficients.png", "Linear Regression - top coefficients"),
            ("ridge_alpha_vs_performance.png", "Ridge - alpha vs performance"),
            ("lasso_coefficients.png", "Lasso - coefficients and sparsity"),
            ("elasticnet_l1_ratio.png", "ElasticNet - l1_ratio vs performance"),
            ("polynomial_degree.png", "Polynomial - degree vs R2 / RMSE"),
            ("linear_models_comparison.png", "R2 / RMSE / MAE comparison"),
            ("best_actual_vs_predicted.png", "Best model - actual vs predicted & residuals"),
        ],
        "Tree / ensemble models (6-10)": [
            ("dt_feature_importance.png", "Decision Tree - top 10 feature importances"),
            ("tree_models_comparison.png", "Algorithm comparison (R2)"),
            ("tree_models_cv_spread.png", "5-fold CV spread"),
        ],
    })

# ---------------------------------------------------------
# TRACK 2: CLASSIFICATION
# ---------------------------------------------------------
elif track == "🚦 Classification — Traffic Congestion":
    st.header("🚦 VANET Traffic Congestion Classification")
    st.write("Enter road, weather, and communication-network measurements to predict the traffic class.")

    try:
        data = load_csv("vanet_traffic_data_cleaned.csv")
        scaler = load_pickle("scaler.pkl")
        encoder = load_pickle("label_encoder.pkl")
        model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "K-Nearest Neighbors": "knn.pkl",
            "Gaussian Naive Bayes": "naive_bayes.pkl",
            "Decision Tree": "decision_tree.pkl",
            "Support Vector Machine": "svm.pkl",
            "Random Forest": "random_forest.pkl",
            "AdaBoost": "adaboost.pkl",
            "Gradient Boosting": "gradient_boosting.pkl",
            "Bagging": "bagging.pkl",
            "MLP Neural Network": "mlp_classifier.pkl",
        }
        available_models = {}
        missing_models = []
        for display_name, filename in model_files.items():
            try:
                available_models[display_name] = load_pickle(filename)
            except FileNotFoundError:
                missing_models.append(filename)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    feature_cols = [c for c in data.columns if c != "label"]
    if len(feature_cols) != getattr(scaler, "n_features_in_", len(feature_cols)):
        st.error(
            f"Feature mismatch: the saved scaler expects {getattr(scaler, 'n_features_in_', 'unknown')} "
            f"features, but the dataset provides {len(feature_cols)}. Re-run preprocessing/training and save the scaler again."
        )
        st.stop()

    sample_idx = st.selectbox(
        "Starting traffic record",
        options=list(range(min(len(data), 10000))),
        format_func=lambda x: f"Record {x + 1}",
        help="Use an existing row as a starting point, or adjust the inputs below.",
    )
    row = data.iloc[sample_idx]
    defaults = {c: row[c] for c in feature_cols}
    overrides = {}
    groups = [
        ("Road traffic", ["avg_speed_kmph", "density_veh_per_km", "avg_wait_time_s", "occupancy_pct", "flow_veh_per_hr", "queue_length_veh"]),
        ("Vehicle / road conditions", ["avg_accel_ms2", "heading_deg", "signal_state_num", "incident_num", "hour_of_day", "day_of_week", "is_rush_hour"]),
        ("Weather", ["temp_c", "visibility_km", "rain_intensity_mmph", "weather_factor"]),
        ("VANET communication", ["channel_busy_ratio_pct", "msg_rate_hz", "avg_comm_delay_ms", "rssi_dbm", "packet_loss_pct", "network_reliability_score"]),
        ("Engineered features", ["speed_density_ratio", "congestion_pressure", "wireless_congestion_intensity", "throughput_per_queued_vehicle", "acceleration_directionality"]),
    ]
    with st.expander("Edit input measurements", expanded=True):
        for group_name, cols in groups:
            valid = [c for c in cols if c in feature_cols]
            if not valid:
                continue
            st.markdown(f"**{group_name}**")
            grid = st.columns(3)
            for j, c in enumerate(valid):
                series = pd.to_numeric(data[c], errors="coerce").dropna()
                lo = float(series.min()) if not series.empty else 0.0
                hi = float(series.max()) if not series.empty else max(float(defaults[c]), 1.0)
                if lo == hi:
                    lo -= 1.0
                    hi += 1.0
                val = float(defaults[c]) if pd.notna(defaults[c]) else 0.0
                with grid[j % 3]:
                    overrides[c] = numeric_input(
                        c.replace("_", " ").title(), val, lo, hi,
                        1 if float(val).is_integer() else max((hi - lo) / 100, 0.01),
                        key=f"class_{c}",
                    )

    model_name = st.selectbox("Classification model", list(available_models.keys()) or ["No models found"])
    if missing_models:
        st.warning("Some model files are missing: " + ", ".join(missing_models))
    if st.button("Predict traffic class", type="primary", disabled=not available_models):
        try:
            input_row = pd.DataFrame([{c: overrides.get(c, defaults[c]) for c in feature_cols}], columns=feature_cols)
            X_scaled = scaler.transform(input_row)
            model = available_models[model_name]
            predicted_code = int(model.predict(X_scaled)[0])
            predicted_label = encoder.inverse_transform([predicted_code])[0]
            st.success(f"Predicted traffic class: {predicted_label}")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_scaled)[0]
                classes = getattr(model, "classes_", np.arange(len(probabilities)))
                labels = encoder.inverse_transform(np.asarray(classes, dtype=int))
                prob_df = pd.DataFrame({"Traffic class": labels, "Probability": probabilities})
                prob_df = prob_df.sort_values("Probability", ascending=False)
                st.subheader("Class probabilities")
                st.bar_chart(prob_df.set_index("Traffic class"))
            st.caption("The scaler and label encoder saved by the classification notebook are used automatically.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.write("Ensure every saved classifier was trained using the same feature order and the current scaler.")

    plots_section("classification", {
        "EDA": [
            ("eda_class_distribution.png", "Traffic class distribution"),
            ("eda_speed_by_class.png", "Average speed by class"),
            ("eda_queue_by_class.png", "Queue length by class"),
            ("eda_speed_vs_density.png", "Speed vs density"),
            ("eda_correlation_heatmap.png", "Correlation heatmap"),
        ],
        "Model insights": [
            ("logreg_coefficients.png", "Logistic Regression coefficients per class"),
            ("knn_accuracy_vs_k.png", "KNN accuracy vs k"),
            ("decision_tree.png", "Decision tree (first 3 levels)"),
            ("dt_feature_importance.png", "Decision Tree - feature importance"),
            ("rf_feature_importance.png", "Random Forest - feature importance"),
        ],
        "Model comparison": [
            ("confusion_matrices_all.png", "Confusion matrices - all 10 models"),
            ("metrics_comparison_all.png", "Metric comparison - all 10 models"),
        ],
    })
    results_path = MODEL_DIR / "classification_results.csv"
    if results_path.exists():
        st.subheader("Results table")
        st.dataframe(pd.read_csv(results_path), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TRACK 3: CLUSTERING
# ---------------------------------------------------------
else:
    st.header("🛍️ Online Retail Customer Clustering")
    st.write(
        "Explore customer segments produced by K-Means and Agglomerative clustering. "
        "The clustering notebook trains on the scaled customer-feature dataset."
    )
    try:
        data = load_csv("online_retail_customer_features_scaled.csv")
        model_bundle = load_pickle("clustering_models.pkl")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    id_col = next((c for c in data.columns if c.strip().lower().replace("_", " ") in ["customer id", "customerid", "customer"]), None)
    feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != id_col]
    if not feature_cols:
        st.error("No numeric customer features were found in the scaled dataset.")
        st.stop()

    if id_col:
        ids = data[id_col].astype(str).tolist()
        selected = st.selectbox(
            "Choose a customer record",
            options=list(range(len(data))),
            format_func=lambda i: f"Customer {ids[i]} (row {i + 1})",
        )
    else:
        selected = st.selectbox("Choose a customer record", options=list(range(len(data))))

    customer = data.iloc[selected]
    st.markdown("**Customer features**")
    st.dataframe(
        pd.DataFrame({"Feature": feature_cols, "Value": [customer[c] for c in feature_cols]}),
        use_container_width=True,
        hide_index=True,
    )
    algo = st.selectbox("Clustering model", list(model_bundle.keys()))
    if st.button("Assign / inspect customer cluster", type="primary"):
        try:
            X_all = data[feature_cols].replace([np.inf, -np.inf], np.nan)
            X_all = X_all.fillna(X_all.median())
            model = model_bundle[algo]
            if hasattr(model, "predict"):
                labels = model.predict(X_all)
                selected_label = int(labels[selected])
                st.success(f"Selected customer cluster: {selected_label}")
                st.caption("Cluster numbers are identifiers, not rankings; cluster 2 is not necessarily better than cluster 1.")
            else:
                # AgglomerativeClustering has no predict() method for new rows.
                labels = getattr(model, "labels_", None)
                if labels is None or len(labels) != len(X_all):
                    st.error("This saved hierarchical model does not contain labels aligned with the current dataset. Re-run the clustering notebook and save the models.")
                    st.stop()
                selected_label = int(labels[selected])
                st.success(f"Selected customer cluster: {selected_label}")
                st.caption("Agglomerative clustering assigns labels to the training dataset; it does not provide predict() for unseen customers.")

            # Show segment sizes using labels aligned to the currently loaded data.
            if len(labels) == len(X_all):
                counts = pd.Series(labels).value_counts().sort_index()
                st.subheader("Cluster sizes")
                st.bar_chart(counts.rename("Customers"))
                st.metric("Customers in this cluster", int(counts.loc[selected_label]))
        except Exception as exc:
            st.error(f"Could not inspect cluster: {exc}")

    st.divider()
    st.subheader("Explore the clustering dataset")
    st.write(f"Dataset rows: {len(data):,} • Customer features: {len(feature_cols)}")
    st.dataframe(data.head(20), use_container_width=True)

    plots_section("clustering", {
        "Preprocessing": [
            ("features_before_transform.png", "Feature distributions BEFORE transformation"),
            ("features_after_transform.png", "Feature distributions AFTER capping + log"),
            ("correlation_matrix.png", "Correlation matrix"),
        ],
        "Choosing k": [
            ("kmeans_elbow.png", "Elbow method"),
            ("kmeans_silhouette_vs_k.png", "Silhouette score vs k"),
            ("ward_dendrogram.png", "Ward dendrogram"),
        ],
        "Cluster views": [
            ("pca_cluster_plots.png", "PCA cluster plots"),
            ("tsne_kmeans.png", "t-SNE of K-Means clusters"),
            ("cluster_sizes.png", "Customers per cluster"),
        ],
        "Evaluation": [
            ("evaluation_metrics.png", "Silhouette / Davies-Bouldin / Calinski-Harabasz"),
        ],
    })

st.markdown("---")
st.caption("Machine Learning Capstone • Regression | Classification | Clustering")
