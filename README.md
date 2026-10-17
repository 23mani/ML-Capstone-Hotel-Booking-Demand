# Machine Learning Capstone Project
## Hotel Booking Demand, Traffic Congestion Classification, and Customer Segmentation

**Course:** 23CSE301 – Machine Learning Capstone Project

## 1. Project Overview

This project applies machine learning techniques to three real-world problems: predicting hotel booking prices, classifying road traffic congestion, and identifying customer segments from online retail data.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, performance evaluation, visualization, and an interactive web application built using Streamlit.

The application brings all three machine learning tracks together in a single interface, allowing users to explore predictions, classification results, and customer clusters.

## 2. Project Objectives

- Analyze real-world datasets using exploratory data analysis.
- Clean and preprocess data to improve its quality and suitability for machine learning.
- Implement and compare multiple machine learning algorithms.
- Evaluate models using appropriate performance metrics.
- Apply dimensionality reduction and visualization techniques to understand customer clusters.
- Develop an interactive Streamlit application that integrates the trained models.
- Explore how machine learning can support decision-making in hospitality, intelligent transportation, and retail.

## 3. Project Modules

### 3.1 Hotel Booking ADR Regression

**Objective:** Predict the Average Daily Rate (ADR) of a hotel booking using relevant booking information.

**Dataset:** `hotel_bookings.csv`

The regression track analyzes hotel booking data to understand the factors associated with room pricing.

Key activities include:
- Data cleaning and preprocessing.
- Exploratory analysis of booking characteristics and ADR.
- Feature selection and preparation.
- Training and comparison of linear regression models and tree-based or ensemble regression models.
- Evaluation using regression metrics.

**Target variable:** `adr` (Average Daily Rate).

**Evaluation metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and coefficient of determination (\(R^2\)), where applicable.

### 3.2 VANET Traffic Congestion Classification

**Objective:** Classify traffic congestion levels using traffic-related features.

**Dataset:** `vanet_traffic_data.csv`

This module explores traffic conditions in a Vehicular Ad Hoc Network (VANET) context. It uses traffic characteristics to classify congestion levels and investigates how vehicle speed, density, queue length, and other available features relate to traffic conditions.

Key activities include:
- Data cleaning and preprocessing.
- Analysis of traffic features and class distributions.
- Feature preparation and model training.
- Comparison of multiple classification algorithms.
- Confusion matrix visualization and performance comparison.

The classification track includes models such as Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naive Bayes, AdaBoost, Bagging, Gradient Boosting, and Multi-Layer Perceptron (MLP).

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, and confusion matrix analysis.

The final model should be selected based on its evaluation results and suitability for the classification task.

### 3.3 Online Retail Customer Clustering

**Objective:** Group customers with similar purchasing characteristics using unsupervised machine learning.

**Dataset:** `online_retail.csv`

Customer clustering helps identify groups of customers with similar behavioral patterns without requiring predefined class labels.

Key activities include:
- Data cleaning and preparation of retail transaction records.
- Customer-level feature engineering.
- Feature scaling and transformation.
- Application of clustering algorithms.
- Selection of suitable cluster counts.
- Evaluation and visualization of the resulting clusters.

The project includes clustering analysis and visualizations such as the K-Means elbow curve, silhouette score comparisons, cluster-size plots, PCA plots, t-SNE visualizations, and a hierarchical clustering dendrogram.

**Evaluation metrics:** Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Index, where applicable.

These metrics help assess how well-separated and compact the resulting clusters are.

## 4. Technologies Used

- **Programming language:** Python
- **Data manipulation:** Pandas, NumPy
- **Data visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Development environment:** Jupyter Notebook
- **Web application:** Streamlit

## 5. Project Structure

```text
ML-Capstone-Hotel-Booking-Demand/
│
├── app.py
├── export_notebook_plots.py
├── requirements.txt
│
├── data/
│   ├── hotel_bookings.csv
│   ├── hotel_bookings_cleaned.csv
│   ├── vanet_traffic_data.csv
│   ├── vanet_traffic_data_cleaned.csv
│   ├── online_retail.csv
│   ├── online_retail_cleaned.csv
│   ├── online_retail_customer_features.csv
│   └── online_retail_customer_features_scaled.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── regression_linear_models.ipynb
│   ├── Hotel_ADR_Regression_Models_6to10 (1).ipynb
│   ├── preprocessing_class.ipynb
│   ├── classification.ipynb
│   ├── preprocessing_clustering.ipynb
│   └── clustering.ipynb
│
├── models/
│   ├── Saved regression models
│   ├── Saved classification models
│   ├── Saved clustering models
│   └── Preprocessing objects and evaluation results
│
└── visualizations/
    ├── regression/
    ├── classification/
    └── clustering/
```

*Note: The structure above highlights the main project files. Individual saved model filenames and additional files may vary.*

## 6. Installation and Setup

### Prerequisites

Install Python and ensure that pip is available. Jupyter Notebook is recommended for running the analysis notebooks.

### Step 1: Clone the repository

```bash
git clone https://github.com/23mani/ML-Capstone-Hotel-Booking-Demand
cd ML-Capstone-Hotel-Booking-Demand
```

Replace the placeholder with the actual GitHub repository URL.

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Jupyter Notebook

```bash
jupyter notebook
```

Open the relevant notebooks in the `notebooks/` directory to explore the data, perform preprocessing, train models, and evaluate their performance.

## 7. Running the Streamlit Application

The Streamlit application provides a unified interface for the three machine learning tracks.

Before starting the application, ensure that the required datasets, saved model files, preprocessing objects, and visualization files are present in their respective directories.

Run the following command from the project root directory:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a web browser to interact with the application.

If a required model file is missing, run the corresponding notebook's model-saving cells before launching the application.

## 8. Application Features

The application provides three main sections:

- **Regression — Hotel ADR:** Explore hotel ADR prediction using the trained regression models.
- **Classification — Traffic Congestion:** Explore traffic congestion classification using the trained classification models.
- **Clustering — Retail Customers:** Examine customer segments and clustering evaluation results.

The application also displays saved notebook visualizations to support the interpretation of the machine learning results.

## 9. Expected Outcomes

- Understand the factors associated with hotel booking prices and predict ADR using regression.
- Compare classification models for identifying traffic congestion levels.
- Discover customer segments based on online retail purchasing patterns.
- Compare models using suitable evaluation metrics instead of relying on a single measure.
- Present the results through an interactive dashboard and supporting visualizations.

Actual model performance depends on the dataset, preprocessing choices, feature selection, and experimental results.

## 10. Future Enhancements

- Improve model performance through hyperparameter tuning and cross-validation.
- Add more advanced regression, classification, and clustering algorithms.
- Improve the Streamlit interface with additional interactive charts and model comparisons.
- Incorporate new or updated datasets for further analysis.
- Explore deployment options to make the application accessible online.
- Provide clearer interpretations of predictions, classification outcomes, and customer segments.

## 11. Conclusion

This Machine Learning Capstone Project demonstrates how supervised and unsupervised learning can address different real-world problems within one integrated application. Regression is used for numerical prediction, classification for assigning traffic congestion categories, and clustering for discovering customer groups.

The combination of Jupyter Notebook experimentation, machine learning evaluation, visualizations, and Streamlit integration provides an end-to-end workflow for developing and presenting machine learning solutions.

## 12. Contributors

| S. No. | Contributor Name | Roll Number |
|---|---|---|
| 1 | Armitha J | CB.SC.U4CSE24504 |
| 2 | K J JayaShree | CB.SC.U4CSE24515 |
| 3 | Mani Nila S | CB.SC.U4CSE24526 |

---

**Course:** 23CSE301 – Machine Learning Capstone Project  
**Project Title:** Hotel Booking ADR Regression, VANET Traffic Congestion Classification, and Online Retail Customer Clustering
