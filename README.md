# ✈️ Flight Delay Prediction (Machine Learning Project)

This data science project analyzes the 2015 US Flight Delays dataset to build a machine learning model that predicts whether a flight will be delayed by 15 minutes or more.

The entire analysis is conducted in a Python notebook and focuses heavily on **feature engineering**, **data visualization**, and handling **imbalanced data** to build a reliable and realistic model.

---

## 🎯 Problem Statement

Flight delays cost airlines and passengers billions of dollars annually. While airlines have access to vast amounts of data, it's often raw and difficult to interpret. The key challenge is to sift through millions of flight records to find patterns that predict *future* delays.

This project addresses two critical data science problems:
1.  **A Classification Problem:** Can we predict *if* a flight will be delayed (e.g., by more than 15 minutes) based on factors known at scheduling time (airline, route, time of day)?
2.  **An Imbalance Problem:** Most flights (80-90%) are *on-time*. A naive model can achieve 90% accuracy by simply guessing "on-time" every time. This is useless for an airline.

This project solves this by:
* **Feature Engineering:** Converting raw data like scheduled times (`1305`) and airport codes (`'JFK'`) into meaningful features (e.g., `DEPARTURE_HOUR`).
* **Handling Imbalance:** Using a `RandomForestClassifier` with `class_weight='balanced'` to force the model to pay special attention to the rare, but important, "delayed" cases.
* **Actionable Insights:** Using feature importance to identify the *biggest* drivers of delays (e.g., specific airlines, times of day, or airports).

---

## 🛠️ Tech Stack & Tools Used

* **Language:** **Python 3**
* **Data Manipulation:** **Pandas** (for data loading and cleaning) & **NumPy** (for numerical operations)
* **Data Visualization:** **Matplotlib** & **Seaborn** (for EDA and plotting results)
* **Machine Learning:** **Scikit-learn (sklearn)**
    * `RandomForestClassifier` (the core model)
    * `train_test_split` (for splitting data)
    * `LabelEncoder` (for feature engineering)
    * `classification_report` & `confusion_matrix` (for model evaluation)
* **Environment:** **Google Colab** (Jupyter Notebook)

---

## 🚀 Key Features & Analysis
* **EDA:** Analyzed and visualized the percentage of delays by airline to identify top and bottom performers.
* **Target Definition:** Created a binary target variable `IS_DELAYED` (1 if `DEPARTURE_DELAY` > 15 mins, 0 otherwise).
* **Feature Engineering:** Extracted `DEPARTURE_HOUR` from `SCHEDULED_DEPARTURE` time, as the hour of the day is a more powerful predictor than the specific minute.
* **Model Building:** Trained a Random Forest model, specifically using `class_weight='balanced'` to solve the class imbalance problem.
* **Model Evaluation:** Instead of relying on simple "accuracy," the model's performance is judged by its **Classification Report (Precision, Recall, F1-Score)** and a **Confusion Matrix** to see how well it catches *actual* delays.
* **Feature Importance:** Plotted the model's feature importances to discover the most significant drivers of flight delays (e.g., departure hour, airline, origin airport).

---

## 🏃 How to Run This Project

1.  **Clone the repository** (or download the `.ipynb` file).
2.  **Get the data:** Download the "2015 Flight Delays and Cancellations" dataset from [this Kaggle link](https://www.kaggle.com/datasets/usdot/flight-delays).
3.  **Upload to Colab:** Open the `.ipynb` file in Google Colab. Upload the `flights.csv` and `airlines.csv` files to your Colab session.
4.  **Run the cells:** Execute the notebook cells sequentially from top to bottom.
