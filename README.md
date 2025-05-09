# 🩺 Glucose Guard – Diabetes Risk Prediction Web App

**Glucose Guard** is a machine learning-based healthcare application designed to predict the risk of type 2 diabetes using patient health metrics.  
This project was developed using Python, Streamlit, and scikit-learn as part of a graduation project for the DEPI initiative supported by the Egyptian Ministry of Communications and Information Technology.

---

## 📌 Project Overview

This project leverages real-world diabetes data and machine learning algorithms to:

- Perform data exploration and visualization.
- Build and evaluate prediction models.
- Deploy an interactive web app for real-time diabetes risk prediction.

The app provides doctors and users with an intuitive interface to input patient data and receive instant feedback with visual analytics.

---

## 🚀 App Features

- Developed using **Streamlit** for real-time prediction.
- Trained with **Random Forest** and other ML models.
- Input data includes age, BMI, HbA1c, glucose level, comorbidities, etc.
- Live prediction with probability and confidence metrics.
- Feature importance visualization using Plotly.
- Downloadable prediction report.

---

## 📂 Repository Structure

```

📁 project/
│
├── app.py                             # Streamlit web app
├── Diabetes Dataset Eda.ipynb         # EDA notebook (Exploratory Data Analysis)
├── Diabetes Dataset Prediction-1.ipynb  # ML modeling notebook
├── diabetes_dataset.csv               # Original dataset
├── Diabetes_model.pkl                 # Trained Random Forest model
├── scaler.pkl                         # Preprocessing scaler
├── selector.pkl                       # Feature selector
├── requirements.txt                   # Python dependencies

````

---

## 💡 How to Run the App

1. **Install dependencies**:

```bash
pip install -r requirements.txt
````

2. **Run the Streamlit app**:

```bash
streamlit run app.py
```

3. **Open in browser**:

Go to `http://localhost:8501`

---

## 🧪 Sample Inputs Used in the App

* Age
* BMI
* HbA1c Level
* Blood Glucose Level
* Gender
* Race
* Hypertension / Heart Disease
* Location Frequency
* Derived metrics: Age \* BMI, Comorbidity Count

---

## 🛠 Technologies Used

* Python
* Streamlit
* scikit-learn
* pandas, numpy
* Plotly
* Joblib

---

## 👥 Team Members

* Esraa Mostafa El Tohamy
* Abanoup Emad Masry
* Alaa El Said El Hadidy
* Ahmed Osama El Ghareeb
