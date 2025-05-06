import streamlit as st
import pandas as pd
import numpy as np
import joblib

# تحميل النموذج
data = joblib.load(open('best_diabetes_model.pkl', 'rb'))

st.title("# Health Care App")
st.info('Application for Diabetes Disease Prediction')

st.sidebar.header('Feature Selection')

# --- مداخل المستخدم ---
year = st.text_input('Year')
age = st.text_input('Age')
AA = st.text_input('Race: African American')
A = st.text_input('Race: Asian')
C = st.text_input('Race: Caucasian')
O = st.text_input('Race: Other')
H = st.text_input('Race: Hispanic')
h = st.text_input('Hypertension (0 or 1)')
hd = st.text_input('Heart Disease (0 or 1)')
b = st.text_input('BMI')
hl = st.text_input('HbA1c Level')
bgl = st.text_input('Blood Glucose Level')
lf = st.text_input('Location Frequency')

# اختيار الجنس (بدل الإدخال اليدوي)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])

# تحويل الجنس إلى One-Hot Encoding
gf = 1 if gender == 'Female' else 0
gm = 1 if gender == 'Male' else 0
go = 1 if gender == 'Other' else 0

# دالة لتحويل النص إلى float مع المعالجة
def to_float(val):
    try:
        return float(val)
    except:
        return 0.0

# حساب المشتقات المطلوبة
age_val = to_float(age)
bmi_val = to_float(b)
hypertension_val = to_float(h)
heart_disease_val = to_float(hd)

age_bmi = age_val * bmi_val
comorbidity_count = hypertension_val + heart_disease_val

# إنشاء DataFrame
df = pd.DataFrame({
    'year': [to_float(year)],
    'age': [age_val],
    'race:AfricanAmerican': [to_float(AA)],
    'race:Asian': [to_float(A)],
    'race:Caucasian': [to_float(C)],
    'race:Hispanic': [to_float(H)],
    'race:Other': [to_float(O)],
    'hypertension': [hypertension_val],
    'heart_disease': [heart_disease_val],
    'bmi': [bmi_val],
    'hbA1c_level': [to_float(hl)],
    'blood_glucose_level': [to_float(bgl)],
    'gender_Female': [gf],
    'gender_Male': [gm],
    'gender_Other': [go],
    'loc_freq': [to_float(lf)],
    'age_bmi': [age_bmi],
    'comorbidity_count': [comorbidity_count]
})

# زر التأكيد
con = st.sidebar.button('Confirm')
if con:
    res = data.predict(df)
    if res[0] == 0:
        st.sidebar.success('THE Patient is Healthy ✅')
    else:
        st.sidebar.error('THE Patient has Diabetes ⚠️')
