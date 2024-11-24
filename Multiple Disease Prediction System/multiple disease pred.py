import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier


# loading the models
diabetes_model = joblib.load('D:/Multiple Disease Prediction System/Multiple Disease Prediction System/models/diabetes_model.sav')
heart_model = joblib.load('D:/Multiple Disease Prediction System/Multiple Disease Prediction System/models/heart_disease_model.sav')
# Load the lung cancer prediction model
lung_cancer_svm_model = joblib.load('D:/Multiple Disease Prediction System/Multiple Disease Prediction System/models/lung_cancer_svm_model.sav')
chronic_disease_model = joblib.load('D:/Multiple Disease Prediction System/Multiple Disease Prediction System/models/chronic_disease_model.sav')
breast_cancer_model = joblib.load('D:/Multiple Disease Prediction System/Multiple Disease Prediction System/models/breast_cancer_model.sav')


# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Diabetes Prediction',
        'Heart disease Prediction',
        'Lung Cancer Prediction',
        'Chronic Kidney prediction',
        'Breast Cancer Prediction',

    ],
        icons=['','activity', 'heart', 'person','person','person','person','bar-chart-fill'],
        default_index=0)




# Diabetes prediction page
if selected == 'Diabetes Prediction':  # pagetitle
    st.title("Diabetes disease prediction")
    # columns
    # no inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnencies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood pressure  value")
    with col1:

        SkinThickness = st.number_input("Sckinthickness value")

    with col2:

        Insulin = st.number_input("Insulin value ")
    with col3:
        BMI = st.number_input("BMI value")
    with col1:
        DiabetesPedigreefunction = st.number_input(
            "Diabetespedigreefunction value")
    with col2:

        Age = st.number_input("AGE")

    # code for prediction
    diabetes_dig = ''

    # button
    if st.button("Diabetes test result"):
        diabetes_prediction=[[]]
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if diabetes_prediction[0] == 1:
            diabetes_dig = "we are really sorry to say but it seems like you are Diabetic."
        
        else:
            diabetes_dig = 'Congratulation,You are not diabetic'

        st.success(name+' , ' + diabetes_dig)
        
        



# Heart prediction page
if selected == 'Heart disease Prediction':
    st.title("Heart disease prediction")
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
    with col2:
        sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            sex = 1
        elif value == "female":
            sex = 0
    with col3:
        cp=0
        display = ("typical angina","atypical angina","non — anginal pain","asymptotic")
        options = list(range(len(display)))
        value = st.selectbox("Chest_Pain Type", options, format_func=lambda x: display[x])
        if value == "typical angina":
            cp = 0
        elif value == "atypical angina":
            cp = 1
        elif value == "non — anginal pain":
            cp = 2
        elif value == "asymptotic":
            cp = 3
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")

    with col2:

        chol = st.number_input("Serum Cholestrol")
    
    with col3:
        restecg=0
        display = ("normal","having ST-T wave abnormality","left ventricular hyperthrophy")
        options = list(range(len(display)))
        value = st.selectbox("Resting ECG", options, format_func=lambda x: display[x])
        if value == "normal":
            restecg = 0
        elif value == "having ST-T wave abnormality":
            restecg = 1
        elif value == "left ventricular hyperthrophy":
            restecg = 2

    with col1:
        exang=0
        thalach = st.number_input("Max Heart Rate Achieved")
   
    with col2:
        oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    with col3:
        slope=0
        display = ("upsloping","flat","downsloping")
        options = list(range(len(display)))
        value = st.selectbox("Peak exercise ST segment", options, format_func=lambda x: display[x])
        if value == "upsloping":
            slope = 0
        elif value == "flat":
            slope = 1
        elif value == "downsloping":
            slope = 2
    with col1:
        ca = st.number_input("Number of major vessels (0–3) colored by flourosopy")
    with col2:
        thal=0
        display = ("normal","fixed defect","reversible defect")
        options = list(range(len(display)))
        value = st.selectbox("thalassemia", options, format_func=lambda x: display[x])
        if value == "normal":
            thal = 0
        elif value == "fixed defect":
            thal = 1
        elif value == "reversible defect":
            thal = 2
    with col3:
        agree = st.checkbox('Exercise induced angina')
        if agree:
            exang = 1
        else:
            exang=0
    with col1:
        agree1 = st.checkbox('fasting blood sugar > 120mg/dl')
        if agree1:
            fbs = 1
        else:
            fbs=0
    # code for prediction
    heart_dig = ''
    

    # button
    if st.button("Heart test result"):
        heart_prediction=[[]]
        # change the parameters according to the model
        
        # b=np.array(a, dtype=float)
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_dig = 'we are really sorry to say but it seems like you have Heart Disease.'
         
            
        else:
            heart_dig = "Congratulation , You don't have Heart Disease."
        st.success(name +' , ' + heart_dig)


# Load the dataset
lung_cancer_data = pd.read_csv('D:/Multiple Disease Prediction System/Multiple Disease Prediction System/data/lung_cancer.csv')

# Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Lung Cancer prediction page
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")

    # Columns
    # No inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
    with col2:
        age = st.number_input("Age")
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

    with col2:
        anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])

    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])

    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    # Code for prediction
    cancer_result = ''

    # Button
    if st.button("Predict Lung Cancer"):
        # Create a DataFrame with user inputs
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        # Map string values to numeric
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

        # Strip leading and trailing whitespaces from column names
        user_data.columns = user_data.columns.str.strip()

        # Convert columns to numeric where necessary
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Perform prediction
        cancer_prediction = lung_cancer_svm_model.predict(user_data)

        # Display result
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
    
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
        st.success(name + ', ' + cancer_result)


from sklearn.preprocessing import LabelEncoder
import joblib
# Chronic Kidney Disease Prediction Page
if selected == 'Chronic Kidney prediction':
    st.title("Chronic Kidney Disease Prediction")
    # Add the image for Chronic Kidney Disease prediction if needed
    name = st.text_input("Name:")
    # Columns
    # No inputs from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Enter your age", 1, 100, 25)  # 2
    with col2:
        bp = st.slider("Enter your Blood Pressure", 50, 200, 120)  # Add your own ranges
    with col3:
        sg = st.slider("Enter your Specific Gravity", 1.0, 1.05, 1.02)  # Add your own ranges

    with col1:
        al = st.slider("Enter your Albumin", 0, 5, 0)  # Add your own ranges
    with col2:
        su = st.slider("Enter your Sugar", 0, 5, 0)  # Add your own ranges
    with col3:
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        rbc = 1 if rbc == "Normal" else 0

    with col1:
        pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
        pc = 1 if pc == "Normal" else 0
    with col2:
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
        pcc = 1 if pcc == "Present" else 0
    with col3:
        ba = st.selectbox("Bacteria", ["Present", "Not Present"])
        ba = 1 if ba == "Present" else 0

    with col1:
        bgr = st.slider("Enter your Blood Glucose Random", 50, 200, 120)  # Add your own ranges
    with col2:
        bu = st.slider("Enter your Blood Urea", 10, 200, 60)  # Add your own ranges
    with col3:
        sc = st.slider("Enter your Serum Creatinine", 0, 10, 3)  # Add your own ranges

    with col1:
        sod = st.slider("Enter your Sodium", 100, 200, 140)  # Add your own ranges
    with col2:
        pot = st.slider("Enter your Potassium", 2, 7, 4)  # Add your own ranges
    with col3:
        hemo = st.slider("Enter your Hemoglobin", 3, 17, 12)  # Add your own ranges

    with col1:
        pcv = st.slider("Enter your Packed Cell Volume", 20, 60, 40)  # Add your own ranges
    with col2:
        wc = st.slider("Enter your White Blood Cell Count", 2000, 20000, 10000)  # Add your own ranges
    with col3:
        rc = st.slider("Enter your Red Blood Cell Count", 2, 8, 4)  # Add your own ranges

    with col1:
        htn = st.selectbox("Hypertension", ["Yes", "No"])
        htn = 1 if htn == "Yes" else 0
    with col2:
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        dm = 1 if dm == "Yes" else 0
    with col3:
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
        cad = 1 if cad == "Yes" else 0

    with col1:
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        appet = 1 if appet == "Good" else 0
    with col2:
        pe = st.selectbox("Pedal Edema", ["Yes", "No"])
        pe = 1 if pe == "Yes" else 0
    with col3:
        ane = st.selectbox("Anemia", ["Yes", "No"])
        ane = 1 if ane == "Yes" else 0

    # Code for prediction
    kidney_result = ''

    # Prediction button
    if st.button("Predict Chronic Kidney Disease"):
        # Create DataFrame for user input
        user_input = pd.DataFrame({
            'age': [age],
            'bp': [bp],
            'sg': [sg],
            'al': [al],
            'su': [su],
            'rbc': [rbc],
            'pc': [pc],
            'pcc': [pcc],
            'ba': [ba],
            'bgr': [bgr],
            'bu': [bu],
            'sc': [sc],
            'sod': [sod],
            'pot': [pot],
            'hemo': [hemo],
            'pcv': [pcv],
            'wc': [wc],
            'rc': [rc],
            'htn': [htn],
            'dm': [dm],
            'cad': [cad],
            'appet': [appet],
            'pe': [pe],
            'ane': [ane]
        })

        # Ensure column order matches training data
        user_input = user_input[['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
                                 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 
                                 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]

        # Perform prediction
        kidney_prediction = chronic_disease_model.predict(user_input)

        # Display result
        if kidney_prediction[0] == 1:
            kidney_prediction_dig = "We are sorry to inform you that it seems you have kidney disease."
        else:
            kidney_prediction_dig = "Congratulations! You do not have kidney disease."

        st.success(f"{name}, {kidney_prediction_dig}")
    

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction")
    name = st.text_input("Name:")

    # Input fields for features
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Enter your Radius Mean", 6.0, 30.0, 15.0)
        texture_mean = st.slider("Enter your Texture Mean", 9.0, 40.0, 20.0)
        perimeter_mean = st.slider("Enter your Perimeter Mean", 43.0, 190.0, 90.0)

    with col2:
        area_mean = st.slider("Enter your Area Mean", 143.0, 2501.0, 750.0)
        smoothness_mean = st.slider("Enter your Smoothness Mean", 0.05, 0.25, 0.1)
        compactness_mean = st.slider("Enter your Compactness Mean", 0.02, 0.3, 0.15)

    with col3:
        concavity_mean = st.slider("Enter your Concavity Mean", 0.0, 0.5, 0.2)
        concave_points_mean = st.slider("Enter your Concave Points Mean", 0.0, 0.2, 0.1)
        symmetry_mean = st.slider("Enter your Symmetry Mean", 0.1, 1.0, 0.5)

    with col1:
        fractal_dimension_mean = st.slider("Enter your Fractal Dimension Mean", 0.01, 0.1, 0.05)
        radius_se = st.slider("Enter your Radius SE", 0.1, 3.0, 1.0)
        texture_se = st.slider("Enter your Texture SE", 0.2, 2.0, 1.0)

    with col2:
        perimeter_se = st.slider("Enter your Perimeter SE", 1.0, 30.0, 10.0)
        area_se = st.slider("Enter your Area SE", 6.0, 500.0, 150.0)
        smoothness_se = st.slider("Enter your Smoothness SE", 0.001, 0.03, 0.01)

    with col3:
        compactness_se = st.slider("Enter your Compactness SE", 0.002, 0.2, 0.1)
        concavity_se = st.slider("Enter your Concavity SE", 0.0, 0.05, 0.02)
        concave_points_se = st.slider("Enter your Concave Points SE", 0.0, 0.03, 0.01)

    with col1:
        symmetry_se = st.slider("Enter your Symmetry SE", 0.1, 1.0, 0.5)
        fractal_dimension_se = st.slider("Enter your Fractal Dimension SE", 0.01, 0.1, 0.05)

    with col2:
        radius_worst = st.slider("Enter your Radius Worst", 7.0, 40.0, 20.0)
        texture_worst = st.slider("Enter your Texture Worst", 12.0, 50.0, 25.0)
        perimeter_worst = st.slider("Enter your Perimeter Worst", 50.0, 250.0, 120.0)

    with col3:
        area_worst = st.slider("Enter your Area Worst", 185.0, 4250.0, 1500.0)
        smoothness_worst = st.slider("Enter your Smoothness Worst", 0.07, 0.3, 0.15)
        compactness_worst = st.slider("Enter your Compactness Worst", 0.03, 0.6, 0.3)

    with col1:
        concavity_worst = st.slider("Enter your Concavity Worst", 0.0, 0.8, 0.4)
        concave_points_worst = st.slider("Enter your Concave Points Worst", 0.0, 0.2, 0.1)
        symmetry_worst = st.slider("Enter your Symmetry Worst", 0.1, 1.0, 0.5)

    with col2:
        fractal_dimension_worst = st.slider("Enter your Fractal Dimension Worst", 0.01, 0.2, 0.1)

    # Code for prediction
    breast_cancer_result = ''

    # Button for prediction
    if st.button("Predict Breast Cancer"):
        # Create a DataFrame with user inputs
        user_input = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave points_mean': [concave_points_mean],
            'symmetry_mean': [symmetry_mean],
            'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se],
            'texture_se': [texture_se],
            'perimeter_se': [perimeter_se],
            'area_se': [area_se],
            'smoothness_se': [smoothness_se],
            'compactness_se': [compactness_se],
            'concavity_se': [concavity_se],
            'concave points_se': [concave_points_se],
            'symmetry_se': [symmetry_se],
            'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave points_worst': [concave_points_worst],
            'symmetry_worst': [symmetry_worst],
            'fractal_dimension_worst': [fractal_dimension_worst],
        })

        # Perform prediction
        breast_cancer_prediction = breast_cancer_model.predict(user_input)
        
        # Display result
        if breast_cancer_prediction[0] == 1:
            breast_cancer_result = "The model predicts that you have Breast Cancer."
        else:
            breast_cancer_result = "The model predicts that you don't have Breast Cancer."

        st.success(breast_cancer_result)