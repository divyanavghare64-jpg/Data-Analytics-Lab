
import streamlit as st
import joblib

st.title("Machine Learning Web App")
st.write("Select the model type and algorithms below.")

st.sidebar.title("Navigation")
task = st.sidebar.radio("Select Problem Type:", ["Classification", "Regression"])

if task == "Classification":
    st.header("Classification Task")
    
    algo = st.selectbox(
        "Select Classification Algorithm:",
        ["Logistic Regression", "Decision Tree", "Support Vector Machine", "K-Nearest Neighbors", "Naive Bayes"]
    )
    st.write(f"Algorithm selected: **{algo}**")
    
    model = joblib.load('best_classification_model.pkl')
    scaler = joblib.load('clf_scaler.pkl')
    features = joblib.load('clf_features.pkl')
    
    st.subheader("Input Values:")
    col1, col2 = st.columns(2)
    with col1:
        f1 = st.number_input("Mean Radius", value=14.0)
        f2 = st.number_input("Mean Texture", value=19.0)
    with col2:
        f3 = st.number_input("Mean Perimeter", value=90.0)
        f4 = st.number_input("Mean Area", value=650.0)
        
    user_data = [f1, f2, f3, f4] + [0.0] * (len(features) - 4)
    
    if st.button("Predict"):
        scaled_data = scaler.transform([user_data])
        res = model.predict(scaled_data)[0]
        if res == 1:
            st.success("Prediction Result: Benign (1)")
        else:
            st.error("Prediction Result: Malignant (0)")

elif task == "Regression":
    st.header("Regression Task")
    
    algo = st.selectbox(
        "Select Regression Algorithm:",
        ["Linear Regression", "Decision Tree Regressor", "Support Vector Regressor", "KNN Regressor"]
    )
    st.write(f"Algorithm selected: **{algo}**")
    
    model = joblib.load('best_regression_model.pkl')
    scaler = joblib.load('reg_scaler.pkl')
    features = joblib.load('reg_features.pkl')
    
    st.subheader("Input Values:")
    col1, col2 = st.columns(2)
    with col1:
        v1 = st.number_input("Age", value=0.03)
        v2 = st.number_input("Sex", value=-0.04)
        v3 = st.number_input("BMI", value=0.06)
        v4 = st.number_input("BP", value=0.02)
    with col2:
        v5 = st.number_input("S1", value=-0.04)
        v6 = st.number_input("S2", value=-0.03)
        v7 = st.number_input("S3", value=-0.04)
        v8 = st.number_input("S4", value=-0.002)
        
    user_data = [v1, v2, v3, v4, v5, v6, v7, v8, 0.0, 0.0]
    
    if st.button("Predict"):
        scaled_data = scaler.transform([user_data])
        res = model.predict(scaled_data)[0]
        st.info(f"Predicted Progression: {res:.2f}")