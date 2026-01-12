import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("student_performance_dataset.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎓 Student Performance App")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "EDA", "Model Training", "Prediction"]
)

# ---------------- PREPROCESSING ----------------
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(["Final_Exam_Score", "Pass_Fail"], axis=1)
y = df_encoded["Final_Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("📊 Student Performance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", df.shape[0])
    col2.metric("Average Study Hours", round(df["Study_Hours_per_Week"].mean(), 2))
    col3.metric("Pass Percentage",
                f"{round((df['Pass_Fail']=='Pass').mean()*100,2)}%")

    st.subheader("📈 Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Final_Exam_Score"], kde=True, ax=ax)
    st.pyplot(fig)

# ---------------- EDA ----------------
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Gender", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Study Hours vs Final Exam Score")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="Study_Hours_per_Week",
        y="Final_Exam_Score",
        hue="Pass_Fail",
        data=df,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- MODEL TRAINING ----------------
elif page == "Model Training":
    st.title("🤖 Model Training & Evaluation")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "KNN": KNeighborsRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = round(r2_score(y_test, preds), 3)

    st.subheader("📌 Model Performance (R² Score)")
    st.dataframe(pd.DataFrame(results.items(),
                              columns=["Model", "R² Score"]))

# ---------------- PREDICTION ----------------
elif page == "Prediction":
    st.title("🎯 Student Score Prediction")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("Study Hours per Week", 1, 50, 20)
        attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
        past_score = st.slider("Past Exam Score", 0, 100, 60)

    with col2:
        gender = st.selectbox("Gender", df["Gender"].unique())
        parent_edu = st.selectbox("Parental Education", df["Parental_Education_Level"].unique())
        internet = st.selectbox("Internet Access", df["Internet_Access_at_Home"].unique())
        activity = st.selectbox("Extracurricular Activities", df["Extracurricular_Activities"].unique())

    if st.button("Predict Score"):
        input_data = pd.DataFrame({
            "Student_ID": [0],
            "Gender": [gender],
            "Study_Hours_per_Week": [study_hours],
            "Attendance_Rate": [attendance],
            "Past_Exam_Scores": [past_score],
            "Parental_Education_Level": [parent_edu],
            "Internet_Access_at_Home": [internet],
            "Extracurricular_Activities": [activity]
        })

        for col in input_data.select_dtypes(include='object'):
            input_data[col] = le.fit_transform(input_data[col])

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        prediction = model.predict(input_data)[0]

        st.success(f"📘 Predicted Final Exam Score: {round(prediction,2)}")

        if prediction >= 60:
            st.success("Result: PASS")
        else:
            st.error("Result: FAIL")
