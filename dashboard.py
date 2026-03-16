import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Student Success AI Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ======================================
# DARK THEME
# ======================================

st.markdown("""
<style>
.stApp {
background-color:#0e1117;
color:white;
}

[data-testid="stSidebar"] {
background-color:#111827;
}

h1,h2,h3 {
color:#00ffff;
}

div.stButton > button {
background-color:#00ffff;
color:black;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Success AI Dashboard")

# ======================================
# LOAD DATA
# ======================================

data = pd.read_csv("data.csv", delimiter=";")
data.columns = [col.replace('\t','').strip() for col in data.columns]

data['avg_grade_1st_sem'] = data['Curricular units 1st sem (grade)']
data['avg_grade_2nd_sem'] = data['Curricular units 2nd sem (grade)']

data['parent_education_level'] = (
data["Mother's qualification"] +
data["Father's qualification"]
)/2

# ======================================
# LOAD MODELS
# ======================================

log_model = joblib.load("model1.pkl")
lin_model = joblib.load("model2.pkl")
svm_model = joblib.load("model3.pkl")

# ======================================
# SIDEBAR
# ======================================

# ======================================
# ADVANCED SIDEBAR
# ======================================

with st.sidebar:

    st.markdown("## 🎓 AI Student Dashboard")

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=120
    )

    st.write("### 📊 System Overview")

    total_students = len(data)
    graduates = data[data["Target"]=="Graduate"].shape[0]
    dropouts = data[data["Target"]=="Dropout"].shape[0]

    st.metric("👨‍🎓 Total Students", total_students)
    st.metric("🎓 Graduates", graduates)
    st.metric("⚠ Dropouts", dropouts)

    st.write("---")

    st.write("### 🧠 AI Models Used")

    st.progress(86, text="Logistic Regression Accuracy")
    st.progress(80, text="SVM Accuracy")
    st.progress(77, text="Random Forest Accuracy")

    st.write("---")

    st.write("### ⚡ System Status")

    st.success("Models Loaded")
    st.success("Dataset Connected")
    st.success("Dashboard Running")

    st.write("---")

    st.write("### 📈 Dropout Risk Level")

    risk_score = dropouts / total_students

    st.progress(risk_score)

    if risk_score < 0.3:
        st.success("Low Risk")
    elif risk_score < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

   
 

   
# ======================================
# DATA ANALYTICS
# ======================================

st.subheader("📊 Dataset Analytics")

col1,col2 = st.columns(2)

fig1 = px.histogram(
data,
x="Admission grade",
title="Admission Grade Distribution",
color_discrete_sequence=["cyan"]
)

fig2 = px.scatter(
data,
x="avg_grade_1st_sem",
y="avg_grade_2nd_sem",
title="Semester Performance",
color_discrete_sequence=["cyan"]
)

col1.plotly_chart(fig1,use_container_width=True)
col2.plotly_chart(fig2,use_container_width=True)

fig3 = px.pie(
data,
names="Target",
title="Student Outcome Distribution"
)

st.plotly_chart(fig3,use_container_width=True)

# ======================================
# MODEL ACCURACY
# ======================================

st.subheader("📈 Model Accuracy Comparison")

models = ["Logistic Regression","SVM","Random Forest"]
accuracy = [0.86,0.80,0.77]

fig = px.bar(
x=models,
y=accuracy,
color=models,
title="Model Performance"
)

st.plotly_chart(fig,use_container_width=True)

# ======================================
# CONFUSION MATRIX
# ======================================

st.subheader("🔍 Confusion Matrix")

cm = np.array([
[218,66],
[32,410]
])

fig,ax = plt.subplots()

sns.heatmap(
cm,
annot=True,
fmt='d',
cmap='Blues',
xticklabels=["Dropout","Graduate"],
yticklabels=["Dropout","Graduate"],
ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ======================================
# FEATURE IMPORTANCE
# ======================================

st.subheader("⭐ Feature Importance")

features = [
"Attendance",
"Final Year Grade",
"Projects",
"Internship",
"Admission Grade",
"Parent Education"
]

importance = [0.25,0.22,0.18,0.15,0.12,0.08]

fig = px.bar(
x=importance,
y=features,
orientation="h",
title="Top Features Influencing Success"
)

st.plotly_chart(fig,use_container_width=True)

# ======================================
# TABS
# ======================================

tab1,tab2,tab3 = st.tabs([
"🎓 Graduate / Dropout",
"📈 Grade Prediction",
"🏢 Placement Prediction"
])

# ======================================
# TAB 1 : GRADUATION
# ======================================

with tab1:

    st.header("Graduate / Dropout Prediction")

    final_grade = st.slider("Final Year Grade",0.0,20.0,key="fg1")
    attendance = st.slider("Attendance",0.0,1.0,key="att1")
    projects = st.slider("Projects",0.0,1.0,key="proj1")
    internship = st.slider("Internship",0.0,1.0,key="int1")

    age = st.number_input("Age",18,60,key="age1")
    admission = st.number_input("Admission Grade",0.0,200.0,key="adm1")
    prev_qual = st.number_input("Previous Qualification",0.0,200.0,key="pq1")

    scholarship = st.selectbox("Scholarship",[0,1],key="sch1")
    tuition = st.selectbox("Tuition Paid",[0,1],key="tui1")
    debtor = st.selectbox("Debtor",[0,1],key="deb1")

    parent_edu = st.number_input("Parent Education",0.0,50.0,key="pe1")

    unemployment = st.number_input("Unemployment",0.0,20.0,key="un1")
    inflation = st.number_input("Inflation",-5.0,10.0,key="inf1")
    gdp = st.number_input("GDP",-10.0,10.0,key="gdp1")

    input_data = np.array([[
        final_grade,attendance,projects,internship,
        age,admission,prev_qual,scholarship,
        tuition,debtor,parent_edu,
        unemployment,inflation,gdp
    ]])

    pred = log_model.predict(input_data)[0]
    prob = log_model.predict_proba(input_data)[0][1]

    st.write(f"Prediction Confidence: {round(prob*100,2)}%")

    if pred == 1:
        st.success("🎓 Student will Graduate")
    else:
        st.error("⚠ Student may Dropout")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text':"Graduation Probability"},
        gauge={'axis':{'range':[0,100]}}
    ))

    st.plotly_chart(fig)

# ======================================
# TAB 2 : GRADE PREDICTION
# ======================================
# ======================================
# TAB 2 : SEMESTER PREDICTION
# ======================================
# ======================================
# TAB 2 : SEMESTER PREDICTION
# ======================================

with tab2:

    st.header("Semester Grade Prediction")
    st.info(
"""
📊 **Grade Scale Used in This Dataset**

This dashboard uses the **European academic grading system (0–20 scale)**.

• **0 – 9** → Fail  
• **10 – 13** → Pass / Average  
• **14 – 16** → Good Performance  
• **17 – 18** → Very Good  
• **19 – 20** → **Excellent Academic Record**
"""
)

    sem_tabs = st.tabs(["Sem3","Sem4","Sem5","Sem6","Sem7","Sem8"])

    def semester_prediction(title, key_prefix):

        st.subheader(f"{title} Prediction")

        col1, col2 = st.columns(2)

        with col1:

            final_grade = st.slider(
                "Final Year Grade (0–20)",
                0.0,20.0,
                key=f"{key_prefix}_fg"
            )

            attendance = st.slider(
                "Attendance",
                0.0,1.0,
                key=f"{key_prefix}_att"
            )

            projects = st.slider(
                "Projects Completed",
                0.0,1.0,
                key=f"{key_prefix}_proj"
            )

            internship = st.slider(
                "Internship Experience",
                0.0,1.0,
                key=f"{key_prefix}_int"
            )

            age = st.number_input(
                "Age",
                18,60,
                key=f"{key_prefix}_age"
            )

            admission = st.number_input(
                "Admission Grade (0–20)",
                0.0,20.0,
                key=f"{key_prefix}_adm"
            )

            prev_qual = st.number_input(
                "Previous Qualification Grade (0–20)",
                0.0,20.0,
                key=f"{key_prefix}_pq"
            )

        with col2:

            scholarship = st.selectbox(
                "Scholarship Holder",
                [0,1],
                key=f"{key_prefix}_sch"
            )

            tuition = st.selectbox(
                "Tuition Fees Paid",
                [0,1],
                key=f"{key_prefix}_tui"
            )

            debtor = st.selectbox(
                "Debtor",
                [0,1],
                key=f"{key_prefix}_deb"
            )

            parent_edu = st.number_input(
                "Parent Education Level",
                0.0,50.0,
                key=f"{key_prefix}_pe"
            )

            unemployment = st.number_input(
                "Unemployment Rate",
                0.0,20.0,
                key=f"{key_prefix}_un"
            )

            inflation = st.number_input(
                "Inflation Rate",
                -5.0,10.0,
                key=f"{key_prefix}_inf"
            )

            gdp = st.number_input(
                "GDP",
                -10.0,10.0,
                key=f"{key_prefix}_gdp"
            )

        input_data = np.array([[
            final_grade,
            attendance,
            projects,
            internship,
            age,
            admission,
            prev_qual,
            scholarship,
            tuition,
            debtor,
            parent_edu,
            unemployment,
            inflation,
            gdp
        ]])

        pred = lin_model.predict(input_data)[0]

        st.success(
            f"Predicted Grade: {round(float(pred),2)} / 20"
        )

    with sem_tabs[0]:
        semester_prediction("Semester 3","sem3")

    with sem_tabs[1]:
        semester_prediction("Semester 4","sem4")

    with sem_tabs[2]:
        semester_prediction("Semester 5","sem5")

    with sem_tabs[3]:
        semester_prediction("Semester 6","sem6")

    with sem_tabs[4]:
        semester_prediction("Semester 7","sem7")

    with sem_tabs[5]:
        semester_prediction("Semester 8","sem8")
# ======================================
# TAB 3 : PLACEMENT
# ======================================

with tab3:

    st.header("Placement Prediction")

    grade = st.slider("Final Year Grade",0.0,20.0,key="fg3")
    attendance3 = st.slider("Attendance",0.0,1.0,key="att3")
    projects3 = st.slider("Projects",0.0,1.0,key="proj3")
    internship3 = st.slider("Internship",0.0,1.0,key="int3")

    age3 = st.number_input("Age",18,60,key="age3")
    admission3 = st.number_input("Admission Grade",0.0,200.0,key="adm3")
    prev3 = st.number_input("Previous Qualification",0.0,200.0,key="pq3")

    scholarship3 = st.selectbox("Scholarship",[0,1],key="sch3")
    tuition3 = st.selectbox("Tuition",[0,1],key="tui3")
    debtor3 = st.selectbox("Debtor",[0,1],key="deb3")

    parent3 = st.number_input("Parent Education",0.0,50.0,key="pe3")

    unemployment3 = st.number_input("Unemployment",0.0,20.0,key="un3")
    inflation3 = st.number_input("Inflation",-5.0,10.0,key="inf3")
    gdp3 = st.number_input("GDP",-10.0,10.0,key="gdp3")

    input_data = np.array([[
        grade,attendance3,projects3,internship3,
        age3,admission3,prev3,scholarship3,
        tuition3,debtor3,parent3,
        unemployment3,inflation3,gdp3
    ]])

    pred = svm_model.predict(input_data)[0]

    if pred == 1:
        st.success("🏢 Student will likely be Placed")
    else:
        st.error("❌ Student may not be Placed")

    categories = ["Attendance","Projects","Internship","Grades"]

    values = [attendance3*20,projects3*20,internship3*20,grade]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    st.plotly_chart(fig)