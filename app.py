import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Student Mental Health Score Predictor",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "mental_health_pipeline.joblib")
    return joblib.load(model_path)

pipeline = load_pipeline()

st.title("Student Mental Health Score Predictor")
st.write(
    "Estimate student wellbeing scores based on digital consumption habits, sleep, "
    "study routines, and physical activity levels."
)

tab1, tab2 = st.tabs(["Score Predictor", "What-If Habit Simulator"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Academics")
        age = st.slider("Age", min_value=18, max_value=25, value=21, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        country = st.selectbox(
            "Country",
            ["India", "USA", "Canada", "Australia", "UK", "Germany", "Turkey", "Mexico", "France", "Spain", "Other"]
        )
        academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"], index=1)
        stress_level = st.selectbox("Perceived Stress Level", ["Low", "Medium", "High", "Very High"], index=1)

    with col2:
        st.subheader("Digital Habits & Daily Routine")
        platform = st.selectbox(
            "Most Used Social Media Platform",
            ["Instagram", "TikTok", "Facebook", "LinkedIn", "YouTube", "Twitter", "Snapchat", "WhatsApp", "Other"],
            index=0
        )
        purpose = st.selectbox(
            "Primary Purpose of Use",
            ["Entertainment", "Education", "Networking", "News"],
            index=0
        )
        screen_hours = st.slider("Daily Screen Time (Hours)", min_value=0.5, max_value=12.0, value=4.0, step=0.1)
        daily_unlocks = st.slider("Daily Phone Unlocks", min_value=20, max_value=300, value=120, step=1)
        study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=12.0, value=4.5, step=0.1)
        activity_hours = st.slider("Daily Physical Activity (Hours)", min_value=0.0, max_value=6.0, value=1.5, step=0.1)
        sleep_hours = st.slider("Sleep Duration (Hours per Night)", min_value=3.0, max_value=12.0, value=7.0, step=0.1)

    st.markdown("---")

    if st.button("Calculate Mental Health Score", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Country": country,
            "Academic_Level": academic_level,
            "Most_Used_Platform": platform,
            "Purpose_Of_Use": purpose,
            "Avg_Daily_Usage_Hours": screen_hours,
            "Daily_Unlocks": daily_unlocks,
            "Study_Hours": study_hours,
            "Physical_Activity_Hours": activity_hours,
            "Sleep_Hours_Per_Night": sleep_hours,
            "Stress_Level": stress_level
        }])

        predicted_score = float(pipeline.predict(input_data)[0])
        predicted_score = round(max(1.0, min(10.0, predicted_score)), 2)

        st.subheader("Assessment Results")
        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 2])
        
        with metric_col1:
            st.metric(label="Predicted Score", value=f"{predicted_score:.2f} / 10.0")
        
        with metric_col2:
            if predicted_score >= 7.5:
                st.success("Status: Healthy Wellbeing")
            elif predicted_score >= 5.5:
                st.info("Status: Moderate Wellbeing")
            else:
                st.warning("Status: Elevated Stress / At Risk")

        with metric_col3:
            st.progress(predicted_score / 10.0)

        st.write("### Lifestyle Insights")
        insights = []
        if sleep_hours < 6.5:
            insights.append("Sleep duration is below 6.5 hours. Sleep duration is the strongest positive predictor of the mental health score.")
        elif sleep_hours >= 8.0:
            insights.append("Adequate sleep duration (>8 hours) provides a strong recovery buffer.")

        if screen_hours > 6.0:
            insights.append(f"Screen usage ({screen_hours} hrs) is elevated. Screen time and frequent unlocks strongly correlate with lower scores.")
        
        if study_hours >= 4.0:
            insights.append("Regular study hours indicate steady daily structure and academic consistency.")

        if activity_hours < 1.0:
            insights.append("Physical activity is under 1 hour daily. Light exercise offers moderate wellbeing improvements.")

        for item in insights:
            st.write(f"- {item}")

with tab2:
    st.subheader("Counterfactual Simulation: Habit Improvements")
    st.write("Adjust your habits to see the immediate predicted impact on your mental health score.")

    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.markdown("**Baseline Profile**")
        base_screen = st.slider("Current Screen Time (Hours)", 1.0, 10.0, 6.5, 0.5, key="base_screen")
        base_sleep = st.slider("Current Sleep (Hours)", 3.0, 10.0, 5.5, 0.5, key="base_sleep")
        base_activity = st.slider("Current Activity (Hours)", 0.0, 4.0, 0.5, 0.5, key="base_act")
        base_stress = st.selectbox("Current Stress Level", ["Low", "Medium", "High", "Very High"], index=2, key="base_stress")

    with sim_col2:
        st.markdown("**Target Improvement Goals**")
        target_screen = st.slider("Target Screen Time (Hours)", 1.0, 10.0, 3.5, 0.5, key="target_screen")
        target_sleep = st.slider("Target Sleep (Hours)", 3.0, 10.0, 8.0, 0.5, key="target_sleep")
        target_activity = st.slider("Target Activity (Hours)", 0.0, 4.0, 2.0, 0.5, key="target_act")
        target_stress = st.selectbox("Target Stress Level", ["Low", "Medium", "High", "Very High"], index=0, key="target_stress")

    if st.button("Run Simulation", type="primary", use_container_width=True):
        base_df = pd.DataFrame([{
            "Age": 21, "Gender": "Female", "Country": "India", "Academic_Level": "Undergraduate",
            "Most_Used_Platform": "Instagram", "Purpose_Of_Use": "Entertainment",
            "Avg_Daily_Usage_Hours": base_screen, "Daily_Unlocks": int(base_screen * 30),
            "Study_Hours": 3.0, "Physical_Activity_Hours": base_activity,
            "Sleep_Hours_Per_Night": base_sleep, "Stress_Level": base_stress
        }])
        
        target_df = pd.DataFrame([{
            "Age": 21, "Gender": "Female", "Country": "India", "Academic_Level": "Undergraduate",
            "Most_Used_Platform": "Instagram", "Purpose_Of_Use": "Entertainment",
            "Avg_Daily_Usage_Hours": target_screen, "Daily_Unlocks": int(target_screen * 20),
            "Study_Hours": 4.5, "Physical_Activity_Hours": target_activity,
            "Sleep_Hours_Per_Night": target_sleep, "Stress_Level": target_stress
        }])

        base_score = round(float(pipeline.predict(base_df)[0]), 2)
        target_score = round(float(pipeline.predict(target_df)[0]), 2)
        delta = round(target_score - base_score, 2)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Baseline Score", f"{base_score:.2f} / 10.0")
        with res_col2:
            st.metric("Target Score", f"{target_score:.2f} / 10.0", delta=f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}")
        with res_col3:
            percentage_gain = round((delta / base_score) * 100, 1)
            st.metric("Expected Improvement", f"{percentage_gain}%")
