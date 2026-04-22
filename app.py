import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model + encoders
model = pickle.load(open("model.pkl", "rb"))
le_gender = pickle.load(open("gender.pkl", "rb"))
le_activity = pickle.load(open("activity.pkl", "rb"))
le_weather = pickle.load(open("weather.pkl", "rb"))

st.set_page_config(page_title="Health Prediction", layout="wide")
st.title("💧 Water Intake & Health Prediction")

# Load dataset for EDA
df = pd.read_csv("your_dataset.csv")

# -------------------------
# USER INPUTS (Available to all tabs)
# -------------------------
st.header("📝 Enter Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Your Age", 0, 100, 0)
    gender = st.selectbox("Gender", le_gender.classes_, index=0)
    body_weight = st.number_input("Body Weight (kg)", 0.0, 200.0, 0.0)

with col2:
    daily_water = st.number_input("Daily Water Intake (L)", 0.0, 10.0, 0.0)
    activity = st.selectbox("Physical Activity", le_activity.classes_, index=0)
    weather = st.selectbox("Weather Condition", le_weather.classes_, index=0)

# -------------------------
# ENCODING INPUT (For prediction)
# -------------------------
gender_encoded = le_gender.transform([gender])[0]
activity_encoded = le_activity.transform([activity])[0]
weather_encoded = le_weather.transform([weather])[0]

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Prediction", "📈 Analysis"])

# ========================
# TAB 1: EDA
# ========================
with tab1:
    st.header("📊 Data Distributions & Your Profile")
    
    # User Input Summary
    st.subheader("📝 Your Current Inputs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Age", f"{age} years")
        st.metric("Gender", gender)
    
    with col2:
        st.metric("Body Weight", f"{body_weight} kg")
        st.metric("Daily Water Intake", f"{daily_water} L")
    
    with col3:
        st.metric("Physical Activity", activity)
        st.metric("Weather Condition", weather)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Your age'], bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax.axvline(x=age, color='red', linestyle='--', linewidth=2, label=f'Your Age: {age}')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Age Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Body Weight Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Body Weight'], bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax.axvline(x=body_weight, color='red', linestyle='--', linewidth=2, label=f'Your Weight: {body_weight}kg')
        ax.set_xlabel('Body Weight (kg)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Body Weight Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Daily Water Intake Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Daily Water Intake'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(x=daily_water, color='red', linestyle='--', linewidth=2, label=f'Your Intake: {daily_water}L')
        ax.set_xlabel('Daily Water Intake (L)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Water Intake Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
    
    with col4:
        st.subheader("Health Status Distribution")
        target_dist = df['target'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FF6B6B', '#4ECDC4']
        labels = ['Unhealthy (0)', 'Healthy (1)']
        ax.bar(labels, target_dist.values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Health Status Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Physical Activity Distribution")
        activity_counts = df['Physical Activity'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(activity_counts.index, activity_counts.values, color='#9b59b6', alpha=0.7, edgecolor='black')
        # Highlight user's activity level
        user_idx = list(activity_counts.index).index(activity) if activity in activity_counts.index else -1
        if user_idx >= 0:
            ax.bar(user_idx, activity_counts.iloc[user_idx], color='red', alpha=0.9, edgecolor='black')
        ax.set_xlabel('Activity Level', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Physical Activity Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col6:
        st.subheader("Weather Condition Distribution")
        weather_counts = df['Weather Condition'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(weather_counts.index, weather_counts.values, color='#e74c3c', alpha=0.7, edgecolor='black')
        # Highlight user's weather condition
        user_idx = list(weather_counts.index).index(weather) if weather in weather_counts.index else -1
        if user_idx >= 0:
            ax.bar(user_idx, weather_counts.iloc[user_idx], color='red', alpha=0.9, edgecolor='black')
        ax.set_xlabel('Weather Condition', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Weather Condition Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ========================
# TAB 2: PREDICTION
# ========================
with tab2:
    st.header("💪 Health Status Prediction")
    
    # Display current inputs
    st.subheader("Your Current Inputs:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Age:** {age} years")
        st.write(f"**Gender:** {gender}")
    
    with col2:
        st.write(f"**Body Weight:** {body_weight} kg")
        st.write(f"**Daily Water Intake:** {daily_water} L")
    
    with col3:
        st.write(f"**Physical Activity:** {activity}")
        st.write(f"**Weather Condition:** {weather}")
    
    # -------------------------
    # PREDICTION
    # -------------------------
    if st.button("🔮 Predict Health Status", key="predict_btn"):
        input_data = np.array([[
            age,
            gender_encoded,
            body_weight,
            daily_water,
            activity_encoded,
            weather_encoded
        ]])

        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)
        
        # Convert prediction to readable text
        status = "🟢 HEALTHY" if prediction[0] == 1 else "🔴 UNHEALTHY"
        confidence = np.max(prob) * 100
        
        # Display results
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.success(f"Prediction: {status}")
            else:
                st.error(f"Prediction: {status}")
        
        with col2:
            st.info(f"Confidence: {confidence:.2f}%")
        
        # Display probability breakdown
        st.subheader("Prediction Details")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Unhealthy Probability", f"{prob[0][0]*100:.2f}%", delta=-confidence)
        with prob_col2:
            st.metric("Healthy Probability", f"{prob[0][1]*100:.2f}%", delta=confidence)

# ========================
# TAB 3: ANALYSIS
# ========================
with tab3:
    st.header("📈 Data Analysis & Insights")
    
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    
    st.subheader("Feature Information")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Healthy Cases: {(df['target'] == 1).sum()} ({(df['target'] == 1).sum()/len(df)*100:.1f}%)")
    st.write(f"Unhealthy Cases: {(df['target'] == 0).sum()} ({(df['target'] == 0).sum()/len(df)*100:.1f}%)")
