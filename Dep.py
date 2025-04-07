import numpy as np
import pickle
import streamlit as st
import os

# 🔹 Load trained model
with open('trained_model_031.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# 🔹 Load scaler
with open('scaler1.sav', 'rb') as file:
    scaler = pickle.load(file)

# 🔹 Convert input data into numeric format
def conversion(a):
    a[0] = 1 if a[0] == "Female" else 0
    a[4] = 1 if a[4] == "Yes" else 0
    a[-1] = 1 if a[-1] == "Yes" else 0
    if a[6] == 'Unhealthy':
        a[6] = 0
    elif a[6] == 'Moderate':
        a[6] = 1
    elif a[6] == 'Healthy':
        a[6] = 2
    else:
        a[6] = 3
    return a

# 🔹 Prediction Function
def depression_prediction(a):
    numeric_data = conversion(a)
    input_data = np.asarray(numeric_data, dtype=float).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = loaded_model.predict(input_data_scaled)

    prediction_msg = {
        1: '⚠️ You may be experiencing depression.',
        0: '✅ You are not showing signs of depression.'
    }

    return prediction_msg.get(prediction[0], "⚠️ Invalid input")

# 🔹 Streamlit App UI
def main():
    st.title("🧠 Depression Detection Web App")

    # Input fields
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', min_value=1, max_value=100)
    Academic_Pressure = st.slider('Academic Pressure (0-5)', 0, 5)
    Study_Satisfaction = st.slider('Study Satisfaction (0-5)', 0, 5)
    Suicidal_Thoughts = st.selectbox('Have you ever had suicidal thoughts?', ['Yes', 'No'])
    Study_Hours = st.number_input('Work/Study Hours per day', min_value=0, max_value=24)
    Dietary_Habits = st.selectbox('Dietary Habits', ['Unhealthy', 'Moderate', 'Healthy', 'Others'])
    Financial_Stress = st.slider('Financial Stress (0-5)', 0, 5)
    Family_History = st.selectbox('Family History of Mental Illness', ['Yes', 'No'])

    # Button
    if st.button('🧪 Detect Depression Level'):
        user_input = [
            Gender, Age, Academic_Pressure,
            Study_Satisfaction, Suicidal_Thoughts,
            Study_Hours, Dietary_Habits, Financial_Stress,
            Family_History
        ]
        result = depression_prediction(user_input)
        st.subheader("🧾 Result:")
        st.success(result)

        if result.startswith("⚠️"):
            st.markdown("Here are some resources you can explore:")
            st.markdown("- [💚 iCall - Mental Health Helpline (TISS)](https://icallhelpline.org/)")
            st.markdown("- [🧠 MindPeers Mental Health Platform](https://www.mindpeers.co/)")
            st.markdown("- [📞 Vandrevala Foundation Mental Health Helpline](https://www.vandrevalafoundation.com/helpline)")
            st.markdown("- [🧑‍⚕️ Find a Therapist in India (Psychology Today)](https://www.psychologytoday.com/intl/counsellors/india)")
            
            st.markdown("### 🧘‍♀️ Self-Help Tips:")
            st.markdown("- Practice mindfulness or deep breathing daily")
            st.markdown("- Stay connected with friends/family")
            st.markdown("- Don’t hesitate to talk to a counselor")
            st.markdown("- Regular physical activity can boost mood")
        else:
            st.markdown("### 🌟 Awesome! You're doing great.")
            st.markdown("Here are some tips to maintain mental wellness:")
            st.markdown("- 🚶 Take short breaks while studying or working")
            st.markdown("- 📒 Maintain a gratitude journal")
            st.markdown("- 🌿 Try meditation or yoga to stay mentally fit")
            st.markdown("- 👥 Help others who might be struggling")


if __name__ == '__main__':
    main()
