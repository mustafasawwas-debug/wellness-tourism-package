import streamlit as st
import pandas as pd
import joblib
import os

# Set page title and styling
st.set_page_config(page_title="Wellness Tourism Package Prediction", layout="centered")

st.title("🌴 Wellness Tourism Package Predictor")
st.write("""
This application predicts whether a customer is likely to purchase the newly introduced **Wellness Tourism Package** 
based on their demographic information and historical interactions.
""")

# Resolve model file path
model_path = "tourism_project/model_building/best_xgb_model.pkl"

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model(model_path)

if model is None:
    st.error(f"Model file not found at '{model_path}'. Please ensure the model is trained and saved correctly.")
else:
    st.success("Model pipeline loaded successfully!")

    # Organize inputs into columns
    st.subheader("Customer Demographics & Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        city_tier = st.selectbox("City Tier (1 = High, 3 = Low)", options=[1, 2, 3], index=0)
        occupation = st.selectbox("Occupation", options=["Salaried", "Small Business", "Large Business", "Freelancer"], index=0)
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Unmarried"], index=1)
        designation = st.selectbox("Designation", options=["Executive", "Manager", "Senior Manager", "AVP", "VP"], index=0)
        
    with col2:
        passport = st.selectbox("Has Valid Passport?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        own_car = st.selectbox("Owns a Car?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=3, step=1)
        monthly_income = st.number_input("Monthly Income (INR)", min_value=0, value=25000, step=1000)
        num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
        num_children_visiting = st.number_input("Number of Children Visiting (< Age 5)", min_value=0, max_value=5, value=0, step=1)

    st.subheader("Customer Interaction Metrics")
    col3, col4 = st.columns(2)
    
    with col3:
        duration_of_pitch = st.number_input("Duration of Pitch (Minutes)", min_value=0, max_value=120, value=15, step=1)
        num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
        
    with col4:
        preferred_property_star = st.slider("Preferred Hotel Rating (Stars)", min_value=1, max_value=5, value=3, step=1)
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3, step=1)
        
    # Categorical drop downs that require mapping
    typeof_contact = st.selectbox("Type of Contact", options=["Self Inquiry", "Company Invited"], index=0)
    product_pitched = st.selectbox("Product Pitched", options=["Basic", "Deluxe", "Standard", "Super Deluxe", "King"], index=0)

    # Predict button
    if st.button("🔮 Predict Purchase Likelihood"):
        # Construct input DataFrame matching raw feature schema
        input_data = pd.DataFrame([{
            "Age": age,
            "TypeofContact": typeof_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_person_visiting,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": num_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income
        }])
        
        # Execute Prediction
        try:
            prediction_proba = model.predict_proba(input_data)[0][1]
            prediction = int(prediction_proba >= 0.5)
            
            st.write("--- ")
            if prediction == 1:
                st.error(f"🎯 **Prediction: High Likelihood to Purchase (Probability: {prediction_proba:.2%})**")
                st.info("Action item: Prioritize targeting this customer for the Wellness Tourism Package campaign.")
            else:
                st.success(f"❌ **Prediction: Low Likelihood to Purchase (Probability: {prediction_proba:.2%})**")
                st.info("Action item: Refrain from active aggressive pitches to maximize conversion efficiency.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
