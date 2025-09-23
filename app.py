import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# 0. Load Trained Model
# -----------------------------
with open("MY_project.pkl", "rb") as f:
    model = pickle.load(f)


# -----------------------------
# 1. Collect User Inputs

# -----------------------------
st.title("🥤 Beverage Consumer Prediction App")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Personal Info")
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    zone = st.selectbox("Zone", ["Urban", "Rural", "Semi-Urban"])

with col2:
    st.subheader("Occupation & Income")
    occupation = st.selectbox("Occupation", ["Working Professional", "Student", "Retired"])
    income = st.selectbox("Income Level", ["None", "<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"])
    health = st.selectbox("Health concerns", [
        "Low (Not very concerned)",
        "Medium (Moderately health-conscious)",
        "High (Very health-conscious)"
    ])

with col3:
    st.subheader("Consumption Habits")
    consume_freq = st.selectbox("Consume frequency (weekly)", ["0-2 times", "3-4 times", "5-7 times"])
    current_brand = st.selectbox("Current brand", ["Established", "Newcomer"])
    pref_size = st.selectbox("Preferable consumption size", ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"])
    awareness = st.selectbox("Awareness of other brands", ["0 to 1", "2 to 4", "above 4"])

with col4:
    st.subheader("Preferences & Purchase")
    reasons = st.selectbox("Reasons for choosing brands", ["Price", "Quality", "Brand Reputation"])
    flavor = st.selectbox("Flavor preference", ["Traditional", "Exotic"])
    purchase_channel = st.selectbox("Purchase channel", ["Retail Store", "Online"])
    packaging = st.selectbox("Packaging preference", ["Simple", "Premium"])
    situation = st.selectbox("Typical consumption situations", [
        "Casual (eg. At home)",
        "Social (eg. Parties)",
        "Active (eg. Sports, gym)"
    ])



# -----------------------------
# 2. Feature Engineering
# -----------------------------

# Age group
if 18 <= age <= 25:
    age_group = "18-25"
elif 26 <= age <= 35:
    age_group = "26-35"
elif 36 <= age <= 45:
    age_group = "36-45"
elif 46 <= age <= 55:
    age_group = "46-55"
elif 56 <= age <= 70:
    age_group = "56-70"
else:
    age_group = "70+"

# Mapping for cf_ab_score
cf_map = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
awareness_map = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}
cf_val = cf_map[consume_freq]
aw_val = awareness_map[awareness]
cf_ab_score = cf_val / (cf_val + aw_val)

# Mapping for ZAS
zone_map = {"Rural": 1, "Semi-Urban": 2, "Urban": 3}
income_map = {
    "None": 0, "<10L": 1, "10L - 15L": 2,
    "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5
}
zas_score = zone_map.get(zone, 0) * income_map.get(income, 0)

# Brand Switching Indicator (BSI)
if current_brand != "Established" and reasons in ["Price", "Quality"]:
    bsi = 1
else:
    bsi = 0

# -----------------------------
# 3. Build Input DataFrame
# -----------------------------
input_dict = {
    "age_group": [age_group],
    "income_levels": [income],
    "health_concerns": [health],
    "consume_frequency(weekly)": [consume_freq],
    "preferable_consumption_size": [pref_size],
    "cf_ab_score": [cf_ab_score],
    "zas_score": [zas_score],
    "bsi": [bsi],
    "gender": [gender],
    "zone": [zone],
    "occupation": [occupation],
    "current_brand": [current_brand],
    "awareness_of_other_brands": [awareness],
    "reasons_for_choosing_brands": [reasons],
    "flavor_preference": [flavor],
    "purchase_channel": [purchase_channel],
    "packaging_preference": [packaging],
    "typical_consumption_situations": [situation]
}

df_input = pd.DataFrame(input_dict)

# -----------------------------
# 4. Encoding (Match Training)
# -----------------------------

# Label encoding (ordinal features)
from sklearn.preprocessing import LabelEncoder

label_encode_cols = [
    "age_group", "income_levels", "health_concerns",
    "consume_frequency(weekly)", "preferable_consumption_size"
]
for col in label_encode_cols:
    le = LabelEncoder()
    df_input[col] = le.fit_transform(df_input[col])

# One-hot encoding (categorical features)
one_hot_cols = [
    "gender", "zone", "occupation", "current_brand",
    "awareness_of_other_brands", "reasons_for_choosing_brands",
    "flavor_preference", "purchase_channel", "packaging_preference",
    "typical_consumption_situations"
]
df_input = pd.get_dummies(df_input, columns=one_hot_cols, drop_first=False)

# Ensure same feature order as training
final_features = [
    'income_levels', 'consume_frequency(weekly)', 'preferable_consumption_size',
    'health_concerns', 'age_group', 'cf_ab_score', 'zas_score', 'bsi',
    'gender_M', 'zone_Rural', 'zone_Semi-Urban', 'zone_Urban',
    'occupation_Retired', 'occupation_Student', 'occupation_Working Professional',
    'current_brand_Established ', 'current_brand_Newcomer',
    'awareness_of_other_brands_2 to 4', 'awareness_of_other_brands_above 4',
    'reasons_for_choosing_brands_Brand Reputation',
    'reasons_for_choosing_brands_Price',
    'reasons_for_choosing_brands_Quality',
    'flavor_preference_Traditional', 'purchase_channel_Retail Store',
    'packaging_preference_Premium', 'packaging_preference_Simple',
    'typical_consumption_situations_Casual (eg. At home)',
    'typical_consumption_situations_Social (eg. Parties)'
]

# Reindex to match training
df_input = df_input.reindex(columns=final_features, fill_value=0)

# -----------------------------
# 5. Prediction
# -----------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(df_input)
        st.success(f"✅ Predicted Class: {prediction[0]}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
