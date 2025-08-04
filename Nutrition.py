import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt

#Load model and encoder
@st.cache_resource
def load_assets():
    model = joblib.load("nutrition_predictor_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_assets()
food_labels = encoder.categories_[0]

# Page settings
st.set_page_config(page_title="Nutrition Predictor", page_icon="🍎", layout="centered")
st.markdown("<h1 style='text-align: center;'>🍽️ Smart Nutrition Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict nutritional facts based on food and weight</p>", unsafe_allow_html=True)
st.markdown("---")

#UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    food = st.selectbox("Choose a food item:", food_labels)
    weight = st.slider("Select weight (grams):", min_value=10, max_value=1000, value=100, step=10)
    predict_btn = st.button("🍽️ Predict Nutrition")

with col2:
    if predict_btn:
        encoded_label = encoder.transform([[food]])
        input_features = np.hstack((encoded_label, [[weight]]))
        prediction = model.predict(input_features)[0]

        nutrients = ["Calories", "Protein", "Carbohydrates", "Fats", "Fiber", "Sugars", "Sodium"]
        values = np.round(prediction, 2)

        df_result = pd.DataFrame({
            "Nutrient": nutrients,
            "Value": values,
            "Unit": ["kcal", "g", "g", "g", "g", "g", "mg"]
        })

        st.subheader(f"📊 Nutritional Content for {weight}g of {food}")
        st.dataframe(df_result, use_container_width=True)

        #Chart
        st.subheader("Nutrient Breakdown")
        chart = alt.Chart(df_result).mark_bar(color="#4CAF50").encode(
            x=alt.X("Nutrient", sort=None),
            y="Value"
        ).properties(width="container", height=300)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("**ℹ️ Note:** Nutritional values are general estimates and may vary based on ingredients, preparation methods, and recipe variations.")
