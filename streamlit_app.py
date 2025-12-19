import streamlit as st
import pickle
import numpy as np
from threading import Thread
from flask import Flask, request, jsonify
import time

# Load model
model = pickle.load(open('wine_model.pkl', 'rb'))

# Flask app (background)
flask_app = Flask(__name__)

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probability.tolist(),
            'message': f'Wine type predicted: {prediction}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Streamlit UI
st.set_page_config(page_title="Wine Type Predictor", layout="wide")
st.title("🍷 Wine Type Predictor")
st.write("Enter wine chemical properties to predict its type")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.slider("Alcohol %", 10.0, 15.0, 14.0)
    malic_acid = st.slider("Malic Acid", 0.0, 6.0, 1.71)
    ash = st.slider("Ash", 1.0, 4.0, 2.43)
    alcalinity = st.slider("Alcalinity of Ash", 10.0, 30.0, 15.6)
    magnesium = st.slider("Magnesium", 70.0, 162.0, 127.0)
    phenols = st.slider("Total Phenols", 0.98, 3.88, 2.8)
    flavanoids = st.slider("Flavanoids", 0.34, 5.08, 3.06)

with col2:
    nonflavanoid = st.slider("Nonflavanoid Phenols", 0.13, 0.66, 0.28)
    proanthocyanins = st.slider("Proanthocyanins", 0.41, 3.58, 2.29)
    color_intensity = st.slider("Color Intensity", 1.3, 13.0, 5.64)
    hue = st.slider("Hue", 0.48, 1.71, 1.04)
    od280 = st.slider("OD280/OD315", 1.27, 4.0, 3.92)
    proline = st.slider("Proline", 278.0, 1680.0, 1065.0)

st.markdown("---")

if st.button("🔮 Predict Wine Type", use_container_width=True):
    features = [alcohol, malic_acid, ash, alcalinity, magnesium, phenols, flavanoids, 
                nonflavanoid, proanthocyanins, color_intensity, hue, od280, proline]
    
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    st.success("✅ Prediction Complete!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Wine Type", prediction)
    with col2:
        st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
    with col3:
        st.metric("Model Status", "Active")
    
    st.write("**Probability Distribution:**")
    prob_data = {
        "Type 0": probabilities[0],
        "Type 1": probabilities[1],
        "Type 2": probabilities[2]
    }
    st.bar_chart(prob_data)
