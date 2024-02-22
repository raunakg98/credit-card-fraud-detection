import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('rf_credit_card_fraud_detection_model.joblib')
    return model

model = load_model()

# Initialize a StandardScaler
# NOTE: Ideally, you should save and load the scaler used during training instead of creating a new one.
scaler = StandardScaler()

# App title and introduction
st.title('Credit Card Fraud Detection App')
st.write('''
This app uses machine learning to predict fraudulent credit card transactions.
The model was trained on a dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Features `V1` to `V28` are the result of a PCA transformation due to confidentiality issues with the original data.
''')

# Sidebar for user input
st.sidebar.header('Transaction Details')

# Function to generate random inputs for all features
def generate_random_features():
    features = {}
    # Random generation for scaled_amount and scaled_time
    features['scaled_amount'] = np.random.uniform(-3, 3)  # Assuming scaled values range
    features['scaled_time'] = np.random.uniform(-3, 3)  # Assuming scaled values range
    # Random generation for V1 to V28
    for i in range(1, 29):
        features[f'V{i}'] = np.random.uniform(-3, 3)  # Assuming scaled values range
    return pd.DataFrame([features])

# Generate random inputs button
if st.sidebar.button('Generate Random Values'):
    input_df = generate_random_features()
else:
    # Manual input for scaled_amount and scaled_time
    scaled_amount = st.sidebar.number_input('Scaled Amount', value=0.0, format="%.2f")
    scaled_time = st.sidebar.number_input('Scaled Time', value=0.0, format="%.2f")
    # Sliders for V features
    v_features = {}
    for i in range(1, 29):
        v_features[f'V{i}'] = st.sidebar.slider(f'V{i}', -3.0, 3.0, 0.0)  # Assuming scaled values range
    v_features['scaled_amount'] = scaled_amount
    v_features['scaled_time'] = scaled_time
    input_df = pd.DataFrame([v_features])

# Display the input DataFrame
st.subheader('User Input Features')
st.write(input_df)

# Predict Button
if st.button('Predict'):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.subheader('Prediction')
    fraud_prob = probability[0][1]
    st.write('Fraudulent' if prediction[0] == 1 else 'Legitimate')
    st.write(f'Probability of Fraud: {fraud_prob:.4f}')

    if fraud_prob > 0.5:
        st.error('This transaction is likely to be fraudulent.')
    else:
        st.success('This transaction is likely to be legitimate.')

# Project Details and Steps Taken
st.write('''
## Project Context and Steps

- The project aims to detect potentially fraudulent transactions using a machine learning model.
- The dataset contains transactions made by credit cards in September 2013 by European cardholders.
- Features `V1` to `V28` are principal components obtained with PCA; `Time` and `Amount` are the only features not transformed.
- We addressed class imbalance using SMOTE and compared different models based on accuracy.
- The final model was chosen based on its ability to accurately predict fraudulent transactions.
''')
