# Credit Card Fraud Detection App

This Streamlit app is designed to predict fraudulent transactions using machine learning. It's built on a dataset that includes credit card transactions made by European cardholders in September 2013.

## Live App

You can access the live app here: [Credit Card Fraud Detection App](https://credit-card-fraud-detection-app.streamlit.app/)

## Features

- **Predictive Modeling**: Utilizes a RandomForestClassifier trained on a comprehensive dataset from Kaggle.
- **Interactive User Interface**: Allows users to input transaction details and receive instant predictions on whether a transaction is likely to be fraudulent.
- **Data Visualization**: Displays user inputs and prediction probabilities in a clear and engaging manner.

## How It Works

1. **Data Input**: Use the sidebar to enter transaction details. You can manually input the `scaled_amount` and `scaled_time`, along with the PCA-transformed features (`V1` to `V28`), or generate random values for all features.
2. **Prediction**: Click the 'Predict' button to evaluate the transaction. The app will display whether the transaction is likely to be fraudulent, along with the probability of fraud.
3. **Visualization**: The main panel displays the input features and the prediction result, offering insights into the model's decision-making process.

## Project Context

The app is based on a dataset containing transactions made by credit card holders in September 2013. To maintain confidentiality, the original features have been transformed using PCA, resulting in features named `V1` to `V28`. Only `Time` and `Amount` have been retained in their original form, although they are scaled before use in the model.

## Getting Started

To run the app locally:

1. Clone the repository:
