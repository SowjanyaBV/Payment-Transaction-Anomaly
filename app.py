import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load dataset, trained model, scaler and feature names
# --------------------------------------------------

dataset = pd.read_csv("data/archive (1)/creditcard.csv")

model = joblib.load("models/decision_tree_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Payment Transaction Anomaly Detection",
    page_icon="💳"
)


# --------------------------------------------------
# Dashboard title
# --------------------------------------------------

st.title("💳 Payment Transaction Anomaly Detection")

st.write(
    "Classify a payment transaction as Routine/Legitimate "
    "or Anomalous/Fraudulent."
)


# --------------------------------------------------
# Select transaction type
# --------------------------------------------------

st.subheader("Transaction Testing")

transaction_type = st.selectbox(
    "Select how you want to test the transaction:",
    [
        "Manual Input",
        "Real Legitimate Transaction",
        "Real Fraudulent Transaction"
    ]
)


# --------------------------------------------------
# Create input data
# --------------------------------------------------

input_data = {}


if transaction_type == "Manual Input":

    st.subheader("Transaction Details")

    st.write(
        "Enter the transaction values below and click "
        "'Predict Transaction'."
    )

    for feature in feature_names:

        input_data[feature] = st.number_input(
            f"{feature}",
            value=0.0
        )


else:

    # Select a real transaction from the dataset

    if transaction_type == "Real Fraudulent Transaction":

        sample = dataset[dataset["Class"] == 1].iloc[0]

        st.warning(
            "A real fraudulent transaction has been loaded "
            "from the dataset."
        )

    else:

        sample = dataset[dataset["Class"] == 0].iloc[0]

        st.info(
            "A real legitimate transaction has been loaded "
            "from the dataset."
        )


    # Store the transaction features

    for feature in feature_names:

        input_data[feature] = sample[feature]


    # Display the loaded transaction

    st.subheader("Loaded Transaction Details")

    input_df_display = pd.DataFrame([input_data])

    st.dataframe(
        input_df_display,
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Transaction"):

    # Validate input values

    if any(pd.isna(value) for value in input_data.values()):

        st.warning(
            "Please enter valid values for all transaction features."
        )

    else:

        # Convert input into DataFrame

        input_df = pd.DataFrame([input_data])


        # Keep exactly the same feature order
        # used during model training

        input_df = input_df[feature_names]


        # Apply the same scaler used during training

        input_scaled = scaler.transform(input_df)


        # Make prediction

        prediction = model.predict(input_scaled)[0]


        # --------------------------------------------------
        # Display prediction
        # --------------------------------------------------

        st.subheader("Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ Anomalous / Fraudulent Transaction"
            )

        else:

            st.success(
                "✅ Routine / Legitimate Transaction"
            )