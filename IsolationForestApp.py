import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Anomaly Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("isolation_forest.pkl")
    return model


model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("💳 Financial Anomaly Detection System")

st.write(
    "Enter transaction information below to determine whether "
    "the transaction is normal or anomalous."
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Transaction Information")

col1, col2 = st.columns(2)

with col1:

    #time = st.number_input(
    #   "Time",
     #   min_value=0.0,
      #  value=0.0
    #)

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)
    v7 = st.number_input("V7", value=0.0)
    v8 = st.number_input("V8", value=0.0)
    v9 = st.number_input("V9", value=0.0)
    v10 = st.number_input("V10", value=0.0)
    v11 = st.number_input("V11", value=0.0)
    v12 = st.number_input("V12", value=0.0)
    v13 = st.number_input("V13", value=0.0)

with col2:

    v14 = st.number_input("V14", value=0.0)
    v15 = st.number_input("V15", value=0.0)
    v16 = st.number_input("V16", value=0.0)
    v17 = st.number_input("V17", value=0.0)
    v18 = st.number_input("V18", value=0.0)
    v19 = st.number_input("V19", value=0.0)
    v20 = st.number_input("V20", value=0.0)
    v21 = st.number_input("V21", value=0.0)
    v22 = st.number_input("V22", value=0.0)
    v23 = st.number_input("V23", value=0.0)
    v24 = st.number_input("V24", value=0.0)
    v25 = st.number_input("V25", value=0.0)
    v26 = st.number_input("V26", value=0.0)
    v27 = st.number_input("V27", value=0.0)
    v28 = st.number_input("V28", value=0.0)


# --------------------------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------------------------

input_data = pd.DataFrame({

    "V1": [v1],
    "V2": [v2],
    "V3": [v3],
    "V4": [v4],
    "V5": [v5],
    "V6": [v6],
    "V7": [v7],
    "V8": [v8],
    "V9": [v9],
    "V10": [v10],
    "V11": [v11],
    "V12": [v12],
    "V13": [v13],
    "V14": [v14],
    "V15": [v15],
    "V16": [v16],
    "V17": [v17],
    "V18": [v18],
    "V19": [v19],
    "V20": [v20],
    "V21": [v21],
    "V22": [v22],
    "V23": [v23],
    "V24": [v24],
    "V25": [v25],
    "V26": [v26],
    "V27": [v27],
    "V28": [v28],

    "Amount": [amount]
})


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button("🔍 Detect Anomaly", use_container_width=True):

    try:

        prediction = model.predict(input_data)

        # Isolation Forest:
        #  1  = Normal
        # -1  = Anomaly


        if prediction[0] == -1:

            st.error(
                "⚠️ ANOMALY DETECTED"
            )

            st.warning(
                "This transaction appears to be anomalous."
            )

        else:

            st.success(
                "✅ NORMAL TRANSACTION"
            )

            st.info(
                "This transaction appears to be normal."
            )


        # --------------------------------------------------
        # ANOMALY SCORE
        # --------------------------------------------------

        if hasattr(model, "decision_function"):

            score = model.decision_function(input_data)[0]

            st.subheader("Anomaly Score")

            st.metric(
                label="Score",
                value=f"{score:.6f}"
            )

            if score < 0:
                st.write(
                    "The score is below zero, indicating an anomalous transaction."
                )
            else:
                st.write(
                    "The score is above zero, indicating a normal transaction."
                )


        # --------------------------------------------------
        # SHOW INPUT
        # --------------------------------------------------

        with st.expander("View Transaction Data"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )