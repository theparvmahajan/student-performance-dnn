
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load models and preprocessing objects
@st.cache_resource
def load_models():

    early_model = tf.keras.models.load_model(
        MODEL_DIR / "dnn_early.keras"
    )

    late_model = tf.keras.models.load_model(
        MODEL_DIR / "dnn_late.keras"
    )

    early_preprocessor = joblib.load(
        MODEL_DIR / "preprocessor_early.pkl"
    )

    late_preprocessor = joblib.load(
        MODEL_DIR / "preprocessor_late.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    return (
        early_model,
        late_model,
        early_preprocessor,
        late_preprocessor,
        label_encoder
    )


# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Academic Performance Prediction")

st.write(
    "Predict student academic performance using "
    "a Deep Neural Network."
)

st.info(
    "This application supports early and late academic "
    "performance prediction."
)

# Select prediction scenario
scenario = st.selectbox(
    "Select Prediction Scenario",
    ["Early Prediction", "Late Prediction"]
)

st.write(
    "Early prediction excludes previous grades (G1 and G2)."
    if scenario == "Early Prediction"
    else "Late prediction includes previous grades (G1 and G2)."
)


# Categorical input options
categorical_options = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["GT3", "LE3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["course", "home", "reputation", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"]
}


# Numerical input ranges: minimum, maximum, default
numerical_fields = {
    "age": (15, 22, 17),
    "Medu": (0, 4, 2),
    "Fedu": (0, 4, 2),
    "traveltime": (1, 4, 1),
    "studytime": (1, 4, 2),
    "failures": (0, 3, 0),
    "famrel": (1, 5, 4),
    "freetime": (1, 5, 3),
    "goout": (1, 5, 3),
    "Dalc": (1, 5, 1),
    "Walc": (1, 5, 1),
    "health": (1, 5, 3),
    "absences": (0, 93, 0),
    "G1": (0, 20, 10),
    "G2": (0, 20, 10)
}


# Input form
with st.form("student_form"):

    st.subheader("Student Information")

    input_data = {}

    st.markdown("### Personal and Family Information")

    col1, col2 = st.columns(2)

    for i, (feature, options) in enumerate(
        categorical_options.items()
    ):

        with [col1, col2][i % 2]:

            input_data[feature] = st.selectbox(
                feature,
                options
            )

    st.markdown("### Academic and Lifestyle Information")

    col1, col2 = st.columns(2)

    numerical_features = list(numerical_fields.keys())

    if scenario == "Early Prediction":
        numerical_features.remove("G1")
        numerical_features.remove("G2")

    for i, feature in enumerate(numerical_features):

        minimum, maximum, default = numerical_fields[feature]

        with [col1, col2][i % 2]:

            input_data[feature] = st.number_input(
                feature,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=1
            )

    submitted = st.form_submit_button(
        "Predict Student Performance"
    )


# Prediction
if submitted:

    try:

        (
            early_model,
            late_model,
            early_preprocessor,
            late_preprocessor,
            label_encoder
        ) = load_models()

        input_df = pd.DataFrame([input_data])

        if scenario == "Early Prediction":

            processed_input = early_preprocessor.transform(
                input_df
            )

            prediction = early_model.predict(
                processed_input,
                verbose=0
            )

        else:

            processed_input = late_preprocessor.transform(
                input_df
            )

            prediction = late_model.predict(
                processed_input,
                verbose=0
            )

        predicted_class = np.argmax(prediction, axis=1)

        predicted_label = label_encoder.inverse_transform(
            predicted_class
        )[0]

        st.success("Prediction completed!")

        st.subheader("Predicted Academic Performance")

        st.title(predicted_label)

    except Exception as error:

        st.error(f"Prediction failed: {error}")
