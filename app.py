import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/best_model.pkl"

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_pipeline(path: str):
    """Load the trained pipeline once and cache it across reruns.

    Streamlit re-executes this whole script top-to-bottom on every widget
    interaction (every slider move, every button click). Without caching,
    a plain joblib.load() at the top of the file would still hit the disk
    and re-deserialize the pipeline on every single interaction. Wrapping
    it in @st.cache_resource makes Streamlit run the function body exactly
    once for the life of the app session and reuse the same object after
    that -- this is what actually satisfies "load once at startup" in
    Streamlit's execution model.
    """
    return joblib.load(path)


pipeline = load_pipeline(MODEL_PATH)

st.title("🚗 Used Car Price Predictor")
st.write("Enter the car's details below to estimate its resale (selling) price.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input(
        "Present Price (in lakhs)",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        help="The car's current ex-showroom / market price, in lakhs of rupees.",
    )
    kms_driven = st.number_input(
        "Kilometres Driven",
        min_value=0,
        step=100,
    )
    car_age = st.slider(
        "Car Age (years)",
        min_value=1,
        max_value=20,
        value=5,
    )
    owner = st.selectbox(
        "Number of Previous Owners",
        options=[0, 1, 2, 3],
    )

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.divider()

if st.button("Predict Selling Price", type="primary"):
    # Column names here must exactly match the column names the pipeline's
    # ColumnTransformer was fit on during training (Present_Price, Kms_Driven,
    # Car_Age, Fuel_Type, Seller_Type, Transmission, Owner). The pipeline
    # selects columns by name internally, so the order of keys below doesn't
    # matter -- but the spelling and capitalization must match exactly.
    input_df = pd.DataFrame(
        [
            {
                "Present_Price": present_price,
                "Kms_Driven": kms_driven,
                "Car_Age": car_age,
                "Fuel_Type": fuel_type,
                "Seller_Type": seller_type,
                "Transmission": transmission,
                "Owner": owner,
            }
        ]
    )

    try:
        prediction = pipeline.predict(input_df)[0]
        prediction = max(prediction, 0)  # a predicted price should never be negative
        st.success(f"Estimated Selling Price: **{prediction:.2f} Lakhs**")
    except Exception as e:
        st.error(
            "Something went wrong while making the prediction. This usually means "
            "the input columns don't match what the model was trained on.\n\n"
            f"Details: {e}"
        )
