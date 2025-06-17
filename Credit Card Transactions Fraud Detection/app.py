import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# --- App Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Model and Feature Information ---
try:
    model = joblib.load('Credit_Card_Fraud_Detection.pkl')
    model_columns = joblib.load('Credit_Card_Fraud_Detection_Features.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run the Jupyter Notebook to generate 'Credit_Card_Fraud_Detection.pkl' and 'Credit_Card_Fraud_Detection_Features.pkl'.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model files: {e}")
    st.stop()


# --- Caching Function to Load Data and Prepare Encoders ---
@st.cache_data
def load_data_and_encoders(data_path='fraud-detection/fraudTrain.csv'):
    """
    Loads the training data to extract unique values for categorical features
    and creates fitted label encoders.
    """
    try:
        df = pd.read_csv(data_path)
        
        # Extract unique values for dropdowns
        merchant_options = df['merchant'].unique()
        category_options = df['category'].unique()
        gender_options = df['gender'].unique()
        job_options = df['job'].unique()

        # Create and fit label encoders
        merchant_encoder = LabelEncoder().fit(df['merchant'])
        category_encoder = LabelEncoder().fit(df['category'])
        gender_encoder = LabelEncoder().fit(df['gender'])
        job_encoder = LabelEncoder().fit(df['job'])
        
        return {
            "merchant": (merchant_options, merchant_encoder),
            "category": (category_options, category_encoder),
            "gender": (gender_options, gender_encoder),
            "job": (job_options, job_encoder)
        }
    except FileNotFoundError:
        st.error(f"Training data not found at '{data_path}'. Please ensure the dataset is downloaded and in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None

# Load the encoders and options
encoders_data = load_data_and_encoders()
if encoders_data is None:
    st.stop()

# --- User Interface ---
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application uses a pre-trained **Support Vector Classifier (SVC)** model to predict whether a credit card transaction is fraudulent. 
Enter the transaction details in the sidebar to get a prediction.
""")

st.sidebar.header("Transaction Details")

def get_user_input():
    """
    Creates sidebar inputs and returns a dictionary of user-provided values.
    """
    merchant = st.sidebar.selectbox("Merchant", encoders_data['merchant'][0])
    category = st.sidebar.selectbox("Category", encoders_data['category'][0])
    amt = st.sidebar.number_input("Amount (USD)", min_value=0.0, value=100.0, step=1.0)
    gender = st.sidebar.selectbox("Gender", encoders_data['gender'][0])
    lat = st.sidebar.number_input("Latitude", value=40.7128)
    long = st.sidebar.number_input("Longitude", value=-74.0060)
    city_pop = st.sidebar.number_input("City Population", min_value=0, value=8419000)
    job = st.sidebar.selectbox("Job", encoders_data['job'][0])
    unix_time = st.sidebar.number_input("Unix Time", value=1371816865, min_value=0)
    merch_lat = st.sidebar.number_input("Merchant Latitude", value=40.7128)
    merch_long = st.sidebar.number_input("Merchant Longitude", value=-74.0060)
    
    data = {
        'merchant': merchant,
        'category': category,
        'amt': amt,
        'gender': gender,
        'lat': lat,
        'long': long,
        'city_pop': city_pop,
        'job': job,
        'unix_time': unix_time,
        'merch_lat': merch_lat,
        'merch_long': merch_long
    }
    return data

user_data = get_user_input()

# --- Prediction Logic ---
st.header("Prediction")

if st.button("Predict Fraud Status"):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_data])

    # Encode categorical features
    try:
        input_df['merchant'] = encoders_data['merchant'][1].transform(input_df['merchant'])
        input_df['category'] = encoders_data['category'][1].transform(input_df['category'])
        input_df['gender'] = encoders_data['gender'][1].transform(input_df['gender'])
        input_df['job'] = encoders_data['job'][1].transform(input_df['job'])
    except Exception as e:
        st.error(f"Error during data encoding: {e}")
        st.stop()

    # Ensure columns are in the same order as the model was trained on
    input_df = input_df[model_columns]

    # Make prediction
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.decision_function(input_df)

        st.subheader("Result:")
        if prediction[0] == 1:
            st.error("🚨 This transaction is likely **Fraudulent**.")
        else:
            st.success("✅ This transaction appears to be **Legitimate**.")

        # Display decision function score
        st.write("---")
        st.write(f"**Model Confidence Score:** `{prediction_proba[0]:.4f}`")
        st.info("The confidence score represents the signed distance to the hyperplane. A negative score corresponds to a legitimate prediction, while a positive score indicates fraud.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
