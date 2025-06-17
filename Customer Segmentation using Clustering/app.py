import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# Set page configuration
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# --- App Title and Description ---
st.title("Customer Segmentation App")
st.markdown("""
This application performs customer segmentation using a K-Means clustering model. 
Upload your customer data in Excel format to see the segmentation in action.
The analysis is based on the 'Online Retail' dataset.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")

        # --- Data Exploration ---
        st.header("Exploratory Data Analysis")
        st.subheader("Raw Data")
        st.dataframe(df.head())

        st.subheader("Data Information")
        # Capture df.info() output
        from io import StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Data Cleaning and Preprocessing")
        
        # Handling duplicates
        st.write(f"Number of duplicate rows before cleaning: {df.duplicated().sum()}")
        df.drop_duplicates(inplace=True)
        st.write(f"Number of duplicate rows after cleaning: {df.duplicated().sum()}")

        # Handling missing values
        st.write(f"Number of missing values before cleaning: \n{df.isnull().sum()}")
        df.dropna(inplace=True)
        st.write(f"Number of missing values after cleaning: \n{df.isnull().sum()}")
        
        # Handling outliers using IQR
        st.write("Removing outliers from 'Quantity' and 'UnitPrice' using the IQR method.")
        
        # Outliers for Quantity
        q1_qty = df["Quantity"].quantile(0.30)
        q3_qty = df["Quantity"].quantile(0.70)
        iqr_qty = q3_qty - q1_qty
        upper_limit_qty = q3_qty + (1.5 * iqr_qty)
        lower_limit_qty = q1_qty - (1.5 * iqr_qty)
        df = df.loc[(df["Quantity"] < upper_limit_qty) & (df["Quantity"] > lower_limit_qty)]

        # Outliers for UnitPrice
        q1_price = df["UnitPrice"].quantile(0.25)
        q3_price = df["UnitPrice"].quantile(0.65)
        iqr_price = q3_price - q1_price
        upper_limit_price = q3_price + (1.5 * iqr_price)
        lower_limit_price = q1_price - (1.5 * iqr_price)
        df = df.loc[(df["UnitPrice"] < upper_limit_price) & (df["UnitPrice"] > lower_limit_price)]
        
        st.subheader("Preprocessed Data")
        st.dataframe(df.head())
        
        # --- Feature Engineering and Clustering ---
        st.header("Customer Segmentation")
        
        # Selecting features and encoding
        X = df[["Quantity", "UnitPrice", "Country"]].copy()
        encoder = LabelEncoder()
        X["Country"] = encoder.fit_transform(X["Country"])
        
        # Creating total_price feature
        X["total_price"] = X["Quantity"] * X["UnitPrice"]
        
        # Scaling features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=5, random_state=20, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        st.subheader("Cluster Distribution")
        st.write(df['Cluster'].value_counts())
        
        # --- Visualizations ---
        st.header("Visualizations")
        
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr = X.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="icefire", ax=ax_corr)
        st.pyplot(fig_corr)
        
        # PCA Visualization
        st.subheader("PCA-Based Cluster Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]
        
        fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette='icefire', s=60, edgecolor='k', ax=ax_pca)
        ax_pca.set_title("PCA-Based Cluster Visualization")
        st.pyplot(fig_pca)

        # Bar plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Countries Frequency")
            country_count = df.Country.value_counts().head(10)
            fig_country, ax_country = plt.subplots()
            sns.barplot(x=country_count.values, y=country_count.index, ax=ax_country)
            ax_country.set_title("Top 10 Countries by Order Frequency")
            ax_country.set_xlabel("Number of Orders")
            ax_country.set_ylabel("Country")
            st.pyplot(fig_country)

        with col2:
            st.subheader("Frequent Items")
            description_count = df["Description"].value_counts().head(10)
            fig_desc, ax_desc = plt.subplots()
            sns.barplot(x=description_count.values, y=description_count.index, ax=ax_desc)
            ax_desc.set_title("Top 10 Frequent Items")
            ax_desc.set_xlabel("Number of Orders")
            ax_desc.set_ylabel("Items")
            st.pyplot(fig_desc)

    except Exception as e:
        st.error(f"An error occurred: {e}")
