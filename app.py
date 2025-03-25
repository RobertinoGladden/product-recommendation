import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset dan model
df = pd.read_csv("data.csv", encoding="latin1")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("nearest_neighbors.pkl")

# Data Cleaning
df = df.dropna(subset=['Description'])
df = df.drop_duplicates()
df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x)) 
df['Description'] = df['Description'].str.lower()
df = df[df['Description'].str.len() > 2]
df = df.drop_duplicates(subset=['StockCode'])

# Fungsi pencarian
def search_by_keyword(df, keyword):
    keyword = keyword.lower()
    return df[df['Description'].str.contains(keyword, regex=False)]

# Fungsi rekomendasi
def recommend_products(keyword, num_recommendations=20):
    filtered_df = search_by_keyword(df, keyword)
    
    if filtered_df.empty:
        query_vector = vectorizer.transform([keyword])  
    else:
        filtered_vectors = vectorizer.transform(filtered_df['Description'])
        if filtered_vectors.shape[0] == 0 or filtered_vectors.sum() == 0:
            return None
        query_vector = np.asarray(filtered_vectors.mean(axis=0))

    distances, indices = model.kneighbors(query_vector, n_neighbors=min(num_recommendations, len(df)))
    indices = [i for i in indices[0] if i < len(df)]
    if not indices:
        return None
    return df.iloc[indices][['StockCode', 'Description']]

# Streamlit UI
st.set_page_config(page_title="Shoope", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>SHOOPE: Smart AI Shopping</h1>", unsafe_allow_html=True)
st.write("<h4 style='text-align: center; color: #FFFFFF;'>Find the Best Products with AI Intelligence</h4>", unsafe_allow_html=True)
st.write("---")

# Input di menu utama
product_input = st.text_input("Enter Product Keyword", placeholder="Type a product...")
search_button = st.button("Get Recommendations")

if search_button:
    with st.spinner("🔄 Searching for the best recommendations..."):
        time.sleep(1.5)
        recommendations = recommend_products(product_input, num_recommendations=20)

        if recommendations is None:
            st.error("❌ No matching products found. Try another keyword.")
        else:
            st.success("✅ Products found! Here are your recommendations:")
            st.write("### Recommended Products")
            st.dataframe(recommendations, use_container_width=True)

# Professional Copywriting for the Website
st.write("---")