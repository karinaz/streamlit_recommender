import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading articles_cleaned dataset
articles_cleaned = pd.read_csv('articles_cleaned.csv')

# Define the function for finding similar articles
def find_similar_articles_html(user_input, articles_cleaned, combined_features_column='combined_features', top_n=3):
    # ... existing code ...

# Streamlit app
def main():
    st.title("Guide on Mental Health")

    # Sidebar for user input
    with st.sidebar:
        user_input = st.text_input("How do you feel? What are you worried about? Which topic you want to explore? Find your answer here:")

    # Main page for displaying results
    if user_input:
        with st.spinner('Finding similar articles...'):
            html_content = find_similar_articles_html(user_input, articles_cleaned)
            st.markdown(html_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
