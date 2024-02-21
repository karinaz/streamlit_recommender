import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading articles_cleaned dataset
articles_cleaned = pd.read_csv('articles_cleaned.csv')

# Define the function for finding similar articles
def find_similar_articles_html(user_input, articles_cleaned, combined_features_column='combined_features', top_n=3):
    # Convert the combined features column to a list and append the user input
    features_list = articles_cleaned[combined_features_column].tolist()
    features_list.append(user_input)

    # Apply CountVectorizer
    cv_char = CountVectorizer(analyzer='word', ngram_range=(1, 1), stop_words='english')
    count_matrix = cv_char.fit_transform(features_list)
    # Calculate cosine similarity for CountVectorizer
    cosine_sim = cosine_similarity(count_matrix)
    
    # Apply TfidfVectorizer
#    tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2))
#    X_tfidf = tfidf_vectorizer.fit_transform(features_list)

    # Calculate cosine similarity for TfidfVectorizer
#    cosine_sim = cosine_similarity(X_tfidf)

    # Get the top N similar articles, excluding the last one (which is the user input)
    sorted_suggested_articles = sorted(list(enumerate(cosine_sim[-1])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    suggested_ids = [i[0] for i in sorted_suggested_articles]

    # Retrieve the articles from the original dataset
    selected_articles = articles_cleaned.loc[suggested_ids]

    # Optionally, create a shortened 'content_preview' if needed
    selected_articles['content_preview'] = selected_articles['content'].apply(lambda x: x[:100] if isinstance(x, str) else '')



    # Create an HTML outputs
    html_output = '<div>'
    for _, row in selected_articles.iterrows():
        # Split the content to separate the author's name
        content_parts = row['content'].split('  ', 1)  # Assumes the author's name and content are separated by two spaces
        author_name = content_parts[0] if len(content_parts) > 1 else ''
        article_content = content_parts[1] if len(content_parts) > 1 else row['content']

        # Generating HTML output with author's name in italic
        html_output += f'<div style="margin-bottom: 20px;">'
        html_output += f'<h3>{row["title"]}</h3>'
        html_output += f'<p><i>{author_name}</i></p>'
        html_output += f'<p>{article_content[:1000]}... <a href="{row["link"]}" target="_blank">Read more</a></p>'
        html_output += '</div>'

    html_output += '</div>'
    return html_output

# Streamlit app
def main():
    st.title("Guide on mental health")

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
