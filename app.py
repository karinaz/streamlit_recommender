import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Loading articles_cleaned dataset
articles_cleaned = pd.read_csv('articles_cleaned.csv')

# Define the function for finding similar articles
def find_similar_articles_html(user_input, articles_cleaned, combined_features_column='combined_features', top_n=3):
    # Convert the combined features column to a list and append the user input
    features_list = articles_cleaned[combined_features_column].tolist()
    features_list.append(user_input)

    # Apply CountVectorizer
    cv_char = CountVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
    count_matrix = cv_char.fit_transform(features_list)
    # Calculate cosine similarity for CountVectorizer
    cosine_sim = cosine_similarity(count_matrix)

    # Get the top N similar articles, excluding the last one (which is the user input)
    sorted_suggested_articles = sorted(list(enumerate(cosine_sim[-1])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    suggested_ids = [i[0] for i in sorted_suggested_articles]

    # Retrieve the articles from the original dataset
    selected_articles = articles_cleaned.loc[suggested_ids]

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
    # Sidebar for input
    st.sidebar.image('image.jpg', width=400)  # Adjust the width as needed
    st.sidebar.markdown("### How do you feel? What are you worried about? Which topic you want to explore? Find your answer here:")
    user_input = st.sidebar.text_input("", placeholder="Enter your feelings or topic here...")

    # Main area for title and article display
    st.title("Guide on Mental Health")

    if user_input:
        with st.spinner('Finding similar articles...'):
            html_content = find_similar_articles_html(user_input, articles_cleaned)
            st.markdown(html_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
