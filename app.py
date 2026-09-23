# 1. Import Libraries

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 2. Load Dataset
products = pd.read_csv("Data/Amazon Product Review.csv")

# 3. Preprocessing (copied from notebook)
# Lowercase
products['reviews.text'] = products['reviews.text'].astype(str).str.lower()

# Remove punctuation
products['reviews.text'] = products['reviews.text'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

products['clean_text'] = products['reviews.text'].apply(remove_stopwords)


# Handle missing values
products['clean_text'] = products['clean_text'].fillna("")

# 4. Vectorization + Similarity Matrix
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(products['clean_text'])
similarity_matrix = cosine_similarity(tfidf_matrix)


# Drop duplicate product names
products = products.drop_duplicates(subset=['name']).reset_index(drop=True)

# Rebuild TF-IDF and similarity matrix after deduplication
tfidf_matrix = vectorizer.fit_transform(products['clean_text'])
similarity_matrix = cosine_similarity(tfidf_matrix)

# 5. Recommendation Function
def recommend(item_name, top_n=5):
    indices = products[products['name'] == item_name].index
    if len(indices) == 0:
        return ["Item not found!"]
    idx = indices[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_items = sim_scores[1:top_n+1]

    recommendations = [products.iloc[i[0]]['name'] for i in top_items]
    return recommendations

# 6. Streamlit UI
st.title("📚 Product Recommendation System")

# Dropdown
selected_item = st.selectbox("Select a product:", products['name'].unique())

# Button
if st.button("Get Recommendations"):
    recs = recommend(selected_item)
    st.write("### Top Recommendations:")
    for r in recs:
        st.write("- ", r)
