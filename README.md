# Project Title: Building & Deploying a Content‑Based Recommendation System

## Overview
This project builds a content‑based recommendation system using cosine similarity.
It recommends products based on their descriptions, not user ratings.
The workflow covers data preprocessing → vectorization → similarity computation → recommendation logic → Streamlit UI → GitHub → Render deployment.

## Tech Stack
- Python Libraries: pandas, scikit-learn, nltk, streamlit
- ML Concept: TF‑IDF Vectorization + Cosine Similarity
- Deployment: GitHub + Render

## Project Workflow
### PART 1 — Data Preprocessing
1. Load dataset using pandas.
2. Handle missing values and check data types.
3. Clean product descriptions (remove punctuation, numbers, stopwords).
4. Ensure each product has a unique name and rating between 1–5.

### PART 2 — Text Vectorization
1. Apply TF‑IDF Vectorizer on product descriptions.
2. Convert text into numerical vectors.
3. Print the shape of the TF‑IDF matrix to confirm successful vectorization.

### PART 3 — Similarity Computation
1. Compute cosine similarity between all product vectors.
2. Display the similarity matrix.
3. Explain how cosine similarity measures closeness between product descriptions.

### PART 4 — Recommendation Logic
1. Create a function recommend(product_name) that:
    - Finds the product index.
    - Retrieves similarity scores.
    - Sorts and returns top 5 similar products.
    - Print recommended items for sample inputs.

### PART 5 — Streamlit Interface
1. Build a simple UI to:
    - Display all products.
    - Show recommendations for selected product.
    - Display product descriptions clearly.

### PART 6 — GitHub & Render Deployment
1. Push project files to GitHub (include README.md, requirements.txt, Procfile).
2. Connect GitHub repo to Render and deploy the app.
3. Verify deployment and share the live link.


🚀 How to Run
Step 1 — Clone Repository
git clone https://github.com/DipaliKhatua/Book-Recommendation-System-With-Deployment.git
cd Book-Recommendation-System-With-Deployment
Step 2 — Install Dependencies
pip install -r requirements.txt
Step 3 — Run Locally
streamlit run app.py
 Then open the local URL (usually http://localhost:8501) in your browser.
Step 4 — Deploy on Render
Push all files to GitHub (already done).
Login to Render.
Connect your GitHub repository.
Configure the build command:
pip install -r requirements.txt && streamlit run app.py
Deploy and get your public app link.
Deploy the app and verify the live link.
