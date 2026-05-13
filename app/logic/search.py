import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessed import preprocess_text, vocab_dict

# Define base directory and path to the processed dataset
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed.csv"

# Load the processed review dataset
df = pd.read_csv(DATA_PATH)

# Handle missing values
for col in ["brand_name", "product_title", "product_tags", "review_text"]:
    df[col] = df[col].fillna("").apply(preprocess_text)

# Aggregate data by product_id
product_df = df.groupby("product_id").agg({
    "brand_name": "first",
    "product_title": "first",
    "product_tags": "first",
    "review_text": lambda x: " ".join(x)
}).reset_index()

# Combine features into a single string for TF-IDF vectorization
product_df["combined_text"] = (
    (product_df["brand_name"] + " ") * 10 +   
    (product_df["product_title"] + " ") * 5 + 
    product_df["product_tags"] + " " +
    product_df["review_text"]                
)

# Initialize TF-IDF Vectorizer using the predefined vocabulary 
tfidf = TfidfVectorizer(vocabulary=vocab_dict)
X_products = tfidf.fit_transform(product_df["combined_text"])


def search_products(user_search, top_n = 20):
    # Clean the search query
    user_search = preprocess_text(user_search)

    if user_search.strip() == "":
        return []

    # Calculate similarity scores between the query and all products
    vec = tfidf.transform([user_search])
    similarity = cosine_similarity(vec, X_products)

    relevant_indices = [i for i, score in enumerate(similarity) if score > 0.1]

    sorted_indices = sorted(relevant_indices, key=lambda i: similarity[i], reverse=True)[:top_n]

    if not sorted_indices:
        return pd.DataFrame()
    
    result = product_df.iloc[sorted_indices][["product_id", "brand_name", "product_title"]]
    return result

