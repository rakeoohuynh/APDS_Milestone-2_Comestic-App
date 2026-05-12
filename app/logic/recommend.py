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
    product_df["brand_name"] + " " +
    product_df["product_title"] + " " +
    product_df["product_tags"] + " " +
    product_df["review_text"]
)

# Initialize TF-IDF Vectorizer using the predefined vocabulary 
tfidf = TfidfVectorizer(vocabulary=vocab_dict)
X_products = tfidf.fit_transform(product_df["combined_text"])


# Recommendation engine
def recommend_products(product_id, top_n = 5):

    # Check if the provided ID exists in our processed database
    if product_id not in product_df["product_id"].values:
        return "Product not found"

    idx = product_df[product_df["product_id"] == product_id].index[0]
    product_vec = X_products[idx]

    # Calculate the Cosine Similarity between the target product and all other products
    similarity = cosine_similarity(product_vec, X_products)

    # Sort similarity scores in descending order.
    # We skip the first one since it will be the product itself (similarity = 1)
    top_indices = similarity[0].argsort()[::-1][1:top_n+1]
    recommendations = product_df.iloc[top_indices][["product_id", "brand_name", "product_title"]]
    return recommendations

