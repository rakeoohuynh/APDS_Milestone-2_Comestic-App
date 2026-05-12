import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessed import preprocess_text, vocab_dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed.csv"

df = pd.read_csv(DATA_PATH)

for col in ["brand_name", "product_title", "product_tags", "review_text"]:
    df[col] = df[col].fillna("")

product_df = df.groupby("product_id").agg({
    "brand_name": "first",
    "product_title": "first",
    "product_tags": "first",
    "review_text": lambda x: " ".join(x)
}).reset_index()

product_df["combined_text"] = (
    product_df["brand_name"] + " " +
    product_df["product_title"] + " " +
    product_df["product_tags"] + " " +
    product_df["review_text"]
)

tfidf = TfidfVectorizer(vocabulary=vocab_dict)
X_products = tfidf.fit_transform(product_df["combined_text"])


def search_products(user_search, top_n=5):
    user_search = preprocess_text(user_search)

    if user_search.strip() == "":
        return []

    vec = tfidf.transform([user_search])
    sim = cosine_similarity(vec, X_products)

    top_idx = sim[0].argsort()[::-1][:top_n]
    result = product_df.iloc[top_idx][["product_id", "brand_name", "product_title"]]
    return result

# Test 
if __name__ == "__main__":
    print(search_products("waterproof mascara"))