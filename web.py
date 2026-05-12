# Load libraries
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.metrics.pairwise import cosine_similarity

# Load vocabulary
vocab_dict = {}

with open("vocab.txt", "r", encoding = "utf-8") as filein:
    for line in filein:
        word, ind = line.strip().split(":")
        vocab_dict[word] = int(ind)

# Load stopwords
with open("stopwords_en.txt", "r", encoding="utf-8") as f:
    stopwords = set(line.strip().lower() for line in f if line.strip())

# Preprocess user input 
TOKENIZER = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def preprocess_text(text):
    """Apply Task 1 preprocessing and keep only vocabulary words."""
    if not isinstance(text, str):
        return ""
    tokens = TOKENIZER.findall(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2 and t not in stopwords and t in vocab_dict]
    return " ".join(tokens)

# Load processed data
df = pd.read_csv("processed.csv")

# Create product-level dataframe

cols = ["brand_name", "product_title", "product_tags", "review_text"]

for col in cols:
    df[col] = df[col].fillna("")

df["brand_name"] = df["brand_name"].apply(preprocess_text)
df["product_title"] = df["product_title"].apply(preprocess_text)
df["product_tags"] = df["product_tags"].apply(preprocess_text)

product_df = (df.groupby("product_id").agg({
    "brand_name": "first",
    "product_title": "first",
    "product_tags": "first",
    "review_text": lambda x: " ".join(x)
})
.reset_index()
)

product_df["combined_text"] = (
    product_df["brand_name"] + " " +
    product_df["product_title"] + " " +
    product_df["product_tags"] + " " +
    product_df["review_text"]
)

product_df.head()

# Initialize TF-IDF vectorizer with the predefined vocabulary
tfidf = TfidfVectorizer(vocabulary = vocab_dict)
X_products = tfidf.fit_transform(product_df["combined_text"])

# Search engine
def search_products(user_search, top_n = 5):
    user_search = preprocess_text(user_search)

    # empty query after preprocessing
    if user_search.strip() == "":
        return "No valid search terms found."
    
    user_search_vec = tfidf.transform([user_search])

    similarity = cosine_similarity(user_search_vec, X_products)

    top_indices = similarity[0].argsort()[::-1][:top_n]
    results = product_df.iloc[top_indices][["product_id", "brand_name", "product_title"]]
    return results


# Recommendation engine
def recommend_products(product_id, top_n = 5):

    if product_id not in product_df["product_id"].values:
        return "Product not found"

    idx = product_df[product_df["product_id"] == product_id].index[0]
    product_vec = X_products[idx]

    similarity = cosine_similarity(product_vec, X_products)

    top_indices = similarity[0].argsort()[::-1][1:top_n+1]
    recommendations = product_df.iloc[top_indices][["product_id", "brand_name", "product_title"]]
    return recommendations

print(search_products("waterproof mascara"))