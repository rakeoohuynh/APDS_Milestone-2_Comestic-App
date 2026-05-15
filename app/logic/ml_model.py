from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from app.logic.products import get_product_by_id
from app.preprocessed import preprocess_text

# Define directory paths for data and model persistence
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "processed.csv"

# Paths for serialized model storage (Pickle files)
COUNT_VEC_PATH = DATA_DIR / "count_vec.pkl"
TFIDF_TRANSFORMER_PATH = DATA_DIR / "tfidf_transformer.pkl" 
LR_PATH = DATA_DIR / "lr_model.pkl"
RF_PATH = DATA_DIR / "rf_model.pkl"
RATING_MODEL_PATH = DATA_DIR / "rating_model.pkl"

# Global variables to hold models in memory
count_vec = None
tfidf_transformer = None
lr_model = None
rf_model = None
rating_model = None

def parse_vector_list(vec_list, indices):
    """
    Parses the saved weighted vector text file into a NumPy array.
    Used for training the Random Forest model on pre-calculated Task 2.3 vectors.
    """
    rows = []
    for i in indices:
        values = vec_list[i].split(",")[1:]  
        rows.append([float(v) for v in values])
    return np.array(rows)

def build_models():
    """
    Trains or loads the ensemble models.
    Combines three different approaches: Bag-of-Words, Word Embeddings, and Numerical Ratings.
    """
    global count_vec, tfidf_transformer, lr_model, rf_model, rating_model

    # Check if pre-trained models already exist to skip training and save time
    if COUNT_VEC_PATH.exists() and TFIDF_TRANSFORMER_PATH.exists() and LR_PATH.exists() and RF_PATH.exists() and RATING_MODEL_PATH.exists():
        count_vec = joblib.load(COUNT_VEC_PATH)
        tfidf_transformer = joblib.load(TFIDF_TRANSFORMER_PATH)
        lr_model = joblib.load(LR_PATH)
        rf_model = joblib.load(RF_PATH)
        rating_model = joblib.load(RATING_MODEL_PATH)
        return

    # Load and clean dataset
    df = pd.read_csv(DATA_PATH)

    feedback_path = DATA_DIR / "test_feedback.csv"
    if feedback_path.exists():
        feedback_df = pd.read_csv(feedback_path)
        df = pd.concat([df, feedback_df], ignore_index=True)

    # Remove empty reviews to ensure training quality
    df = df[df["review_text"].fillna("").str.strip() != ""].reset_index(drop=True)

    texts = df["review_text"].fillna("")
    y = df["is_a_buyer"].astype(int) 
    non_empty_idx = df.index.tolist()
    df['review_rating'] = df['review_rating'].fillna(3)

    # --- MODEL 1: Bag-of-Words (Count Vectors) + Logistic Regression ---
    # Focuses on specific keyword frequencies
    count_vec = TfidfVectorizer(use_idf=False, norm=None, ngram_range=(1, 2))
    X = count_vec.fit_transform(texts)
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X, y)

    # --- MODEL 2: Weighted Word Embeddings + Random Forest ---
    # Focuses on semantic meaning using TF-IDF weighted GloVe vectors
    tfidf_transformer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_transformer.fit(texts)

    with open(DATA_DIR / "weighted_vectors.txt", "r") as f:
        weighted_vectors_raw = [line.strip() for line in f if line.strip()]
    
    df_base = pd.read_csv(DATA_PATH)
    base_indices = df_base[df_base["review_text"].fillna("").str.strip() != ""].index.tolist()
    X_base = parse_vector_list(weighted_vectors_raw, base_indices)

    if feedback_path.exists():
        feedback_df = pd.read_csv(feedback_path)
        fb_texts = feedback_df["review_text"].fillna("").str.strip()
        fb_texts = fb_texts[fb_texts != ""]

        X_fb = np.array([get_single_weighted_vector(t) for t in fb_texts])

        X_final = np.vstack([X_base, X_fb])
    else:
        X_final = X_base

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X_final, y)

    # --- MODEL 3: Rating-based Analysis (Gaussian Naive Bayes) ---
    # Focuses on numerical patterns 

    X_rating = df[["review_rating"]].values
    rating_model = GaussianNB().fit(X_rating, y)

    # Persist trained models to disk
    joblib.dump(count_vec, COUNT_VEC_PATH)
    joblib.dump(tfidf_transformer, TFIDF_TRANSFORMER_PATH)
    joblib.dump(lr_model, LR_PATH)
    joblib.dump(rf_model, RF_PATH)
    joblib.dump(rating_model, RATING_MODEL_PATH)

def get_single_weighted_vector(text):
    """
    Generates a feature vector for a new review by multiplying 
    Word Vectors (GloVe) with TF-IDF weights.
    Implementation follows Milestone 1.
    """
    global glove_model, tfidf_transformer

    if 'glove_model' not in globals() or glove_model is None:
        import gensim.downloader as api
        glove_model = api.load("glove-wiki-gigaword-50")

    review_vec = np.zeros(50)
    
    # Calculate TF-IDF for the new incoming review text
    tfidf_vec = tfidf_transformer.transform([text])
    feature_names = tfidf_transformer.get_feature_names_out()

    for i in tfidf_vec.indices:
        word = feature_names[i]
        if word in glove_model:
            word_vec = glove_model[word]
            weight = tfidf_vec[0, i] 
            review_vec += weight * word_vec
    return review_vec

def predict(review_text, rating=3):
    """
    Final Fusion Model: Combines probabilities from all three models.
    Returns 1 (Buyer) if the average probability >= 0.5, else 0 (Non-buyer).
    """
    global count_vec, lr_model, rf_model, rating_model
    if count_vec is None:
        build_models()

    # Preprocess text 
    text = preprocess_text(review_text)
    
    # 1. Get Probability from Model 1 
    vec_count = count_vec.transform([text])
    pred_lr = lr_model.predict_proba(vec_count)[0][1]

    # 2. Get Probability from Model 2 
    vec_weighted = get_single_weighted_vector(text).reshape(1, -1)
    pred_rf = rf_model.predict_proba(vec_weighted)[0][1]

    # 3. Get Probability from Model 3 
    pred_rating = rating_model.predict_proba([[rating]])[0][1]

    # # Print debug to check which model predicted wrong
    # print(f"LR (Keyword): {pred_lr:.4f}")
    # print(f"RF (Semantic): {pred_rf:.4f}")
    # print(f"GNB (Rating): {pred_rating:.4f}")
    
    # Calculate Final Score 
    final_score = (pred_lr * 0.45) + (pred_rf * 0.45) + (pred_rating * 0.1)
    # print(f"FINAL SCORE: {final_score:.4f}")

    return 1 if final_score >= 0.7 else 0
    
def save_new_review(product_id, review_title, review_text, author, rating, final_label, review_date=None):
    """
    Saves the user-validated (or overridden) review into a feedback CSV file.
    This data can be used for future model retraining (Human-in-the-loop).
    """
    new_review = pd.DataFrame([{
        "product_id":    str(product_id),
        "review_title":  review_title,
        "review_text":   review_text,
        "author":        author,
        "review_rating": rating,
        "is_a_buyer":    final_label
    }])

    file_path = DATA_DIR / "test_feedback.csv"
    file_exists = file_path.exists()
    new_review.to_csv(file_path, mode="a", header=not file_exists, index=False)

if __name__ == "__main__":
    # Internal test script
    sample_review = "This product is amazing! I loved the results and will buy again."
    sample_rating = 5
    prediction = predict(sample_review, rating=sample_rating)
    print(f"Sample prediction for review: {'Buyer' if prediction == 1 else 'Non-buyer'}")