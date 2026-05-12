import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from app.preprocessed import preprocess_text

# Define directory paths for data and model persistence
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "processed.csv"

# Paths for saved serialized models (pickles)
TFIDF_PATH = DATA_DIR / "tfidf.pkl"
LR_PATH = DATA_DIR / "lr_model.pkl"
RF_PATH = DATA_DIR / "rf_model.pkl"

tfidf = None
lr_model = None
rf_model = None

def build_models():
    global tfidf, lr_model, rf_model

    # Check if pre-trained models already exist to save time
    if TFIDF_PATH.exists() and LR_PATH.exists() and RF_PATH.exists():
        tfidf = joblib.load(TFIDF_PATH)
        lr_model = joblib.load(LR_PATH)
        rf_model = joblib.load(RF_PATH)
        return

    # Load and clean dataset
    df = pd.read_csv(DATA_PATH)
    df = df[df["review_text"].fillna("").str.strip() != ""].reset_index(drop=True)

    # Use processed review text as input and buyer status as target label
    texts = df["review_text"].fillna("")
    y = df["is_a_buyer"].astype(int)

    # Convert text to numerical vectors using TF-IDF
    tfidf = TfidfVectorizer(max_features = 5000)
    X = tfidf.fit_transform(texts)

    # Train Models
    lr_model = LogisticRegression(max_iter = 1000).fit(X, y)
    rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1).fit(X, y)

    # Save trained models for future use
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(lr_model, LR_PATH)
    joblib.dump(rf_model, RF_PATH)


def predict(review_text, rating=3):
    if tfidf is None:
        build_models()

    # Preprocess and vectorize the input text
    text = preprocess_text(review_text)
    vec = tfidf.transform([text])

    # Get predictions from both classifiers
    pred_lr = lr_model.predict(vec)[0]
    pred_rf = rf_model.predict(vec)[0]

    # Calculate the average score and determine buyer status
    final_score = (pred_lr + pred_rf) / 2
    if final_score >= 0.5:
        return 1 
    else:
        return 0

