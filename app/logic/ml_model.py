import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from app.preprocessed import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "processed.csv"

TFIDF_PATH = DATA_DIR / "tfidf.pkl"
LR_PATH = DATA_DIR / "lr_model.pkl"
RF_PATH = DATA_DIR / "rf_model.pkl"

tfidf = None
lr_model = None
rf_model = None

def build_models():
    global tfidf, lr_model, rf_model

    if TFIDF_PATH.exists() and LR_PATH.exists() and RF_PATH.exists():
        tfidf = joblib.load(TFIDF_PATH)
        lr_model = joblib.load(LR_PATH)
        rf_model = joblib.load(RF_PATH)
        return

    df = pd.read_csv(DATA_PATH)
    df = df[df["review_text"].fillna("").str.strip() != ""].reset_index(drop=True)

    texts = df["review_text"].fillna("")
    y = df["is_a_buyer"].astype(int)
    texts_clean = texts.apply(preprocess_text)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(texts_clean)

    # Train Models
    lr_model = LogisticRegression(max_iter=1000).fit(X, y)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)

    # 3. LƯU models ra file .pkl để lần sau không phải train lại
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(lr_model, LR_PATH)
    joblib.dump(rf_model, RF_PATH)


def predict(review_text, rating=3):
    if tfidf is None:
        build_models()

    text = preprocess_text(review_text)
    vec = tfidf.transform([text])

    pred_lr = lr_model.predict(vec)[0]
    pred_rf = rf_model.predict(vec)[0]

    final_score = (pred_lr + pred_rf) / 2
    if final_score >= 0.5:
        return 1 
    else:
        return 0

