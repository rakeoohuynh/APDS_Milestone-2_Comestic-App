import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOCAB_FILE = DATA_DIR / "vocab.txt"
STOPWORDS_FILE = DATA_DIR / "stopwords_en.txt"

# # Load vocabulary mapping from file
vocab_dict = {}
with VOCAB_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        word, ind = line.strip().split(":")
        vocab_dict[word] = int(ind)

# Load stopwords
with STOPWORDS_FILE.open("r", encoding="utf-8") as f:
    stopwords = set(line.strip().lower() for line in f if line.strip())

TOKENIZER = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def preprocess_text(text):
    """
    Cleans input text by removing non-essential words while preserving 
    sentiment-critical negations.
    """
    if not isinstance(text, str):
        return ""
    
    tokens = TOKENIZER.findall(text)
    tokens = [t.lower() for t in tokens]

    keep_words = {
        'not', 'no', 'never', 'neither', 'nor', 'none', 
        "don't", "can't", "wasn't", "isn't", "aren't", "couldn't", "won't"
    }
    cleaned_tokens = []
    for t in tokens:
        if t in keep_words:
            cleaned_tokens.append(t)
        elif len(t) >= 2 and t not in stopwords:
            cleaned_tokens.append(t)

    return " ".join(cleaned_tokens)


