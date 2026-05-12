import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOCAB_FILE = DATA_DIR / "vocab.txt"
STOPWORDS_FILE = DATA_DIR / "stopwords_en.txt"

# load vocab
vocab_dict = {}
with VOCAB_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        word, ind = line.strip().split(":")
        vocab_dict[word] = int(ind)

# load stopwords
with STOPWORDS_FILE.open("r", encoding="utf-8") as f:
    stopwords = set(line.strip().lower() for line in f if line.strip())

TOKENIZER = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = TOKENIZER.findall(text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2 and t not in stopwords and t in vocab_dict]
    return " ".join(tokens)


