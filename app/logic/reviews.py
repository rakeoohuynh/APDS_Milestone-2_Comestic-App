import csv
from datetime import datetime
from pathlib import Path

from app.logic.products import get_product_by_id
from app.preprocessed import preprocess_text
from app.logic.ml_model import save_new_review

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_RAW_CSV  = _BASE_DIR / "data" / "cosmetics_beauty_products_reviews.csv"
_PROC_CSV = _BASE_DIR / "data" / "processed.csv"
_FEEDBACK_CSV = _BASE_DIR / "data" / "test_feedback.csv"

# Column order must match the CSV headers exactly
_COLUMNS = [
    "product_id", "brand_name", "review_id", "review_title", "review_text",
    "author", "review_date", "review_rating", "is_a_buyer",
    "product_title", "price", "avg_product_rating", "product_rating_count",
    "product_tags", "product_url",
]


def get_product_reviews(product_id):
    all_reviews = []
    pid = str(product_id)

    if _PROC_CSV.exists():
        with open(_PROC_CSV, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['product_id'] == pid:
                    try:
                        rating = float(row['review_rating'])
                    except (ValueError, TypeError):
                        rating = 0.0
                    all_reviews.append({
                        'review_id': row.get('review_id', ''),
                        'title':     row.get('review_title', ''),
                        'text':      row.get('review_text', ''),
                        'author':    row.get('author', 'Anonymous'),
                        'date':      row.get('review_date', '').split(' ')[0],
                        'rating':    rating,
                        'is_buyer':  str(row.get('is_a_buyer', '0')).strip() in ('1', 'True', 'true', 'TRUE'),
                    })

    feedback_path = _BASE_DIR / "data" / "test_feedback.csv"
    if feedback_path.exists():
        with open(feedback_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['product_id'] == pid:
                    try:
                        rating = float(row['review_rating'])
                    except (ValueError, TypeError):
                        rating = 0.0
                    all_reviews.append({
                        'review_id': row.get('review_id', ''),
                        'title':     row.get('review_title', ''),
                        'text':      row.get('review_text', ''),
                        'author':    row.get('author', 'Anonymous'),
                        'date':      row.get('review_date', '').split(' ')[0],
                        'rating':    rating,
                        'is_buyer':  str(row.get('is_a_buyer', '0')).strip() in ('1', 'True', 'true', 'TRUE'),
                    })

    return all_reviews


def _generate_review_id():
    return str(int(datetime.now().timestamp() * 1000))


def _insert_row_grouped(path, row):
    """Insert row after the last existing row for the same product_id.
    Falls back to appending at the end if the product has no existing rows."""
    pid = str(row['product_id'])

    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find the index of the last row that belongs to this product
    insert_after = -1
    for i, r in enumerate(rows):
        if r['product_id'] == pid:
            insert_after = i

    new_row = {col: row.get(col, '') for col in _COLUMNS}

    if insert_after == -1:
        rows.append(new_row)          # product not found — append at end
    else:
        rows.insert(insert_after + 1, new_row)

    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# TODO: add is_buyer=False parameter to append_review signature
#   - pass is_buyer from add_review route (check product_id in session['purchased'])
#   - set raw_row['is_a_buyer']       = 'TRUE' if is_buyer else 'FALSE'
#   - set processed_row['is_a_buyer'] = 'True' if is_buyer else 'False'
def append_review(product_id, author, rating, title, text, is_buyer=False):

    from app.logic.products import get_product_by_id
    from app.logic.ml_model import save_new_review
    from datetime import datetime
    
    product = get_product_by_id(str(product_id))

    if not product:
        return False

    final_label = 1 if is_buyer else 0

    # Save review to feedback CSV for model retraining
    save_new_review(
        product_id=str(product_id),
        review_title=title,
        review_text=text,
        author=author,
        rating=int(rating),
        final_label=final_label
    )

    return True