import csv
import datetime
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_RAW_CSV  = _BASE_DIR / "data" / "cosmetics_beauty_products_reviews.csv"
_PROC_CSV = _BASE_DIR / "data" / "processed.csv"

# Column order must match the CSV headers exactly
_COLUMNS = [
    "product_id", "brand_name", "review_id", "review_title", "review_text",
    "author", "review_date", "review_rating", "is_a_buyer",
    "product_title", "price", "avg_product_rating", "product_rating_count",
    "product_tags", "product_url",
]


def get_product_reviews(product_id):
    reviews = []
    pid = str(product_id)

    with open(_RAW_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['product_id'] == pid:
                try:
                    rating = float(row['review_rating'])
                except (ValueError, TypeError):
                    rating = 0.0
                reviews.append({
                    'review_id': row['review_id'],
                    'title':     row['review_title'],
                    'text':      row['review_text'],
                    'author':    row['author'],
                    'date':      row['review_date'].split(' ')[0],
                    'rating':    rating,
                    'is_buyer':  row['is_a_buyer'].strip().lower() == 'true',
                })

    return reviews


def _generate_review_id():
    return str(int(datetime.datetime.now().timestamp() * 1000))


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
def append_review(product_id, author, rating, title, text):
    from app.logic.products import get_product_by_id
    from app.preprocessed import preprocess_text

    product = get_product_by_id(str(product_id))
    if not product:
        return False

    review_id   = _generate_review_id()
    review_date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')

    # Row written to the raw CSV (original text, int rating, TRUE/FALSE)
    raw_row = {
        'product_id':          str(product_id),
        'brand_name':          product['brand'],
        'review_id':           review_id,
        'review_title':        title,
        'review_text':         text,
        'author':              author,
        'review_date':         review_date,
        'review_rating':       int(rating),
        'is_a_buyer':          'TRUE',
        'product_title':       product['name'],
        'price':               product['price'],
        'avg_product_rating':  product['rating'],
        'product_rating_count': product['rating_count'],
        'product_tags':        product['tags'],
        'product_url':         product['url'],
    }

    # Row written to processed CSV (preprocessed text, float rating, True/False)
    processed_row = dict(raw_row)
    processed_row['review_text']   = preprocess_text(text)
    processed_row['review_rating'] = float(rating)
    processed_row['is_a_buyer']    = 'True'

    _insert_row_grouped(_RAW_CSV,  raw_row)
    _insert_row_grouped(_PROC_CSV, processed_row)

    return True