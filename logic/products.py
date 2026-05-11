import csv
import os

_PRODUCTS = None

def _load_products():
    global _PRODUCTS
    if _PRODUCTS is not None:
        return _PRODUCTS

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'processed.csv')
    seen = set()
    products = []

    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['product_id']
            if pid not in seen:
                seen.add(pid)
                try:
                    price = int(float(row['price']))
                except (ValueError, TypeError):
                    price = 0
                try:
                    rating = float(row['avg_product_rating'])
                except (ValueError, TypeError):
                    rating = 0.0
                try:
                    rating_count = int(float(row['product_rating_count']))
                except (ValueError, TypeError):
                    rating_count = 0
                products.append({
                    'id': pid,
                    'name': row['product_title'],
                    'brand': row['brand_name'],
                    'price': price,
                    'rating': rating,
                    'rating_count': rating_count,
                    'url': row['product_url'],
                    'tags': row.get('product_tags', ''),
                })

    _PRODUCTS = products
    return _PRODUCTS


def get_filtered_products(query=None):
    products = _load_products()

    if not query:
        return products

    query = query.lower()
    return [
        p for p in products
        if query in p['name'].lower() or query in p['brand'].lower()
    ]
