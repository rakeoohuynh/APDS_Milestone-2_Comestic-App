import csv
import re
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_IMAGES_DIR = _BASE_DIR / "static" / "images"
_SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.svg')

_PRODUCTS = None
_PRODUCTS_MAP = None  # {product_id: product_dict} for O(1) lookup after TF-IDF search


def _get_image(pid):
    for ext in _SUPPORTED_EXTS:
        if (_IMAGES_DIR / f'{pid}{ext}').exists():
            return f'images/{pid}{ext}'
    return 'images/placeholder.svg'


def _load_products():
    global _PRODUCTS, _PRODUCTS_MAP
    if _PRODUCTS is not None:
        return _PRODUCTS, _PRODUCTS_MAP

    csv_path = _BASE_DIR / "data" / "processed.csv"
    seen = set()
    products = []
    products_map = {}

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

                brand = row['brand_name'].strip()
                tags = row.get('product_tags', '').strip()
                description = tags if tags else f'Premium {brand} beauty product'

                product = {
                    'id': pid,
                    'name': row['product_title'].strip(),
                    'brand': brand,
                    'description': description,
                    'tags': tags,
                    'price': price,
                    'rating': rating,
                    'rating_count': rating_count,
                    'url': row['product_url'].strip(),
                    'image': _get_image(pid),
                }
                products.append(product)
                products_map[pid] = product

    _PRODUCTS = products
    _PRODUCTS_MAP = products_map
    return _PRODUCTS, _PRODUCTS_MAP


def search_products(query=None):
    products, products_map = _load_products()

    if not query or not query.strip():
        return products

    # --- TF-IDF cosine similarity search ---
    try:
        from app.logic.search import search_products as tfidf_search
        matches = tfidf_search(query, top_n=len(products))

        if hasattr(matches, 'empty') and not matches.empty:
            results = []
            for pid in matches['product_id'].astype(str):
                if pid in products_map:
                    results.append(products_map[pid])
            if results:
                return results
    except Exception:
        pass

    # --- Regex fallback if TF-IDF fails or finds nothing ---
    try:
        pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
    except re.error:
        return products

    return [
        p for p in products
        if pattern.search(p['brand']) or pattern.search(p['name'])
    ]


def clear_products_cache():
    global _PRODUCTS, _PRODUCTS_MAP
    _PRODUCTS = None
    _PRODUCTS_MAP = None


def get_product_by_id(product_id):
    _, products_map = _load_products()
    return products_map.get(str(product_id))


def get_paginated_products(query=None, page=1, per_page=12):
    results = search_products(query)
    total = len(results)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    return {
        'products': results[start:start + per_page],
        'total': total,
        'page': page,
        'total_pages': total_pages,
        'per_page': per_page,
    }
