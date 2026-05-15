import csv
import re
from pathlib import Path

# Go up 3 levels from this file to reach the project root folder
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_IMAGES_DIR = _BASE_DIR / "static" / "images"

# Image formats we support (checked in this order)
_SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.svg','.avif')

# These act like a simple in-memory cache so we don't re-read the CSV on every request
_PRODUCTS = None       # list of all products
_PRODUCTS_MAP = None   # same products but stored as a dict {product_id: product} for fast lookup


def _get_image(pid):
    # Check if there's a real image file for this product, otherwise use placeholder
    for ext in _SUPPORTED_EXTS:
        if (_IMAGES_DIR / f'{pid}{ext}').exists():
            return f'images/{pid}{ext}'
    return 'images/placeholder.svg'


def _load_products():
    # Only read the CSV once — return cached data on subsequent calls
    global _PRODUCTS, _PRODUCTS_MAP
    if _PRODUCTS is not None:
        return _PRODUCTS, _PRODUCTS_MAP

    csv_path = _BASE_DIR / "data" / "processed.csv"

    # Use a set to track which product IDs we've already added
    # (the CSV has many rows per product since each row is one review)
    seen = set()
    products = []
    products_map = {}

    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['product_id']

            # Skip duplicate product IDs — we only need one entry per product
            if pid not in seen:
                seen.add(pid)

                # Safely convert numeric fields — use 0 as fallback if the value is missing
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

                # Use product tags as the description; fall back to a generic label
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

    # Save to cache
    _PRODUCTS = products
    _PRODUCTS_MAP = products_map
    return _PRODUCTS, _PRODUCTS_MAP


def search_products(query=None):
    products, products_map = _load_products()

    # No query — just return everything
    if not query or not query.strip():
        return products

    # First try: TF-IDF cosine similarity search (smarter, handles typos and related words)
    try:
        from app.logic.search import search_products as tfidf_search

        # Pass top_n = total products so we rank ALL products, not just top 5
        matches = tfidf_search(query, top_n=len(products))

        if hasattr(matches, 'empty') and not matches.empty:
            # Map the matched product IDs back to our full product dicts
            results = []
            for pid in matches['product_id'].astype(str):
                if pid in products_map:
                    results.append(products_map[pid])
            if results:
                return results
    except Exception:
        pass  # TF-IDF not available — fall through to regex

    # Fallback: simple regex search on brand name and product title
    try:
        pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
    except re.error:
        return products

    return [
        p for p in products
        if pattern.search(p['brand']) or pattern.search(p['name'])
    ]


def clear_products_cache():
    # Reset the cache so the next request re-reads the CSV (used after adding a review)
    global _PRODUCTS, _PRODUCTS_MAP
    _PRODUCTS = None
    _PRODUCTS_MAP = None


def get_product_by_id(product_id):
    # Look up a single product by its ID
    _, products_map = _load_products()
    return products_map.get(str(product_id))


def get_paginated_products(query=None, page=1, per_page=12):
    results = search_products(query)
    total = len(results)

    # Calculate how many pages we need
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Clamp page number so it stays within valid range
    page = max(1, min(page, total_pages))

    # Slice the results for the requested page
    start = (page - 1) * per_page
    return {
        'products': results[start:start + per_page],
        'total': total,
        'page': page,
        'total_pages': total_pages,
        'per_page': per_page,
    }