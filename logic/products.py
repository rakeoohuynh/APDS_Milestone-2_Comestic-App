import csv
import os
import re

_PRODUCTS = None
_IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'images')
_SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.svg')

def _get_image(pid):
    """Return 'images/{pid}.ext' if a matching file exists, else the placeholder."""
    for ext in _SUPPORTED_EXTS:
        if os.path.isfile(os.path.join(_IMAGES_DIR, f'{pid}{ext}')):
            return f'images/{pid}{ext}'
    return 'images/placeholder.svg'

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

                brand = row['brand_name'].strip()
                tags = row.get('product_tags', '').strip()
                description = tags if tags else f'Premium {brand} beauty product'

                products.append({
                    'id': pid,
                    'name': row['product_title'].strip(),
                    'brand': brand,
                    'description': description,
                    'price': price,
                    'rating': rating,
                    'rating_count': rating_count,
                    'url': row['product_url'].strip(),
                    'image': _get_image(pid),
                })

    _PRODUCTS = products
    return _PRODUCTS


def search_products(query=None):
    """Return all products, or those matching query using case-insensitive regex."""
    products = _load_products()

    if not query or not query.strip():
        return products

    try:
        pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)
    except re.error:
        return products

    return [
        p for p in products
        if pattern.search(p['brand']) or pattern.search(p['name']) or pattern.search(p['description'])
    ]


def get_paginated_products(query=None, page=1, per_page=12):
    """Return a page slice plus metadata dict for the template."""
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
