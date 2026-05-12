from flask import Flask, render_template, request, redirect, url_for, flash
from app.logic.products import get_paginated_products, get_product_by_id, clear_products_cache
from app.logic.reviews import get_product_reviews, append_review

app = Flask(__name__)
app.secret_key = 'beauty-shop-secret'

@app.route('/')
def index():
    keyword = request.args.get('search_box', '').strip()
    page = request.args.get('page', 1, type=int)
    data = get_paginated_products(query=keyword, page=page)
    return render_template('index.html', **data, last_search=keyword)

@app.route('/product/<product_id>')
def product_detail(product_id):
    product = get_product_by_id(product_id)
    if not product:
        return redirect(url_for('index'))
    reviews = get_product_reviews(product_id)
    return render_template('product.html', product=product, reviews=reviews)

@app.route('/product/<product_id>/review', methods=['POST'])
def add_review(product_id):
    author = request.form.get('author', '').strip()
    rating = request.form.get('rating', '').strip()
    title  = request.form.get('review_title', '').strip()
    text   = request.form.get('review_text', '').strip()

    if not all([author, rating, title, text]):
        flash('Please fill in all fields.', 'error')
        return redirect(url_for('product_detail', product_id=product_id))

    ok = append_review(product_id, author, rating, title, text)
    if ok:
        clear_products_cache()
        flash('Your review has been submitted. Thank you!', 'success')
    else:
        flash('Could not save your review. Product not found.', 'error')

    return redirect(url_for('product_detail', product_id=product_id))

if __name__ == '__main__':
    app.run(debug=True)