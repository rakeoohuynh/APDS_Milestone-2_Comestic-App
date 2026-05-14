from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from app.logic.products import get_paginated_products, get_product_by_id, clear_products_cache
from app.logic.reviews import get_product_reviews, append_review
from app.logic.recommend import recommend_products
from app.logic.ml_model import predict, build_models, save_new_review

app = Flask(__name__)
app.secret_key = 'beauty-shop-secret'

print("--- Load AI Models ... ---")
build_models()

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

    recommendations = []
    try:
        recs = recommend_products(int(product_id), top_n=5)
        if not isinstance(recs, str):  # returns "Product not found" string on failure
            for pid in recs['product_id'].astype(str):
                p = get_product_by_id(pid)
                if p:
                    recommendations.append(p)
    except Exception:
        pass

    # Label each review: 1 = good (predicted buyer), 0 = bad (predicted non-buyer)
    # Falls back to rating >= 4 as "good" if ml_model is unavailable
    try:
        from app.logic.ml_model import predict as ml_predict
        ml_available = True
    except Exception:
        ml_available = False

    for review in reviews:
        if ml_available:
            try:
                review['ml_label'] = ml_predict(review['text'], int(review['rating']))
            except Exception:
                review['ml_label'] = 1 if review['rating'] >= 4 else 0
        else:
            review['ml_label'] = 1 if review['rating'] >= 4 else 0

    return render_template('product.html', product=product, reviews=reviews, recommendations=recommendations)

# TODO: GET /product/<product_id>/checkout
#   - fetch product by id, redirect to index if not found
#   - render checkout.html with product

# TODO: POST /product/<product_id>/checkout
#   - validate form fields: full_name, email, address, quantity
#   - if valid: add product_id to session['purchased'] list, set session.modified = True
#   - flash 'Order placed! You can now leave a verified review.' success
#   - redirect to product_detail

@app.route('/api/predict_label', methods=['POST'])
def api_predict():
    data = request.json
    text = data.get('text', '')
    rating = int(data.get('rating', 3))
    
    # Gọi hàm predict từ ml_model.py
    label_id = predict(text, rating)
    label_text = "Buy" if label_id == 1 else "Not Buy"
    
    return jsonify({"label": label_text})

@app.route('/product/<product_id>/review', methods=['POST'])
def add_review(product_id):
    author = request.form.get('author', '').strip()
    rating = int(request.form.get('rating', '').strip())
    title  = request.form.get('review_title', '').strip()
    text   = request.form.get('review_text', '').strip()

    final_label = int(request.form.get('final_label', 1))

    if not all([author, rating, title, text]):
        flash('Please fill in all fields.', 'error')
        return redirect(url_for('product_detail', product_id=product_id))

    ok = append_review(product_id, author, rating, title, text)
    if ok:
        save_new_review(product_id, title, text, rating, final_label)

        clear_products_cache()
        flash('Your review has been submitted. Thank you!', 'success')
    else:
        flash('Could not save your review. Product not found.', 'error')

    return redirect(url_for('product_detail', product_id=product_id))

    print("FORM DATA:", request.form)
    print("RATING RECEIVED:", request.form.get('rating'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)