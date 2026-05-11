from flask import Flask, render_template, request
from logic.products import get_filtered_products

app = Flask(__name__)

@app.route('/')
def index():
    keyword = request.args.get('search_box', '')
    products_to_show = get_filtered_products(keyword)
    return render_template('index.html', products=products_to_show, last_search=keyword)

if __name__ == '__main__':
    app.run(debug=True)
