OSC3801/3015 Advanced Programming for Data Science
Assignment 3: NLP Web-based Data Application - Milestone II [2]
=============================================================================

GROUP INFORMATION
-----------------------------------------------------------------------------
Group Name: UN_group1

Student 1 Name: Huynh Nguyen Minh Nhu
Student 1 ID: s4104540

Student 2 Name: Robin Pinel
Student 2 ID: s4216998

Student 3 Name: Dilys Huynh
Student 3 ID: s3938224



PROJECT OVERVIEW
-----------------------------------------------------------------------------
This project is an online shopping website for cosmetic and beauty products [2]. It allows users to browse items, search for products, leave reviews, and receive recommendations based on machine learning models developed in Milestone I [2]. 

IMPLEMENTED FEATURES
-----------------------------------------------------------------------------
* Task 1: Item Search - Users can search by brand name or description. The system handles similar keyword strings and displays match counts along with item previews [3, 4].
* Task 2: Review Creation & Automated Labeling - Users can submit reviews. The system uses a classification model to generate a "buy" or "not buy" label, which the user can override before the review is posted via a specific URL [4].
* Task 3: Item Recommendations - Viewing an item automatically displays a curated set of similar items using defined similarity measures (e.g., text features, embeddings) [5].
* Task 4: Custom Functionality - [Describe your custom feature for buyers/administrators here] [6].


HOW TO RUN THE APPLICATION
-----------------------------------------------------------------------------
IMPORTANT: Python 3.12 is required. gensim 4.4.0 is not compatible with
Python 3.13 or later.

Step 1 — Create a Python 3.12 virtual environment
    py -3.12 -m venv myenv312          (Windows)
    python3.12 -m venv myenv312        (macOS / Linux)

Step 2 — Activate the virtual environment
    myenv312\Scripts\activate          (Windows)
    source myenv312/bin/activate       (macOS / Linux)

Step 3 — Install dependencies
    pip install -r requirements.txt

    Note: gensim will download the GloVe word vectors (~66 MB) the first
    time the app starts. This is a one-time download cached locally.

Step 4 — Run the web server
    python app.py

Step 5 — Open the app in your browser
    http://127.0.0.1:5000

FEATURES WALKTHROUGH
    - Search: use the search bar on the home page to find products by
      brand name or keyword.
    - Product detail: click any product card to view details, reviews,
      and similar product recommendations.
    - Add a review: click "+ Add Review" on a product page. After filling
      in the form and clicking Submit, the AI model will classify the
      review as "Buy" or "Not Buy". You may override the label before
      confirming.
    - Buy Now: click "Buy Now" on a product page to go to the checkout
      form.


IMPORTANT DETAILS & DATA
-----------------------------------------------------------------------------
* Cosmetic/beauty review data from Milestone I is included [7].
* Additional artificial data (e.g., product images) has been created to enhance the display [7].
* The 4-minute video demonstration (.mp4) is included in the submission package [1, 6].


EXTERNAL DATA LINK
-----------------------------------------------------------------------------
OneDrive URL: https://rmiteduau-my.sharepoint.com/:f:/g/personal/s4104540_rmit_edu_vn/IgC09VNc4JonQqBsRDUz9-7SAZjjLq71DgUvDaYcLx8OnRs?e=3o5tP6

=============================================================================