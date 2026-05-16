# APDS_Milestone-2_Comestic-App
## Assignment 3: NLP Web-based Data Application - Milestone II [2]

### Group Information
**Group Name:** UN_group1
* **Student 1 Name:** Huynh Nguyen Minh Nhu | **Student ID:** s4104540 [1]
* **Student 2 Name:** Robin Pinel | **Student ID:** s4216998 [1]
* **Student 3 Name:** Dilys Huynh | **Student ID:** s3938224 [1]


### Project Overview
This project is an online shopping website for cosmetic and beauty products [2]. It allows users to browse items, search for products, leave reviews, and receive recommendations based on the machine learning models developed in Milestone I [2].

### Implemented Features
* **Task 1: Item Search:** Users can search for items by brand name or description [3]. The system gracefully handles similar keyword strings and displays the total match count alongside relevant item previews [3, 4].
* **Task 2: Review Creation & Automated Labeling:** Users can submit new reviews including a title, description, and rating [4]. The system utilizes a classification model to automatically generate a predicted "buy" or "not buy" label, which the user can override before the review is officially posted and made accessible via a specific URL [4].
* **Task 3: Item Recommendations:** When an item is selected, the website automatically displays a curated set of similar items based on defined similarity measures (e.g., text features, embeddings) [5].
* **Task 4: Custom Functionality:** [Describe your fully implemented custom feature for buyers or administrators here] [6].

### How to Run the Application

> **Python 3.12 is required.** `gensim 4.4.0` is not compatible with Python 3.13 or later.

**Step 1 — Create a Python 3.12 virtual environment**
```bash
# Windows
py -3.12 -m venv myenv312

# macOS / Linux
python3.12 -m venv myenv312
```

**Step 2 — Activate the virtual environment**
```bash
# Windows
myenv312\Scripts\activate

# macOS / Linux
source myenv312/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```
> Note: gensim will download the GloVe word vectors (~66 MB) the first time the app starts. This is a one-time download cached locally.

**Step 4 — Run the web server**
```bash
python app.py
```

**Step 5 — Open the app in your browser**
```
http://127.0.0.1:5000
```

**Features Walkthrough**
- **Search:** Use the search bar on the home page to find products by brand name or keyword.
- **Product detail:** Click any product card to view details, reviews, and similar product recommendations.
- **Add a review:** Click `+ Add Review` on a product page. After filling in the form and clicking Submit, the AI model classifies the review as **Buy** or **Not Buy**. You may override the label before confirming.
- **Buy Now:** Click `Buy Now` on a product page to go to the checkout form.
- **Admin Dashboard:** Click `Admin Login` in the top-right corner of the home page, or navigate to `http://127.0.0.1:5000/admin/login`
  - **Password:** `admin123`

### Important Details & Data
* Cosmetic/beauty review data from Milestone I is utilized [7].
* Additional artificial data (e.g., product images) has been incorporated to enhance the visual display [7].
* An up-to-4-minute video demonstration (`.mp4`) is included in the submission package [1, 6].

### External Data Link 

**OneDrive URL:** [\[Link Here\]](https://rmiteduau-my.sharepoint.com/:f:/g/personal/s4104540_rmit_edu_vn/IgC09VNc4JonQqBsRDUz9-7SAZjjLq71DgUvDaYcLx8OnRs?e=3o5tP6)
