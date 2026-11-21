# sephora-recommendation-system
Content-Based Beauty Product Recommendation System using Python and Machine Learning
# 💄 Sephora Beauty Recommendation System

A personalized, content-based beauty product recommendation system developed as a university **Data Mining Group Project**.

The system analyzes Sephora product information and historical user reviews to recommend five products based on users' previously liked items.

## 1. Project Overview

The goal of this project is to develop a recommendation system that helps users discover beauty products aligned with their preferences.

The system uses **TF-IDF Vectorization** to represent product features and **Cosine Similarity** to identify products similar to a user's preference profile.

A web interface built with Streamlit enables users to enter their User ID and receive their top five product recommendations.

### Main Features

- Personalized Top 5 product recommendations
- Content-based recommendation using TF-IDF
- Cosine Similarity for product matching
- User interaction analysis and data preprocessing
- Comparison of three recommendation models
- Performance evaluation with Precision@5, Recall@5 and MAP@5
- Interactive Streamlit application

## 2. Dataset

**Source:** [Sephora Products and Skincare Reviews — Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

The project uses two types of data:

| Dataset | Description |
|---|---|
| `product_info.csv` | Product information, including names, brands, categories, ingredients, prices and ratings |
| `reviews_*.csv` | User reviews containing user IDs, product IDs, ratings and review-related information |

### Dataset Statistics

| Property | Value |
|---|---:|
| Total products | 8,494 |
| Product attributes | 27 |
| Total review records | 1,094,411 |
| Unique users | 503,216 |
| Reviewed products | 2,351 |
| Positive reviews (4–5 stars) | 898,340 |
| Users with at least 5 positively rated products | 30,078 |

The review data primarily covers skincare products. Although the product catalog includes other beauty categories, personalized recommendations depend on the available user interaction history.

### Downloading the Dataset

Large CSV files are excluded from GitHub using `.gitignore`.

Download the dataset from Kaggle and place the files in:

`data/raw/`

The expected files are:

```text
data/raw/
├── product_info.csv
├── reviews_0-250.csv
├── reviews_250-500.csv
├── reviews_500-750.csv
├── reviews_750-1250.csv
└── reviews_1250-end.csv
```

Extract the ZIP archives before running the notebooks.

## 3. Technologies

- Python 3.13
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Matplotlib and Seaborn
- Jupyter Notebook
- Streamlit
- Git and GitHub

## 4. Project Structure

```text
sephora-recommendation-system/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_user_interaction_analysis.ipynb
│   ├── 03_content_based_recommendation.ipynb
│   └── 04_model_optimization.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── recommendation.py
│   └── evaluation.py
└── tests/
```

The notebooks contain the exploratory analyses, model development and evaluation experiments.

The `src/recommendation.py` module implements the recommendation algorithm used by the Streamlit application.

## 5. Methodology

### 5.1 Exploratory Data Analysis

The dataset was examined to understand:

- Product categories and brands
- Missing values
- Duplicate product and user-product records
- User rating distribution
- Number of reviews per user
- Product and review ID matching

### 5.2 Data Preprocessing

Product text attributes were cleaned and combined to create content features.

Duplicate user-product interactions were aggregated using their average rating.

Interactions with ratings of at least 4 were considered positive.

Users with at least five distinct positively rated products were selected for model evaluation.

### 5.3 Training and Testing

Positive user-product interactions were divided into approximately 80% training and 20% testing for each eligible user.

Training interactions were used to construct user preference profiles, while held-out test interactions were used for recommendation evaluation.

The same user-product interaction was not included in both sets.

### 5.4 TF-IDF Vectorization

Text features were transformed into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency).

The baseline configuration used:

- `max_features=10000`
- `min_df=2`
- `ngram_range=(1, 2)`
- English stop-word removal

### 5.5 Cosine Similarity

For each user, the system calculates a preference profile using the mean TF-IDF vectors of products they positively rated in the training dataset.

Cosine Similarity is then used to rank candidate products according to their similarity to the user's profile.

Previously reviewed products are excluded from the application's recommendations.

The five highest-scoring eligible products are presented to the user.

## 6. Model Optimization

Three content-based recommendation approaches were compared:

1. **Baseline TF-IDF:** Uses combined product text features.
2. **Weighted Content:** Gives additional emphasis to categories and brand information.
3. **Ingredients Focused:** Gives additional emphasis to ingredients and product highlights.

All three approaches were compared using the same sample of 100 users.

## 7. Experimental Results

The following results were obtained during the model development comparison.

| Model | Precision@5 | Recall@5 | MAP@5 |
|---|---:|---:|---:|
| Baseline TF-IDF | 0.0560 | 0.1812 | **0.1321** |
| Weighted Content | **0.0580** | **0.1829** | 0.1294 |
| Ingredients Focused | 0.0560 | 0.1812 | 0.1294 |

The Weighted Content model achieved slightly higher Precision@5 and Recall@5 scores.

The Baseline TF-IDF model achieved the highest MAP@5 score and was selected for the current Streamlit prototype.

**Note:** These are development-stage results based on a 100-user sample, not independent final test results.

## 8. Installation and Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/simgedurak/sephora-recommendation-system.git
cd sephora-recommendation-system
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4 — Prepare the Dataset

Download and extract the Kaggle CSV files into `data/raw/`.

Run the preprocessing and analysis notebooks in the following order:

1. `01_exploratory_data_analysis.ipynb`
2. `02_user_interaction_analysis.ipynb`
3. `03_content_based_recommendation.ipynb`
4. `04_model_optimization.ipynb`

Ensure that the required processed CSV files have been generated in `data/processed/` before launching the application.

### Step 5 — Launch the Application

```bash
python -m streamlit run app.py
```

Open the local address shown in the terminal, usually:

`http://localhost:8501`

## 9. Application Usage

Enter a valid User ID and click **Get My Top 5 Recommendations**.

Example demonstration User ID:

`10000117144`

The application displays up to five recommendations with:

- Product name
- Brand
- Category and subcategory
- Price in USD
- Product rating
- Content similarity score

The similarity score represents content similarity, not the probability of purchase.

## 10. Limitations and Future Work

The current system has several limitations:

- User interaction data is concentrated in skincare products.
- Users without sufficient training history cannot receive personalized recommendations.
- The evaluation uses historical positive ratings as relevance signals.
- Model comparison was conducted on a limited development sample.
- Additional independent evaluation and popularity-based comparisons would improve the reliability of conclusions.

Possible future improvements include hybrid recommendation methods, cold-start strategies, more comprehensive evaluation and performance optimization.

## 11. Team Collaboration

This project is developed collaboratively using Git and GitHub.

Contributors can work on separate branches and submit their changes through pull requests.

Large datasets and local virtual environments are excluded from version control.

## 12. References

- [Sephora Products and Skincare Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Project Type:** University Data Mining Group Project

**Recommendation Approach:** Content-Based Filtering

**Main Algorithms:** TF-IDF and Cosine Similarity