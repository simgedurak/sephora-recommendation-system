
import streamlit as st
import pandas as pd

from src.recommendation import SephoraRecommender


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sephora Beauty Recommender",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #fff8fb;
        color: #242024;
    }

    /* Main headings and text */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp p,
    .stApp label {
        color: #242024 !important;
    }

    /* Markdown content */
    [data-testid="stMarkdownContainer"] {
        color: #242024 !important;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 43px;
        font-weight: 800;
        color: #d63384 !important;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666 !important;
        margin-bottom: 30px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fff0f6;
    }

    [data-testid="stSidebar"] * {
        color: #242024;
    }

    /* Input labels */
    [data-testid="stTextInput"] label {
        color: #242024 !important;
        font-weight: 600;
    }

    /* Text input */
    [data-testid="stTextInput"] input {
        background-color: white;
        color: #242024;
        border: 1px solid #e6a4c4;
        border-radius: 8px;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #d63384;
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 12px;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #b82b70;
        color: white !important;
    }

    /* Metric labels and values */
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"] * {
        color: #242024 !important;
    }

    /* Product cards */
    .product-card {
        background-color: #ffffff;
        border: 1px solid #f2c5d9;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .product-brand {
        color: #d63384 !important;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .product-name {
        color: #242024 !important;
        font-size: 21px;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 8px;
    }

    .product-detail {
        color: #555 !important;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .rank-number {
        color: #d63384 !important;
        font-size: 16px;
        font-weight: 700;
    }

    /* Dividers */
    hr {
        border-color: #f2c5d9;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD RECOMMENDATION MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_recommender():
    return SephoraRecommender()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '💄 Sephora Beauty Recommender'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover personalized beauty products '
    'based on your preferences.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("💗 About the Project")

    st.write(
        "This application is a Content-Based "
        "Recommendation System developed as "
        "a university Data Mining project."
    )

    st.divider()

    st.subheader("Technologies")

    st.write("🐍 Python")
    st.write("📊 TF-IDF Vectorization")
    st.write("🔎 Cosine Similarity")
    st.write("💻 Streamlit")

    st.divider()

    st.subheader("How It Works")

    st.write(
        "1. Enter a valid User ID."
    )

    st.write(
        "2. The system analyzes products "
        "you previously liked."
    )

    st.write(
        "3. It recommends five similar products "
        "you have not reviewed before."
    )

    st.divider()

    st.caption(
        "Dataset: Sephora Products and "
        "Skincare Reviews — Kaggle"
    )


# --------------------------------------------------
# INITIALIZE MODEL
# --------------------------------------------------

try:
    with st.spinner("Loading recommendation model..."):
        recommender = load_recommender()

except Exception as error:
    st.error("Unable to load the recommendation model.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("✨ Find Your Personalized Products")

st.write(
    "Enter your User ID to discover "
    "your top five beauty matches."
)

user_id = st.text_input(
    "User ID",
    placeholder="Example: 10000117144"
)

recommend_button = st.button(
    "✨ Get My Top 5 Recommendations",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# RECOMMENDATION RESULTS
# --------------------------------------------------

if recommend_button:

    user_id = user_id.strip()

    if not user_id:

        st.warning(
            "Please enter your User ID."
        )

    elif user_id not in recommender.user_histories:

        st.error(
            "User not found in the training dataset. "
            "Please enter a valid User ID."
        )

    else:

        try:

            with st.spinner(
                "Finding your personalized recommendations..."
            ):

                recommendations = recommender.recommend(
                    user_id,
                    top_n=5
                )

            if recommendations.empty:

                st.warning(
                    "No recommendations are available "
                    "for this user."
                )

            else:

                st.success(
                    "Your personalized recommendations are ready!"
                )

                st.subheader("💗 Your Top 5 Beauty Matches")

                st.write(
                    "These products were selected based "
                    "on their similarity to products "
                    "you previously liked."
                )

                # Display recommendation cards
                for index, product in recommendations.iterrows():

                    product_name = product["product_name"]
                    brand = product["brand_name"]
                    category = product["primary_category"]
                    secondary = product["secondary_category"]
                    price = product["price_usd"]
                    rating = product["rating"]
                    similarity = product["similarity_score"]

                    with st.container(border=True):

                        st.markdown(
                            f"### {index + 1}. {product_name}"
                        )

                        st.write(f"**Brand:** {brand}")
                        st.write(f"**Category:** {category}")

                        if pd.notna(secondary) and str(secondary).strip():
                            st.write(
                                f"**Subcategory:** {secondary}"
                            )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            if pd.notna(price):
                                st.metric(
                                    "Price",
                                    f"${price:.2f}"
                                )
                            else:
                                st.metric(
                                    "Price",
                                    "N/A"
                                )

                        with col2:

                            if pd.notna(rating):
                                st.metric(
                                    "Product Rating",
                                    f"{rating:.2f} / 5"
                                )
                            else:
                                st.metric(
                                    "Product Rating",
                                    "N/A"
                                )

                        with col3:

                            st.metric(
                                "Content Similarity",
                                f"{similarity:.3f}"
                            )

                st.divider()

                st.info(
                    "Content similarity measures how "
                    "similar a product is to your "
                    "preference profile. It is not "
                    "a purchase probability."
                )

        except Exception as error:

            st.error(
                "An error occurred while generating recommendations."
            )

            st.exception(error)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Sephora Beauty Recommender | "
    "Data Mining Group Project | "
    "TF-IDF & Cosine Similarity"
)
