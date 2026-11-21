
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"


class SephoraRecommender:

    def __init__(self):

        # Load product content
        self.products = pd.read_csv(
            PROCESSED_DIR / "product_info_cleaned.csv",
            dtype={"product_id": str}
        ).reset_index(drop=True)

        # Load training interactions only
        self.train = pd.read_csv(
            PROCESSED_DIR / "train_interactions.csv",
            dtype={
                "author_id": str,
                "product_id": str
            }
        )

        # Build TF-IDF model
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            min_df=2,
            ngram_range=(1, 2),
            dtype=np.float32
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.products["combined_features"].fillna("")
        )

        # Map product IDs to matrix positions
        self.product_indices = {
            pid: index
            for index, pid in enumerate(
                self.products["product_id"]
            )
        }

        # Save training histories
        self.user_histories = (
            self.train.groupby("author_id")["product_id"]
            .apply(list)
            .to_dict()
        )

    def get_user_history(self, user_id):
        """
        Find all products previously reviewed by a user.

        These products will not be recommended again.
        """

        reviewed = set()

        review_files = sorted(
            RAW_DIR.glob("reviews_*.csv")
        )

        for file in review_files:

            chunks = pd.read_csv(
                file,
                usecols=["author_id", "product_id"],
                dtype=str,
                chunksize=100000
            )

            for chunk in chunks:

                matching = chunk.loc[
                    chunk["author_id"] == str(user_id),
                    "product_id"
                ]

                reviewed.update(matching.dropna())

        return reviewed

    def recommend(self, user_id, top_n=5):
        """
        Recommend the top N products for a user.
        """

        user_id = str(user_id).strip()

        if user_id not in self.user_histories:
            return pd.DataFrame()

        # Products positively rated in training
        liked_ids = self.user_histories[user_id]

        liked_indices = [
            self.product_indices[pid]
            for pid in liked_ids
            if pid in self.product_indices
        ]

        if not liked_indices:
            return pd.DataFrame()

        # Build user preference profile
        user_profile = np.asarray(
            self.tfidf_matrix[liked_indices].mean(axis=0),
            dtype=np.float32
        )

        # Compute cosine similarity
        similarity_scores = cosine_similarity(
            user_profile,
            self.tfidf_matrix
        ).ravel()

        # Exclude all previously reviewed products
        reviewed_ids = self.get_user_history(user_id)

        excluded_indices = [
            self.product_indices[pid]
            for pid in reviewed_ids
            if pid in self.product_indices
        ]

        similarity_scores[excluded_indices] = -np.inf
        similarity_scores[liked_indices] = -np.inf

        # Rank products
        ranked_indices = np.argsort(
            similarity_scores
        )[::-1]

        top_indices = [
            idx for idx in ranked_indices
            if np.isfinite(similarity_scores[idx])
        ][:top_n]

        # Prepare recommendations
        recommendations = self.products.iloc[
            top_indices
        ][[
            "product_id",
            "product_name",
            "brand_name",
            "primary_category",
            "secondary_category",
            "rating",
            "price_usd"
        ]].copy()

        recommendations["similarity_score"] = (
            similarity_scores[top_indices]
        )

        return recommendations.reset_index(drop=True)
