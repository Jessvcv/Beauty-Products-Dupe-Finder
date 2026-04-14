import numpy as np
import pandas as pd
import kagglehub

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer


class BeautyDupeRecommender:

    def __init__(self):
        self.df = None
        self.tfidf_matrix = None
        self.embeddings = None
        self.jaccard_matrix = None
        self.weights = (0.5, 0.3, 0.2)

        # Load model once (IMPORTANT for performance)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------------
    # LOAD DATA
    # -------------------------
    def load_data(self):
        path = kagglehub.dataset_download(
            "nadyinky/sephora-products-and-skincare-reviews"
        )

        df = pd.read_csv(f"{path}/product_info.csv")

        cols = [
            "product_name",
            "brand_name",
            "primary_category",
            "ingredients",
            "highlights",
            "price_usd"
        ]

        df = df[cols].dropna(subset=["ingredients"]).reset_index(drop=True)

        # Fill missing values
        df["highlights"] = df["highlights"].fillna("")

        # Combine text features
        df["combined_text"] = df["ingredients"] + " " + df["highlights"]

        # Clean ingredient list
        df["ingredient_list"] = df["ingredients"].apply(
            lambda x: [i.strip() for i in str(x).lower().split(",")]
        )

        self.df = df

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    def build_features(self):

        # TF-IDF (ingredient similarity)
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.df["ingredients"])

        # Sentence embeddings (semantic similarity)
        self.embeddings = self.model.encode(
            self.df["combined_text"].tolist(),
            normalize_embeddings=True
        )

        # Jaccard similarity (ingredient overlap)
        mlb = MultiLabelBinarizer()
        X = mlb.fit_transform(self.df["ingredient_list"])

        # Correct Jaccard similarity
        self.jaccard_matrix = 1 - pairwise_distances(X, metric="jaccard")

    # -------------------------
    # SET WEIGHTS
    # -------------------------
    def set_weights(self, w1, w2, w3):
        self.weights = (w1, w2, w3)

    # -------------------------
    # HYBRID SCORE
    # -------------------------
    def hybrid_score(self, i, j):
        w1, w2, w3 = self.weights

        # TF-IDF similarity
        tfidf_score = cosine_similarity(
            self.tfidf_matrix[i],
            self.tfidf_matrix[j]
        )[0][0]

        # Embedding similarity
        emb_score = cosine_similarity(
            self.embeddings[i].reshape(1, -1),
            self.embeddings[j].reshape(1, -1)
        )[0][0]

        # Jaccard similarity
        jac_score = self.jaccard_matrix[i, j]

        return w1 * tfidf_score + w2 * emb_score + w3 * jac_score

    # -------------------------
    # GET DUPES
    # -------------------------
    def get_top_dupes(self, idx, top_n=5):

        scores = []

        for i in range(len(self.df)):
            if i == idx:
                continue

            score = self.hybrid_score(idx, i)
            scores.append((i, score))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Format output
        results = []
        for i, s in scores[:top_n]:
            results.append({
                "product_name": self.df.iloc[i]["product_name"],
                "brand": self.df.iloc[i]["brand_name"],
                "price": self.df.iloc[i]["price_usd"],
                "similarity": float(s)
            })

        return pd.DataFrame(results)