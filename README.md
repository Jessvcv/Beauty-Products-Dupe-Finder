# 💄 DupeLab: Beauty Product Duplication Recommender

## 📌 Overview
DupeLab is a **content-based recommendation system** that identifies affordable “dupes” for beauty and skincare products.

Given a product, it finds similar alternatives using a combination of:
- Ingredient similarity
- Semantic meaning of product descriptions
- Ingredient overlap

The goal is to help users discover **lower-cost alternatives without sacrificing formulation similarity**.

---

## 🎯 Problem Statement
Consumers often want cheaper alternatives (“dupes”) to high-end beauty products, but:
- Ingredient lists are hard to compare manually
- Product similarity is subjective
- Price-to-performance comparisons are not standardized

DupeLab solves this using **data-driven similarity modeling**.

---

## 📊 Dataset
- Source: Kaggle — *Sephora Products and Skincare Reviews*
- Size: ~8,000+ products
- Key fields:
  - product_name
  - brand_name
  - primary_category
  - ingredients
  - highlights
  - price_usd

---

## ⚙️ Feature Engineering

Each product is transformed into multiple representations:

### 1. TF-IDF Representation
- Applied to raw ingredient lists
- Captures lexical similarity between formulations

### 2. Sentence Embeddings
- Model: `all-MiniLM-L6-v2` (Sentence Transformers)
- Encodes:
  - ingredients
  - product highlights
- Captures semantic similarity between products

### 3. Jaccard Similarity
- Measures overlap between ingredient sets
- Uses binary multi-label encoding

---

## 🧠 Hybrid Similarity Model

Final similarity score is computed as:
- score = w1 * TF-IDF + w2 * Embeddings + w3 * Jaccard

Default weights:
- TF-IDF: 0.5
- Embeddings: 0.3
- Jaccard: 0.2

---

## 🔍 Recommendation Engine

For a given product index:

1. Compare against all other products
2. Compute hybrid similarity score
3. Rank results in descending order
4. Return top-N most similar products

### Output includes:
- Product name
- Brand
- Price
- Similarity score

---

## ⚡ Key Improvements in This Version
- Preloaded SentenceTransformer model (faster inference)
- Cleaner modular architecture
- Correct Jaccard similarity using `pairwise_distances`
- Unified hybrid scoring function
- More scalable feature pipeline

---

## 🚀 How to Run

### Install dependencies
```bash
pip install numpy pandas scikit-learn sentence-transformers kagglehub
```
Run recommender
- from recommender import BeautyDupeRecommender
- model = BeautyDupeRecommender()
- model.load_data()
- model.build_features()

Get recommendations
model.get_top_dupes(idx=10, top_n=5)

🌐 Streamlit App (DupeLab UI)

DupeLab includes a Streamlit-based interactive interface.

### Features:
- 🔍 Product search
- 💄 Product selection panel
- 💡 Top dupe recommendations
- 🎲 “Surprise Me” random product exploration

### Architecture:
- Streamlit frontend
- Cached ML model loading (@st.cache_resource)
- Real-time inference using hybrid similarity engine

### 🔬 Key Insights
- Hybrid similarity significantly improves recommendation quality
- Ingredient overlap alone is insufficient → embeddings add semantic context
- Price filtering + similarity creates meaningful “dupe discovery”
- Precomputed embeddings improve runtime efficiency

### 🔮 Future Work
- Add FAISS for faster nearest-neighbor search
- Deploy web app (Streamlit Cloud / HuggingFace Spaces)
- Add explainability (“why this is a dupe” feature)
- Fine-tune embeddings on skincare domain data
- Add user preference personalization layer

Author
- Jessica Tran
