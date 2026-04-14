# 💄 Finding Affordable Alternatives: A Similarity-Based System for Beauty Product Duplication

## 📌 Overview
This project builds a **content-based recommendation system** that identifies affordable “dupes” for beauty and skincare products.

Consumers often struggle to find lower-cost alternatives to premium products with similar ingredients and effects. This system automates that process by comparing products using **ingredient similarity, semantic embeddings, and pricing information**.

Given a product, the system returns a ranked list of **similar but potentially cheaper alternatives**.

---

## 🎯 Objective
- Identify functionally similar beauty products (“dupes”)
- Reduce reliance on anecdotal or influencer-based recommendations
- Explore whether **ingredient-level similarity can approximate product equivalence**
- Incorporate **price-awareness into recommendation ranking**

---

## 📊 Dataset
- Source: Kaggle — *Sephora Products and Skincare Reviews*
- Size: ~8,000+ products
- Key attributes:
  - product_name
  - brand_name
  - primary_category
  - secondary_category
  - highlights
  - ingredients
  - price_usd

---

## 🧹 Data Preprocessing
The dataset is cleaned and standardized using NLP techniques:

- Lowercasing and punctuation removal
- Ingredient normalization (e.g., aqua → water, parfum → fragrance)
- Filtering noisy ingredient tokens
- Splitting ingredients into structured lists and sets
- Creation of combined textual representation:
  - ingredients + product highlights

---

## 🧠 Methodology

This system uses a **hybrid similarity approach** combining three signals:

### 1. TF-IDF Vectorization
Captures lexical similarity between ingredient lists using:
- Word and bi-grams
- Document frequency filtering

### 2. Sentence Embeddings
Uses `SentenceTransformer (all-MiniLM-L6-v2)` to encode:
- Ingredient context
- Product highlights

This captures deeper **semantic similarity** beyond exact word overlap.

### 3. Jaccard Similarity
Measures overlap between ingredient sets to capture **chemical similarity**.

---

## ⚖️ Hybrid Similarity Model
Final similarity score is computed as a weighted combination:

- TF-IDF similarity → 50%
- Embedding similarity → 30%
- Jaccard similarity → 20%

This balances:
- lexical match
- semantic meaning
- ingredient overlap

---

## 🔎 Recommendation System

For a given product, the system:

1. Computes similarity with all other products
2. Applies filters:
   - Same product category (optional)
   - Only cheaper alternatives (optional)
3. Ranks candidates by hybrid similarity score
4. Returns top-N most similar products

---

## 💰 Price-Aware Filtering
A key feature of this system is **cost-sensitive recommendation**:
- Prioritizes products that are similar but lower in price
- Helps identify realistic “dupes” instead of purely similar items

---

## 📈 Evaluation
The system includes a simple evaluation metric:

- Measures the proportion of recommended products that are cheaper than the query product
- Helps quantify the effectiveness of price-aware filtering

---

## 📉 Visualization (Planned / Optional Extension)
Future improvements include:
- PCA / t-SNE visualization of product clusters
- Embedding space analysis of product similarity
- Category-based clustering insights

---

## 🚀 How to Run

### Install dependencies
```bash
pip install pandas numpy scikit-learn sentence-transformers kagglehub
