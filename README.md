# 💄 DupeLab: Beauty Product Duplication Recommender

## 📌 Overview

DupeLab is a hybrid content-based recommendation system designed to identify affordable “dupes” for beauty and skincare products. Given a product, the system recommends similar alternatives that preserve formulation similarity while minimizing price.

The motivation behind this project is that consumers often want lower-cost alternatives to high-end skincare products, but comparing ingredients and product functionality manually is difficult and subjective. DupeLab addresses this problem using a combination of natural language processing, embedding-based semantic similarity, and ingredient-level overlap modeling.

---

## 🎯 Problem Statement

Consumers face three major challenges when searching for product dupes:

- Ingredient lists are long, unstructured, and difficult to compare manually
- Product similarity is subjective and depends on both formulation and purpose
- Price-to-performance comparisons are not standardized

DupeLab solves this by building a **data-driven hybrid similarity model** that ranks products based on ingredient, semantic, and structural similarity.

---

## 📊 Dataset

- **Source:** Kaggle — Sephora Products and Skincare Reviews  
- **Size:** ~8,000+ products  
- **Key Features:**
  - product_name
  - brand_name
  - primary_category
  - secondary_category
  - ingredients
  - highlights
  - price_usd

---

## 🧹 Data Preprocessing & Feature Engineering

### 1. Text Cleaning
All textual fields are standardized:
- Lowercasing
- Removal of special characters
- Filtering invalid ingredient tokens

### 2. Ingredient Standardization
Domain-specific mappings were applied:
- “aqua” → “water”
- “eau” → “water”
- fragrance-related synonyms grouped

### 3. Ingredient Filtering
Removed:
- invalid tokens
- overly long or noisy ingredient entries

---

## 🧠 Feature Representations

Each product is represented using three complementary approaches:

### 1. TF-IDF (Ingredient-Level Similarity)
Captures lexical similarity between ingredient lists using weighted n-grams.

### 2. Sentence Embeddings (Semantic Similarity)
Model: `all-MiniLM-L6-v2`

Encodes:
- product ingredients
- product highlights

This captures functional similarity beyond exact word overlap.

### 3. Jaccard Similarity (Ingredient Overlap)
Measures set overlap between ingredient lists, reinforcing formulation similarity.

---

## ⚖️ Hybrid Similarity Model

Final similarity score:

\[
score = w_1(TF\text{-}IDF) + w_2(Embeddings) + w_3(Jaccard)
\]

### Default Weights:
- TF-IDF: 0.5  
- Embeddings: 0.3  
- Jaccard: 0.2  

These weights were selected empirically to balance:
- lexical similarity (TF-IDF)
- semantic meaning (embeddings)
- ingredient overlap (Jaccard)

---

## 🔍 Recommendation Engine

For a given product:

1. Compare against all products in dataset  
2. Compute hybrid similarity score  
3. Apply optional filters:
   - same product category
   - cheaper-only constraint  
4. Rank and return top-N recommendations  

### Output Includes:
- product name  
- brand  
- price  
- similarity score  

---

## 📊 Evaluation Strategy

Since no ground-truth "dupe labels" exist, evaluation is performed using proxy metrics:

### 1. Percent Cheaper Metric
Measures the proportion of recommended products that are cheaper than the original product.

This directly evaluates the system’s goal of identifying cost-effective alternatives.

### 2. Qualitative Case Analysis
Manual inspection of recommendation outputs comparing:
- TF-IDF only
- Embedding only
- Hybrid model

---

## ⚡ Key Insights

- TF-IDF alone overemphasizes common ingredients (e.g., water, glycerin), reducing recommendation quality.
- Sentence embeddings significantly improve functional similarity matching between products.
- Jaccard similarity strengthens ingredient-level consistency.
- The hybrid model produces the most balanced and realistic dupe recommendations.
- Price filtering improves practical usability by ensuring recommendations are economically meaningful.

---

## 🔧 Technical Improvements

- Preloaded sentence transformer model for faster inference
- Modular recommender class design
- Efficient feature pipeline structure
- Hybrid scoring function with tunable weights
- Scalable design for larger datasets

---

## 📈 Example Output

**Input Product:**
- Drunk Elephant Protini Polypeptide Cream ($68)

**Top Recommendation:**
- The Ordinary Natural Moisturizing Factors ($12)

**Reasoning:**
- High overlap in moisturizing ingredients
- Similar semantic description of skin barrier support
- Strong hybrid similarity score
- Significantly lower price (successful dupe)

---

## 🚀 How to Run

### Install dependencies:
```bash
pip install numpy pandas scikit-learn sentence-transformers kagglehub
```
## Run Recommender 
```python
from recommender import BeautyDupeRecommender

model = BeautyDupeRecommender()
model.load_data()
model.build_features()

print(model.get_top_dupes(idx=10, top_n=5))
```

### 🌐 Streamlit App

## DupeLab includes an interactive Streamlit interface.

## Features:
- Product search and selection
- Top dupe recommendations
- Random product exploration (“Surprise Me”)
## Architecture:
- Streamlit frontend
- Cached ML model loading
- Real-time hybrid similarity inference
## 🔮 Future Work
- Implement FAISS for faster nearest-neighbor search
- Fine-tune embeddings on skincare-specific corpus
- Add explainability layer (“why this is a dupe”)
- Introduce user personalization
- Deploy via Streamlit Cloud or HuggingFace Spaces
## Advanatced Topics
- NLP
- Feature Engineering
- PCA
- T-SNE
- simple website creation
- online database
### 👩‍💻 Author
Jessica Tran
