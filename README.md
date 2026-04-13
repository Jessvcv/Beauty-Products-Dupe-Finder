# Beauty-Products-Dupe-Finder

A content-based recommendation system that identifies **affordable skincare and beauty product dupes** using ingredient similarity, natural language processing, and price-aware ranking.

---

## 🚀 Overview

Consumers often search for “dupes” — lower-cost alternatives to expensive beauty products with similar formulations. However, identifying these alternatives typically requires manual comparison or unreliable online recommendations.

This project solves that problem by building a **content-based recommender system** that automatically finds similar products based on their **ingredient lists and product features**, while prioritizing **cheaper alternatives**.

---

## 🎯 Key Features

- 🧠 **Hybrid similarity model**
  - TF-IDF (ingredient overlap)
  - Sentence-BERT embeddings (semantic similarity)
  - Jaccard similarity (ingredient set overlap)

- 💰 **Price-aware ranking**
  - Prioritizes cheaper alternatives when recommending dupes

- 🔍 **Smart filtering**
  - Same product category
  - Option to return only cheaper products

- 📊 **Evaluation metric**
  - Measures % of recommended products that are cheaper

- 🌐 **Interactive web app**
  - Built with Streamlit for real-time recommendations

---

## 🧠 Methodology

### 1. Data Preprocessing
- Lowercasing and cleaning ingredient text
- Removing noise and invalid entries
- Standardizing ingredient names (e.g., "aqua" → "water")

### 2. Feature Engineering
- Ingredient tokenization
- Ingredient set construction for overlap comparison

### 3. Similarity Modeling

We compute similarity using three approaches:

- **TF-IDF + Cosine Similarity**
- **Sentence-BERT Embeddings**
- **Jaccard Similarity**

### 4. Hybrid Model

Final similarity is computed as a weighted combination: Hybrid = w1(TF-IDF) + w2(Embeddings) + w3(Jaccard)


### 5. Price-Aware Ranking
Final Score = 0.8 * Similarity + 0.2 * Price Score


---

## 📊 Dataset

- Source: Kaggle — Sephora Products and Skincare Reviews
- ~8,000+ products
- Features used:
  - Product name
  - Brand name
  - Category
  - Ingredients
  - Highlights
  - Price (USD)

---

## 🌐 Streamlit App

The application allows users to:

- Select a product
- View product details
- Receive top dupe recommendations
- Compare price and similarity scores

---

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- SentenceTransformers
- Streamlit
- KaggleHub

---

## 📁 Project Structure
beauty-dupe-app/
│
├── app.py # Streamlit frontend
├── model.py # ML pipeline + recommender
├── requirements.txt # Dependencies
└── README.md

---

## ▶️ How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
streamlit run app.py
```
---

📈 Evaluation

The model is evaluated using:

Top-K similarity scoring
Percentage of recommendations that are cheaper than the original product

This ensures recommendations are both relevant and cost-effective.

💡 Key Insights
Ingredient similarity alone is not enough → embeddings improve performance
Hybrid models outperform single-method approaches
Price-aware ranking makes recommendations more practical
Ingredient normalization is essential due to inconsistent naming (~15k unique ingredients)

🔮 Future Work
Add explanation system (“Why this is a dupe”)
Improve weight optimization with learning-to-rank methods
Expand dataset (Ulta, drugstore brands)
Precompute similarity for faster deployment
Add user personalization features

👩‍💻 Author
Jessica Tran
Computer Science / Data Science Student
Belmont University

⭐ Acknowledgments
Kaggle dataset contributors
SentenceTransformers
Scikit-learn
Streamlit
