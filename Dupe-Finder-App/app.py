import streamlit as st
import random
from recommender import BeautyDupeRecommender


# ===========================================
# PAGE CONFIG
# ===========================================
st.set_page_config(
    page_title="DupeLab 💄",
    layout="wide"
)


# ===========================================
# LOAD MODEL (cached)
# ===========================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    model = BeautyDupeRecommender()
    model.load_data()
    model.build_features()
    return model


recommender = load_model()
df = recommender.df


# ===========================================
# HEADER
# ===========================================
st.title("💄 DupeLab")
st.caption("Find smarter, cheaper beauty dupes using AI")


# ===========================================
# SEARCH
# ===========================================
all_products = df["product_name"].tolist()

search = st.text_input("Search products 🔍")

if search:
    options = [
        p for p in all_products
        if search.lower() in str(p).lower()
    ]
else:
    options = all_products[:20]

if not options:
    st.warning("No products found.")
    st.stop()

selected = st.selectbox("Select product", options)

match = df[df["product_name"] == selected]

if match.empty:
    st.error("Product not found.")
    st.stop()

idx = match.index[0]
query = df.iloc[idx]


# ===========================================
# SAFE CARD (NO HTML)
# ===========================================
def render_card(brand, name, price, match_pct=None, tag="Top Dupe"):
    st.markdown(f"### {name}")
    st.caption(brand)
    st.write(f"💰 ${price}")

    if match_pct is not None:
        st.progress(match_pct / 100)
        st.caption(f"Match: {match_pct}%")

    st.markdown(f"**{tag}**")
    st.divider()


# ===========================================
# SELECTED PRODUCT
# ===========================================
st.subheader("Selected Product")

render_card(
    query["brand_name"],
    query["product_name"],
    query["price_usd"],
    tag="Selected"
)


# ===========================================
# FIND DUPES
# ===========================================
if st.button("Find Dupes 🔎"):

    results = recommender.get_top_dupes(idx, top_n=9)

    st.subheader("Top Dupe Matches 💡")

    cols = st.columns(3)

    for i, row in results.iterrows():
        match_pct = int(row["similarity"] * 100)

        with cols[i % 3]:
            render_card(
                row["brand"],
                row["product_name"],
                row["price"],
                match_pct=match_pct,
                tag="Top Dupe"
            )


# ===========================================
# SURPRISE ME
# ===========================================
st.divider()

if st.button("🎲 Surprise Me"):

    random_idx = random.randint(0, len(df) - 1)
    query = df.iloc[random_idx]

    st.success(f"Try this: {query['product_name']}")

    results = recommender.get_top_dupes(random_idx, top_n=9)

    cols = st.columns(3)

    for i, row in results.iterrows():
        match_pct = int(row["similarity"] * 100)

        with cols[i % 3]:
            render_card(
                row["brand"],
                row["product_name"],
                row["price"],
                match_pct=match_pct,
                tag="Recommended"
            )