import streamlit as st

st.set_page_config(
    page_title="KA Homework - Word2Vec Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

st.title("📚 Word2Vec Homework Dashboard")

st.write("""
Use the sidebar on the left to navigate through the following tasks:
- **Q1-1:** Basic Word2Vec 2D/3D visualization
- **Q2:** Skip-gram model with modified parameters
- **Q3:** CBOW model with modified parameters
- **Comparison:** Compare Skip-gram vs CBOW embeddings

Type your sentences inside each page to update results live!
""")
