import streamlit as st
import pandas as pd
import plotly.express as px
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
import nltk

st.title("Comparison: Skip-gram vs CBOW - User Input Words")

# Download Brown corpus if not already
nltk.download('brown')
sentences = nltk.corpus.brown.sents()

# Preprocess: remove stopwords
tokenized_sentences = [
    [word for word in simple_preprocess(remove_stopwords(" ".join(sentence)))]
    for sentence in sentences
]

# --- Train models ---
skip_gram_model = Word2Vec(
    tokenized_sentences, vector_size=100, window=5, min_count=5, workers=4, sg=1
)
cbow_model = Word2Vec(
    tokenized_sentences, vector_size=100, window=5, min_count=5, workers=4, sg=0
)

# --- User input for words (no default) ---
st.sidebar.subheader("Select Words for Comparison")

word1 = st.sidebar.text_input("First Word")
word2 = st.sidebar.text_input("Second Word")

# --- Main Logic ---
if not word1 or not word2:
    st.info("✏️ Please input two words in the sidebar to compare their embeddings.")
else:
    # Check if words exist
    if word1 in skip_gram_model.wv and word2 in skip_gram_model.wv and word1 in cbow_model.wv and word2 in cbow_model.wv:

        df = pd.DataFrame({
            'Model': ['SKIP-gram', 'CBOW', 'SKIP-gram', 'CBOW'],
            'Word': [word1, word1, word2, word2],
            'X': [
                skip_gram_model.wv[word1][0], cbow_model.wv[word1][0],
                skip_gram_model.wv[word2][0], cbow_model.wv[word2][0]
            ],
            'Y': [
                skip_gram_model.wv[word1][1], cbow_model.wv[word1][1],
                skip_gram_model.wv[word2][1], cbow_model.wv[word2][1]
            ]
        })

        fig = px.scatter(df, x='X', y='Y', color='Model', hover_name='Word')
        fig.update_layout(
            title=f"Word Embedding Comparison: '{word1}' vs '{word2}'",
            width=800,
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show Similarity
        st.subheader("Similarity Comparison")
        st.write(f"Similarity (SKIP-gram) between '{word1}' and '{word2}': **{skip_gram_model.wv.similarity(word1, word2):.4f}**")
        st.write(f"Similarity (CBOW) between '{word1}' and '{word2}': **{cbow_model.wv.similarity(word1, word2):.4f}**")

    else:
        st.warning(f"❗ One or both words ('{word1}', '{word2}') not found in the vocabulary. Please try other words.")
