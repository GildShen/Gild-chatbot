import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

st.title("Q3: CBOW Model - Adjust Parameters and Display Selected Sentence Lines")

# --- Initialize ---
if "sentences" not in st.session_state:
    st.session_state.sentences = []

# --- Sidebar adjustable parameters ---
st.sidebar.subheader("Adjust Word2Vec Parameters (CBOW)")

vector_size = st.sidebar.slider(
    "Vector Size", min_value=10, max_value=300, value=100, step=10
)
window_size = st.sidebar.slider(
    "Window Size", min_value=1, max_value=10, value=5, step=1
)

chart_type = st.sidebar.selectbox("Chart Type", ["2D", "3D"])

# --- Functions ---
def train_model(sentences, vector_size, window_size):
    tokenized = [simple_preprocess(remove_stopwords(s)) for s in sentences]
    model = Word2Vec(
        tokenized,
        vector_size=vector_size,
        window=window_size,
        min_count=1,
        workers=4,
        sg=0  # sg=0 for CBOW
    )
    return model, tokenized

def reduce_vectors(model):
    vectors = np.array([model.wv[w] for w in model.wv.index_to_key])
    if vectors.shape[0] < 3:
        return None
    else:
        return PCA(n_components=3).fit_transform(vectors)

def get_color_map():
    return {
        0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange',
        5: 'cyan', 6: 'magenta', 7: 'yellow', 8: 'brown', 9: 'pink'
    }

def plot_2d(model, reduced_vectors, tokenized_sentences, selected_sentences):
    color_map = get_color_map()

    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_map[i % len(color_map)])
                break

    scatter = go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=8),
        hovertemplate="Word: %{text}"
    )

    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        if f"Sentence {i+1}" in selected_sentences:
            vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence if word in model.wv]
            if len(vectors) > 1:
                line = go.Scatter(
                    x=[v[0] for v in vectors],
                    y=[v[1] for v in vectors],
                    mode='lines',
                    name=f"Sentence {i+1}",
                    line=dict(color=color_map[i % len(color_map)], width=2),
                    showlegend=True
                )
                line_traces.append(line)

    fig = go.Figure(data=[scatter] + line_traces)
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title=f"2D Word Embeddings (CBOW, vector_size={vector_size}, window_size={window_size})",
        width=800,
        height=800
    )
    return fig

def plot_3d(model, reduced_vectors, tokenized_sentences, selected_sentences):
    color_map = get_color_map()

    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_map[i % len(color_map)])
                break

    word_ids = [f"word-{i}" for i in range(len(model.wv.index_to_key))]

    scatter = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=4),
        customdata=word_colors,
        ids=word_ids,
        hovertemplate="Word: %{text}<br>Color: %{customdata}"
    )

    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        if f"Sentence {i+1}" in selected_sentences:
            vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence if word in model.wv]
            if len(vectors) > 1:
                line = go.Scatter3d(
                    x=[v[0] for v in vectors],
                    y=[v[1] for v in vectors],
                    z=[v[2] for v in vectors],
                    mode='lines',
                    line=dict(color=color_map[i % len(color_map)], dash='solid'),
                    name=f"Sentence {i+1}",
                    showlegend=True,
                    hoverinfo='none'
                )
                line_traces.append(line)

    fig = go.Figure(data=[scatter] + line_traces)

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title=f"3D Word Embeddings (CBOW, vector_size={vector_size}, window_size={window_size})",
        width=1000,
        height=1000
    )
    return fig

# --- Chat Input for new sentences ---
if prompt := st.chat_input("Type your sentence here:"):
    st.session_state.sentences.append(prompt)

# --- Display ---
if len(st.session_state.sentences) == 0:
    st.warning("⚠️ Please input a sentence first.")
else:
    model, tokenized_sentences = train_model(st.session_state.sentences, vector_size, window_size)
    reduced_vectors = reduce_vectors(model)

    if reduced_vectors is None:
        st.warning("❗ Not enough words to perform PCA. Please input more sentences.")
    else:
        # Sidebar: choose which sentences to display lines for
        available_sentences = [f"Sentence {i+1}" for i in range(len(tokenized_sentences))]
        selected_sentences = st.sidebar.multiselect(
            "Select Sentences to Draw Lines",
            options=available_sentences,
            default=available_sentences  # Default to show all
        )

        if chart_type == "2D":
            fig = plot_2d(model, reduced_vectors, tokenized_sentences, selected_sentences)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plot_3d(model, reduced_vectors, tokenized_sentences, selected_sentences)
            st.plotly_chart(fig, use_container_width=True)

# --- Word Query Section ---
st.subheader("🔎 Check Word Vector and Similar Words")

query_word = st.text_input("Enter a word to check:")

if query_word:
    if query_word in model.wv:
        st.write(f"**Vector for '{query_word}'** (showing first 5 dimensions):")
        st.write(model.wv[query_word][:5])  # Only show first 5 numbers to keep it clean

        similar_words = model.wv.most_similar(query_word)
        st.write(f"**Most similar words to '{query_word}':**")
        st.table(similar_words)
    else:
        st.warning(f"⚠️ The word '{query_word}' is not in the vocabulary. Please try another.")