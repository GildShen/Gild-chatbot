import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

st.title("Q1-1: Word2Vec 2D/3D Visualization with Selectable Sentence Lines")

# --- Color Map ---
def get_color_map():
    return {
        0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange',
        5: 'cyan', 6: 'magenta', 7: 'yellow', 8: 'brown', 9: 'pink'
    }

# --- Training and Reduction ---
def train_model(sentences):
    tokenized_sentences = [simple_preprocess(s) for s in sentences]
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model, tokenized_sentences

def reduce(model):
    vectors = np.array([model.wv[w] for w in model.wv.index_to_key])
    return PCA(n_components=3).fit_transform(vectors)

# --- Plotting ---
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
                    mode='lines+markers',
                    name=f"Sentence {i+1}",
                    line=dict(color=color_map[i % len(color_map)], width=2),
                    showlegend=True
                )
                line_traces.append(line)

    fig = go.Figure(data=[scatter] + line_traces)
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title="2D Word Embeddings",
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

    # --- Line traces for each selected sentence ---
    line_traces = []
    for i, sentence in enumerate(tokenized_sentences):
        if f"Sentence {i+1}" in selected_sentences:
            vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence if word in model.wv]
            if len(vectors) >= 2:  # Must have at least 2 points to draw a line
                line_trace = go.Scatter3d(
                    x=[v[0] for v in vectors],
                    y=[v[1] for v in vectors],
                    z=[v[2] for v in vectors],
                    mode='lines',
                    line=dict(color=color_map[i % len(color_map)], width=4, dash='solid'),
                    name=f"Sentence {i+1}",
                    showlegend=True,
                    hoverinfo='none'
                )
                line_traces.append(line_trace)

    fig = go.Figure(data=[scatter] + line_traces)

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Visualization of Word Embeddings",
        width=1000,
        height=1000
    )
    return fig

# --- Streamlit App Logic ---

if "sentences" not in st.session_state:
    st.session_state.sentences = []

chart_type = st.sidebar.selectbox("Chart Type", ["2D", "3D"])

if prompt := st.chat_input("Type your sentence here:"):
    st.session_state.sentences.append(prompt)

if len(st.session_state.sentences) == 0:
    st.warning("⚠️ Please input a sentence first.")
else:
    model, tokenized_sentences = train_model(st.session_state.sentences)
    reduced_vectors = reduce(model)

    # Sidebar: Choose which sentences to display lines for
    available_sentences = [f"Sentence {i+1}" for i in range(len(tokenized_sentences))]
    selected_sentences = st.sidebar.multiselect(
        "Select Sentences to Draw Lines",
        options=available_sentences,
        default=available_sentences  # by default, show all lines
    )

    if chart_type == "2D":
        fig = plot_2d(model, reduced_vectors, tokenized_sentences, selected_sentences)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plot_3d(model, reduced_vectors, tokenized_sentences, selected_sentences)
        st.plotly_chart(fig, use_container_width=True)
