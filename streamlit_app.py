import streamlit as st
import time
import re
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

# --- Word2Vec Training and Plotting Functions ---

def train_word2vec_model(sentences, sg_flag):
    tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences]
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2, sg=sg_flag)
    return model, tokenized_sentences

def reduce_vectors(model):
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(word_vectors)
    return reduced_vectors

def get_color_map():
    return {
        0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange',
        5: 'cyan', 6: 'magenta', 7: 'yellow', 8: 'brown', 9: 'pink'
    }

def plot_word2vec_2d(model, reduced_vectors, tokenized_sentences):
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
        line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence if word in model.wv]
        if len(line_vectors) > 1:
            line_trace = go.Scatter(
                x=[vector[0] for vector in line_vectors],
                y=[vector[1] for vector in line_vectors],
                mode='lines',
                line=dict(color=color_map[i % len(color_map)], width=1, dash='solid'),
                showlegend=False
            )
            line_traces.append(line_trace)

    fig = go.Figure(data=[scatter] + line_traces)
    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title="2D Word Embeddings (Updated)",
        width=800,
        height=800
    )
    return fig

def plot_word2vec_3d(model, reduced_vectors, tokenized_sentences):
    color_map = get_color_map()

    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(color_map[i % len(color_map)])
                break

    scatter = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        textposition='top center',
        marker=dict(color=word_colors, size=4)
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Word Embeddings (Updated)",
        width=800,
        height=800
    )
    return fig

# --- Chatbot Setup ---

user_name = "Fifi"
user_image = "https://www.w3schools.com/howto/img_avatar.png"
placeholderstr = "Type a sentence to update Word2Vec plot"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def main():
    st.set_page_config(
        page_title='K-Assistant - Word2Vec Trainer (SKIP-GRAM / CBOW)',
        layout='wide',
        initial_sidebar_state='auto',
        page_icon="img/favicon.ico"
    )

    st.title(f"💬 {user_name}'s Word2Vec Chat Trainer")

    if "sentences" not in st.session_state:
        st.session_state.sentences = []  # No preset sentences

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=0)
        model_type = st.selectbox("Model Type", ["SKIP-GRAM", "CBOW"])
        chart_type = st.selectbox("Chart Type", ["2D", "3D"])

    st_c_chat = st.container(border=True)

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st_c_chat.chat_message(msg["role"], avatar=user_image).markdown(msg["content"])
        elif msg["role"] == "assistant":
            st_c_chat.chat_message(msg["role"]).markdown(msg["content"])

    def chat(prompt: str):
        st_c_chat.chat_message("user", avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Add user sentence to Word2Vec training corpus
        st.session_state.sentences.append(prompt)

        response = f"Sentence added to Word2Vec model!"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

    # --- Update and Display the Word2Vec Visualization ---

    if len(st.session_state.sentences) == 0:
        st.warning("⚠️ No sentences yet. Please input a sentence to start training Word2Vec.")
    else:
        sg_flag = 1 if model_type == "SKIP-GRAM" else 0
        model, tokenized_sentences = train_word2vec_model(st.session_state.sentences, sg_flag)
        reduced_vectors = reduce_vectors(model)

        if chart_type == "2D":
            fig = plot_word2vec_2d(model, reduced_vectors, tokenized_sentences)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "3D":
            fig = plot_word2vec_3d(model, reduced_vectors, tokenized_sentences)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
