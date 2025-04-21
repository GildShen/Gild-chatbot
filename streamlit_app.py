import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

st.set_page_config(
        page_title='AI Course Advisor - FifiBot',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'Course advisor powered by course dataset search'
        },
        page_icon="img/favicon.ico"
    )


placeholderstr = "Please input your command"
user_name = "Fifi"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

# Load course data from CSV
@st.cache_data
def load_courses():
    return pd.read_csv("coursea_data.csv")

courses_df = load_courses()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(courses_df['course_title'])

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def search_courses(user_input, top_n=3, threshold=0.1):
    input_vec = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1]
    filtered_indices = [i for i in top_indices if cosine_sim[i] >= threshold][:top_n]
    return courses_df.iloc[filtered_indices]

def generate_response(prompt):
    pattern = r'\b(i(\'?m| am| feel| think i(\'?)?m)?\s*(so\s+)?(stupid|ugly|dumb|idiot|worthless|loser|useless))\b'
    if re.search(pattern, prompt, re.IGNORECASE):
        return "Don't say that about yourself — you're here to learn and grow! 🌱"

    results = search_courses(prompt)
    if results.empty:
        return "Sorry, I couldn’t find a matching course. Please try a different topic."

    response = "Here are some courses you might find useful:\n"
    for _, row in results.iterrows():
        response += f"\n📘 **{row['course_title']}** ({row['course_organization']})\n"
        response += f"⭐ Rating: {row['course_rating']} | Level: {row['course_difficulty']} | Enrolled: {row['course_students_enrolled']}\n"
    return response

def main():
    st.title(f" {user_name}'s AI Course Advisor Bot")

    with st.sidebar:
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=0)
        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image(user_image)
            st.markdown("**Ask me about topics like AI, Python, ML, or Calculus — and I'll recommend relevant courses!**")

    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            avatar = user_image if msg["role"] == "user" else None
            st_c_chat.chat_message(msg["role"], avatar=avatar).markdown(msg["content"])

    def chat(prompt: str):
        st_c_chat.chat_message("user", avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

if __name__ == "__main__":
    main()
