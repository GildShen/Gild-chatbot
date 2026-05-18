import os

import streamlit as st
from autogen import ConversableAgent, LLMConfig
from autogen.code_utils import content_str
from dotenv import load_dotenv

from coding.utils import AGENT_AVATARS, display_session_msg, paging, show_chat_history


load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
OPEN_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")

placeholderstr = "Enter a topic or question for the student to learn"
user_name = "Gild"
teacher_image = AGENT_AVATARS["Teacher_Agent"]
student_image = AGENT_AVATARS["Student_Agent"]

llm_config_gemini = LLMConfig({
    "api_type": "google",
    "model": "gemini-2.0-flash",
    "api_key": GEMINI_API_KEY,
})

llm_config_openai = LLMConfig({
    "api_type": "openai",
    "model": "gpt-4o-mini",
    "api_key": OPEN_API_KEY,
})

def save_lang():
    st.session_state["lang_setting"] = st.session_state.get("language_select")


def main():
    st.set_page_config(
        page_title="K-Assistant - Two-Agent Demo",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://streamlit.io/",
            "Report a bug": "https://github.com",
            "About": "Two-agent classroom demo",
        },
        page_icon="img/favicon.ico",
    )

    st.title(f"{user_name}'s Socratic Two-Agent Demo")
    st.caption("A student agent tries to learn a topic while a teacher agent guides the student with questions.")

    with st.sidebar:
        paging()

        selected_lang = st.selectbox(
            "Language",
            ["English", "Traditional Chinese"],
            index=0,
            on_change=save_lang,
            key="language_select",
        )
        lang_setting = st.session_state.get("lang_setting", selected_lang)
        st.session_state["lang_setting"] = lang_setting

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image(teacher_image, caption="Teacher_Agent")

    st_c_chat = st.container(border=True)
    display_session_msg(st_c_chat, teacher_image)

    student_persona = f"""
    You are Student_Agent in a classroom demo about AI agents.
    You are curious but not fully confident.

    Conversation behavior:
    1. Start by explaining your current, possibly incomplete understanding of the user's topic.
    2. Ask Teacher_Agent one concrete question about what confuses you.
    3. After Teacher_Agent guides you, revise your understanding in your own words.
    4. End your final message with "ALL DONE".

    Keep each message short and natural, like a student in class.
    Please output in {lang_setting}.
    """

    teacher_persona = f"""
    You are Teacher_Agent in a classroom demo about AI agents.
    Your job is to guide Student_Agent using a Socratic teaching style.

    Conversation behavior:
    1. Do not give a long lecture immediately.
    2. Identify what part of the student's understanding is correct.
    3. Ask one guiding question that helps the student reason further.
    4. Give a short clarification or example only when needed.
    5. Ask Student_Agent to revise their understanding at the end.

    Keep each message concise and beginner-friendly.
    Please output in {lang_setting}.
    """

    student_agent = ConversableAgent(
        name="Student_Agent",
        system_message=student_persona,
        llm_config=llm_config_openai,
        is_termination_msg=lambda x: content_str(x.get("content")).find("ALL DONE") >= 0,
        human_input_mode="NEVER",
    )

    teacher_agent = ConversableAgent(
        name="Teacher_Agent",
        system_message=teacher_persona,
        llm_config=llm_config_openai,
        is_termination_msg=lambda x: content_str(x.get("content")).find("ALL DONE") >= 0,
        human_input_mode="NEVER",
    )

    def generate_response(prompt):
        chat_result = student_agent.initiate_chat(
            teacher_agent,
            message=(
                "The user wants a classroom two-agent demo about this topic: "
                f"{prompt}\n\n"
                "Student_Agent should begin by stating an incomplete understanding and asking a question. "
                "Teacher_Agent should guide with one Socratic question and a short clarification. "
                "Student_Agent should finish by revising their understanding."
            ),
            max_turns=6,
            summary_method="reflection_with_llm",
        )
        return chat_result.chat_history

    def chat(prompt: str):
        response = generate_response(prompt)
        show_chat_history(
            st_c_chat,
            response,
            teacher_image,
            avatar_map={
                "Teacher_Agent": teacher_image,
                "Student_Agent": student_image,
            },
        )

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)


if __name__ == "__main__":
    main()
