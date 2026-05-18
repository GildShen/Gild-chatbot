import json
import os

import streamlit as st
from autogen import ConversableAgent, LLMConfig
from autogen.code_utils import content_str
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from dotenv import load_dotenv

from coding.utils import AGENT_AVATARS, display_session_msg, paging, save_messages_to_json, show_chat_history


load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
OPEN_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")

placeholderstr = "Ask the agent group a question"
user_name = "Gild"
teacher_image = AGENT_AVATARS["Teacher_Agent"]
student_image = AGENT_AVATARS["Student_Agent"]
tech_image = AGENT_AVATARS["Tech_Agent"]
general_image = AGENT_AVATARS["General_Agent"]

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
        page_title="K-Assistant - Group Agents",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://streamlit.io/",
            "Report a bug": "https://github.com",
            "About": "Multi-agent classroom demo",
        },
        page_icon="img/favicon.ico",
    )

    st.title(f"{user_name}'s Group Agents")
    st.caption("A classroom demo where multiple agents discuss the same user question.")

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

    teacher_agent = ConversableAgent(
        name="Teacher_Agent",
        system_message=(
            "You are Teacher_Agent in a classroom demo about AI agents. "
            "Explain the user's question clearly for beginners. "
            "Focus on the main concept and keep your response concise. "
            f"Please output in {lang_setting}."
        ),
        llm_config=llm_config_openai,
        human_input_mode="NEVER",
    )

    tech_agent = ConversableAgent(
        name="Tech_Agent",
        system_message=(
            "You are Tech_Agent in a classroom demo about AI agents. "
            "Add a practical technical perspective, such as implementation details, "
            "limitations, or engineering tradeoffs. "
            f"Please output in {lang_setting}."
        ),
        llm_config=llm_config_openai,
        human_input_mode="NEVER",
    )

    general_agent = ConversableAgent(
        name="General_Agent",
        system_message=(
            "You are General_Agent in a classroom demo about AI agents. "
            "Add a simple everyday perspective that a non-technical learner can understand. "
            f"Please output in {lang_setting}."
        ),
        llm_config=llm_config_openai,
        human_input_mode="NEVER",
    )

    student_agent = ConversableAgent(
        name="Student_Agent",
        system_message=(
            "You are Student_Agent in a classroom demo about AI agents. "
            "After hearing the other agents, summarize the key takeaway in a short paragraph. "
            "End your final message with 'ALL DONE'. "
            f"Please output in {lang_setting}."
        ),
        llm_config=llm_config_openai,
        human_input_mode="NEVER",
        is_termination_msg=lambda x: content_str(x.get("content")).find("ALL DONE") >= 0,
    )

    user_agent = ConversableAgent(
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: content_str(x.get("content")).find("ALL DONE") >= 0,
    )

    pattern = AutoPattern(
        initial_agent=teacher_agent,
        agents=[teacher_agent, tech_agent, general_agent, student_agent],
        user_agent=user_agent,
        group_manager_args={"llm_config": llm_config_openai},
    )

    def generate_response(prompt):
        chat_result, _, _ = initiate_group_chat(
            pattern=pattern,
            messages=(
                "Discuss this user question as a classroom group-agent demo. "
                "Teacher_Agent should explain first, Tech_Agent and General_Agent should add perspectives, "
                f"and Student_Agent should summarize at the end: {prompt}"
            ),
            max_rounds=8,
        )
        return chat_result.chat_history

    def chat(prompt: str):
        response = generate_response(prompt)
        conv_res = show_chat_history(
            st_c_chat,
            response,
            teacher_image,
            avatar_map={
                "Teacher_Agent": teacher_image,
                "Student_Agent": student_image,
                "Tech_Agent": tech_image,
                "General_Agent": general_image,
            },
        )
        messages = json.loads(conv_res)
        file_path = save_messages_to_json(messages, output_dir="chat_logs")
        st.write(f"Saved chat history to `{file_path}`")

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)


if __name__ == "__main__":
    main()
