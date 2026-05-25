import os

import streamlit as st
from autogen import AssistantAgent, LLMConfig, UserProxyAgent
from autogen.code_utils import content_str
from dotenv import load_dotenv

from coding.utils import AGENT_AVATARS, display_session_msg, paging, render_chat_message


load_dotenv(override=True)

OPEN_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")

placeholderstr = "Ask the assistant a question"
user_name = "Gild"
user_image = AGENT_AVATARS["User"]

llm_config_openai = LLMConfig({
    "api_type": "openai",
    "model": "gpt-4o-mini",
    "api_key": OPEN_API_KEY,
})


def save_lang():
    st.session_state["lang_setting"] = st.session_state.get("language_select")


def main():
    st.set_page_config(
        page_title="K-Assistant - Basic Agent",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://streamlit.io/",
            "Report a bug": "https://github.com",
            "About": "Basic assistant-agent classroom demo",
        },
        page_icon="img/favicon.ico",
    )

    st.title(f"{user_name}'s Basic Assistant Agent")
    st.caption("A classroom demo of one pure assistant agent answering the user's question.")

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
            st.image(user_image, caption="User")

    st_c_chat = st.container(border=True)
    display_session_msg(st_c_chat)

    if not OPEN_API_KEY:
        st.warning("OPENAI_API_KEY is not set. Add it to your .env file before using the assistant.")

    assistant = AssistantAgent(
        name="Basic_Assistant_Agent",
        system_message=(
            f"You are Basic_Assistant_Agent in a classroom demo about AI agents. "
            "Answer the user's question clearly and directly. "
            "Do not use tools. Do not ask another agent for help. "
            f"Please output in {lang_setting}. "
            "End your final response with 'ALL DONE'."
        ),
        llm_config=llm_config_openai,
        max_consecutive_auto_reply=1,
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda x: content_str(x.get("content")).find("ALL DONE") >= 0,
    )

    def generate_response(prompt):
        if not OPEN_API_KEY:
            return "OPENAI_API_KEY is not set. Please add it to your .env file and restart Streamlit."

        result = user_proxy.initiate_chat(
            recipient=assistant,
            message=prompt,
            max_turns=2,
        )
        return result.summary.replace("ALL DONE", "").strip()

    def chat(prompt: str):
        render_chat_message(st_c_chat, "user", prompt, name="User", avatar=user_image)

        response = generate_response(prompt)

        render_chat_message(
            st_c_chat,
            "assistant",
            response,
            name="Basic_Assistant_Agent",
        )

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)


if __name__ == "__main__":
    main()
