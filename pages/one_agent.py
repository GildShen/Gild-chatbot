import os
from functools import wraps

import streamlit as st
from autogen import ConversableAgent, LLMConfig, UserProxyAgent
from autogen.code_utils import content_str
from dotenv import load_dotenv

from coding.agenttools import AG_search_expert, AG_search_news, AG_search_textbook, get_time
from coding.utils import AGENT_AVATARS, clean_agent_content, display_session_msg, paging, render_chat_message, split_agent_messages


load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
OPEN_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")

placeholderstr = "Ask Teacher Agent a question"
user_name = "Gild"
user_image = AGENT_AVATARS["User"]
teacher_image = AGENT_AVATARS["Teacher_Agent"]

llm_config_gemini = LLMConfig({
    "api_type": "google",
    "model": "gemini-3.0-flash",
    "api_key": GEMINI_API_KEY,
})

llm_config_openai = LLMConfig({
    "api_type": "openai",
    "model": "gpt-5-nano",
    "api_key": OPEN_API_KEY,
})


def save_lang():
    st.session_state["lang_setting"] = st.session_state.get("language_select")


def main():
    st.set_page_config(
        page_title="K-Assistant - Teacher Agent",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://streamlit.io/",
            "Report a bug": "https://github.com",
            "About": "Single teacher-agent classroom demo",
        },
        page_icon="img/favicon.ico",
    )

    st.title(f"{user_name}'s Teacher Agent")
    st.caption("A classroom demo where the user talks directly to one Teacher Agent.")
    st.caption(
        """
        Available Tools
        - `get_time`: retrieves the current date and time.
        - `AG_search_news`: searches recent Taipei Times news by keyword, section, or date range.
        - `AG_search_expert`: finds a related expert from the sample expert list.
        - `AG_search_textbook`: finds a related textbook from the sample textbook list.
        """
    )

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

    teacher_persona = f"""
    You are Teacher_Agent in a classroom demo about AI agents.
    The user is talking directly to you.

    Your job:
    1. Answer the user's question clearly and patiently.
    2. Use tools when they are helpful:
       - get_time: check the current date and time.
       - AG_search_news: search recent Taipei Times news.
       - AG_search_expert: find a related expert from the sample expert list.
       - AG_search_textbook: find a related textbook from the sample textbook list.
    3. If you use news, briefly connect it to a discipline, expert, and textbook.
    4. Keep the answer suitable for beginners and under 500 words.
    5. End your final response with "##ALL DONE##".

    Please output in {lang_setting}.
    """

    teacher_agent = ConversableAgent(
        name="Teacher_Agent",
        system_message=teacher_persona,
        llm_config=llm_config_openai,
        is_termination_msg=lambda x: content_str(x.get("content")).find("##ALL DONE##") >= 0,
        human_input_mode="NEVER",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda x: content_str(x.get("content")).find("##ALL DONE##") >= 0,
    )

    def make_tool_executor(name, func):
        @wraps(func)
        def wrapped_tool(*args, **kwargs):
            st.write(f"Using tool: `{name}`")
            return func(*args, **kwargs)

        return wrapped_tool

    def register_agent_methods(agent, proxy, methods):
        for name, description, func in methods:
            agent.register_for_llm(name=name, description=description)(func)
            proxy.register_for_execution(name=name)(make_tool_executor(name, func))

    methods_to_register = [
        ("get_time", "Retrieve the current date and time.", get_time),
        ("AG_search_expert", "Search EXPERTS_LIST by name, discipline, or interest.", AG_search_expert),
        ("AG_search_textbook", "Search TEXTBOOK_LIST by title, discipline, or related_expert.", AG_search_textbook),
        ("AG_search_news", "Search a pre-fetched news DataFrame by keywords, sections, and date range.", AG_search_news),
    ]
    register_agent_methods(teacher_agent, user_proxy, methods_to_register)

    def generate_response(prompt):
        chat_result = user_proxy.initiate_chat(
            teacher_agent,
            message=prompt,
            max_turns=6,
        )
        return chat_result.chat_history

    def extract_tool_names(chat_history):
        tool_names = []

        for entry in chat_history:
            function_call = entry.get("function_call")
            if isinstance(function_call, dict) and function_call.get("name"):
                tool_names.append(function_call["name"])

            for tool_call in entry.get("tool_calls") or []:
                function_info = tool_call.get("function", {})
                if function_info.get("name"):
                    tool_names.append(function_info["name"])

            if entry.get("role") in {"tool", "function"} and entry.get("name"):
                tool_names.append(entry["name"])

        return list(dict.fromkeys(tool_names))

    def display_teacher_messages(chat_history):
        displayed = False

        for entry in chat_history:
            if entry.get("role") == "tool":
                continue

            name = entry.get("name") or entry.get("role") or ""
            content = entry.get("content")
            if not isinstance(content, str):
                continue

            for parsed in split_agent_messages(content, name):
                parsed_name = parsed["name"]
                if "Teacher" not in parsed_name and entry.get("role") != "assistant":
                    continue

                answer = clean_agent_content(parsed["content"])
                if not answer:
                    continue

                render_chat_message(st_c_chat, "assistant", answer, name="Teacher_Agent", avatar=teacher_image)
                displayed = True

        if not displayed:
            render_chat_message(
                st_c_chat,
                "assistant",
                "Teacher Agent finished without a visible response.",
                name="Teacher_Agent",
                avatar=teacher_image,
            )

    def chat(prompt: str):
        with st.status("Teacher Agent is thinking...", expanded=True) as status:
            st.write("Reading your question.")
            response = generate_response(prompt)
            tool_names = extract_tool_names(response)
            if tool_names:
                st.write("Tools used: " + ", ".join(f"`{name}`" for name in tool_names))
            else:
                st.write("Tools used: none")
            st.write("Preparing the answer.")
            status.update(label="Teacher Agent finished.", state="complete", expanded=False)

        display_teacher_messages(response)

    if pending_prompt := st.session_state.pop("pending_teacher_prompt", None):
        chat(pending_prompt)

    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        st.session_state.setdefault("messages", []).append(
            {"role": "user", "name": "User", "content": prompt, "image": user_image}
        )
        st.session_state["pending_teacher_prompt"] = prompt
        st.rerun()


if __name__ == "__main__":
    main()
