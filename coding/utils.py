import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

AGENT_PREFIX_RE = re.compile(r"(Basic_Assistant_Agent|Teacher_Agent|Student_Agent|Tech_Agent|General_Agent):\s*")

AGENT_AVATARS = {
    "User": "https://www.w3schools.com/w3images/avatar2.png",
    "Teacher_Agent": "https://www.w3schools.com/w3images/avatar3.png",
    "Student_Agent": "https://www.w3schools.com/w3images/avatar6.png",
    "Tech_Agent": "https://www.w3schools.com/w3images/avatar5.png",
    "General_Agent": "https://www.w3schools.com/w3images/avatar4.png",
}


def paging():
    st.page_link("streamlit_app.py", label="Home")
    st.page_link("pages/one_agent.py", label="Teacher Agent")
    st.page_link("pages/two_agents.py", label="Two Agents")
    st.page_link("pages/group_agents.py", label="Group Agents")


def display_session_msg(container_obj, user_image: Optional[str] = None):
    messages = st.session_state.setdefault("messages", [])

    for msg in messages:
        role = msg.get("role", "user")
        name = msg.get("name", "")
        content = msg.get("content", "")
        avatar = msg.get("image")

        if not avatar and name:
            avatar = AGENT_AVATARS.get(name)

        if not avatar and role == "assistant" and user_image:
            avatar = user_image

        if avatar:
            container_obj.chat_message(role, avatar=avatar).markdown(content)
        else:
            container_obj.chat_message(role).markdown(content)


def render_chat_message(container_obj, role: str, content: str, name: str = "", avatar: Optional[str] = None) -> Dict[str, Any]:
    if not avatar and name:
        avatar = AGENT_AVATARS.get(name)

    message = {"role": role, "name": name or role, "content": content}
    if avatar:
        message["image"] = avatar

    st.session_state.setdefault("messages", []).append(message)

    if avatar:
        container_obj.chat_message(role, avatar=avatar).write(content)
    else:
        container_obj.chat_message(role).write(content)

    return message


def clean_agent_content(content: str) -> str:
    return content.replace("##ALL DONE##", "").replace("ALL DONE", "").strip()


def is_internal_prompt(content: str) -> bool:
    return content.startswith("The user wants a classroom") or content.startswith("Discuss this user question")


def split_agent_messages(content: str, fallback_name: str) -> List[Dict[str, str]]:
    content = clean_agent_content(content)
    if not content or is_internal_prompt(content):
        return []

    parts = AGENT_PREFIX_RE.split(content)
    if len(parts) < 3:
        return [{"name": fallback_name, "content": content}]

    messages = []
    for idx in range(1, len(parts), 2):
        name = parts[idx]
        message = clean_agent_content(parts[idx + 1])
        if message:
            messages.append({"name": name, "content": message})

    return messages


def role_for_agent(name: str) -> str:
    if name in {"Teacher_Agent", "Basic_Assistant_Agent", "Tech_Agent", "General_Agent"}:
        return "assistant"
    return "user"


def show_chat_history(
    container_obj,
    chat_history: List[Dict[str, Any]],
    user_image=None,
    avatar_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Display non-tool messages from an AutoGen chat history and return the same
    messages as JSON so classroom demos can save the visible conversation.
    """
    st.session_state.setdefault("messages", [])
    processed = []

    for entry in chat_history:
        if entry.get("role") == "tool":
            continue

        content = entry.get("content")
        if not isinstance(content, str):
            continue

        fallback_name = entry.get("name") or entry.get("role") or "agent"
        for parsed in split_agent_messages(content, fallback_name):
            name = parsed["name"]
            role = role_for_agent(name)
            avatar = (avatar_map or {}).get(name)
            if not avatar and role == "assistant" and user_image:
                avatar = user_image

            message = render_chat_message(container_obj, role, parsed["content"], name=name, avatar=avatar)
            processed.append(message)

    return json.dumps(processed, ensure_ascii=False, indent=2)


def save_messages_to_json(
    messages: List[Dict[str, Any]],
    output_dir: str = ".",
) -> str:
    """
    Save chat messages to a timestamped JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    filename = f"{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    return filepath
