import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from autogen import AssistantAgent, LLMConfig, UserProxyAgent
from autogen.code_utils import content_str
from dotenv import load_dotenv
from openai import OpenAI

from components.st_link_analysis_compat import EdgeStyle, NodeStyle, st_link_analysis
from coding.utils import AGENT_AVATARS, paging


load_dotenv(override=True)

OPEN_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
CV_AGENT_IMAGE = AGENT_AVATARS.get("Teacher_Agent")
CV_JSON_PATH = Path(__file__).resolve().parents[1] / "db" / "CV.json"
USER_IMAGE = AGENT_AVATARS["User"]

llm_config_openai = LLMConfig({
    "api_type": "openai",
    "model": "gpt-4o-mini",
    "api_key": OPEN_API_KEY,
})


def extract_text_from_pdf(file_bytes: bytes) -> Dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF parsing requires pymupdf. Install it with: pip install pymupdf") from exc

    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": index, "text": text})

    return {"pages": pages}


def extract_text_from_docx(file_bytes: bytes) -> Dict[str, Any]:
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError("DOCX parsing requires python-docx. Install it with: pip install python-docx") from exc

    document = docx.Document(BytesIO(file_bytes))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return {"pages": [{"page": 1, "text": "\n".join(paragraphs)}]}


def decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp950", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def load_uploaded_cv(uploaded_file) -> Dict[str, Any]:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix == "pdf":
        parsed = extract_text_from_pdf(file_bytes)
    elif suffix == "docx":
        parsed = extract_text_from_docx(file_bytes)
    elif suffix == "json":
        parsed = json.loads(decode_text(file_bytes))
    elif suffix in {"txt", "md"}:
        parsed = {"pages": [{"page": 1, "text": decode_text(file_bytes).strip()}]}
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, TXT, MD, or JSON.")

    if isinstance(parsed, dict):
        parsed.setdefault("filename", filename)
        return parsed

    return {"filename": filename, "pages": [{"page": 1, "text": json.dumps(parsed, ensure_ascii=False)}]}


def compact_pages(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = json_data.get("pages", [])
    if isinstance(pages, list):
        return [
            {"page": item.get("page", 1), "text": str(item.get("text", "")).strip()}
            for item in pages
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]

    text = json.dumps(json_data, ensure_ascii=False)
    return [{"page": 1, "text": text}]


def refine_cv_content(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract CV sections into a fixed schema.
    Each item includes text, page, and title.
    """
    if not OPEN_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file and restart Streamlit.")

    refine_template = """You are an expert resume parser.
Below are brief definitions for each CV section to extract:
- PERSONAL INFORMATION: personal details (name, address, email, phone)
- AFFILIATION: organizations or institutions (employers, schools)
- CHARACTERISTICS: personal traits or attributes
- RESEARCH: research topics or interests
- AWARD: awards received
- HONOR: honors or recognitions
- SKILL: professional or technical skills
- EXPERIENCE: work or project experience
- PUBLICATION: published works
- INTEREST: hobbies or areas of interest
- EDUCATION: degrees and academic background
- OTHERS: any additional relevant information

For each item, include:
- text: the exact extracted content
- page: the page number
- title: a concise distilled concept of the item based on its text

Output strictly JSON following this schema. Omit any section or item that is not present.
Every top-level key must be one of:
PERSONAL INFORMATION, AFFILIATION, CHARACTERISTICS, RESEARCH, AWARD, HONOR, SKILL, EXPERIENCE,
PUBLICATION, INTEREST, EDUCATION, OTHERS.

Each section must have this shape:
{
  "items": [
    {
      "text": "exact extracted content",
      "page": 1,
      "title": "brief distilled concept"
    }
  ]
}
"""

    payload = {
        "filename": json_data.get("filename"),
        "pages": compact_pages(json_data),
    }

    client = OpenAI(api_key=OPEN_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": refine_template},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def build_cv_graph(result: Dict[str, Any]) -> Dict[str, List[Dict[str, Dict[str, Any]]]]:
    nodes = [{"data": {"id": "cv", "label": "CV", "name": "CV"}}]
    edges = []

    for section, section_data in result.items():
        if not isinstance(section_data, dict):
            continue

        items = section_data.get("items", [])
        if not isinstance(items, list) or not items:
            continue

        section_id = f"section::{section}"
        nodes.append(
            {
                "data": {
                    "id": section_id,
                    "label": "SECTION",
                    "name": section.title(),
                    "section": section,
                }
            }
        )
        edges.append(
            {
                "data": {
                    "id": f"edge::cv::{section_id}",
                    "source": "cv",
                    "target": section_id,
                    "label": "HAS_SECTION",
                    "name": "has section",
                }
            }
        )

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            item_id = f"item::{section}::{index}"
            title = str(item.get("title") or section.title())
            text = str(item.get("text") or "")
            nodes.append(
                {
                    "data": {
                        "id": item_id,
                        "label": "ITEM",
                        "name": title[:80],
                        "section": section,
                        "page": item.get("page"),
                        "text": text,
                    }
                }
            )
            edges.append(
                {
                    "data": {
                        "id": f"edge::{section_id}::{item_id}",
                        "source": section_id,
                        "target": item_id,
                        "label": "HAS_ITEM",
                        "name": "has item",
                    }
                }
            )

    return {"nodes": nodes, "edges": edges}


def render_cv_graph(result: Dict[str, Any]) -> None:
    elements = build_cv_graph(result)
    if len(elements["nodes"]) <= 1:
        st.info("No graphable CV sections found.")
        return

    selected = st_link_analysis(
        elements,
        layout="cose",
        node_styles=[
            NodeStyle("CV", color="#60A5FA", fill="#1E3A8A", caption="name", icon="description", size=58),
            NodeStyle("SECTION", color="#34D399", fill="#064E3B", caption="name", icon="folder", size=46),
            NodeStyle(
                "ITEM",
                color="#FBBF24",
                fill="#78350F",
                caption="name",
                icon="badge",
                size=36,
                text_wrap="wrap",
                text_max_width=130,
                font_size=11,
            ),
        ],
        edge_styles=[
            EdgeStyle("HAS_SECTION", color="#93C5FD", caption="name", directed=True, weight=2),
            EdgeStyle("HAS_ITEM", color="#FCD34D", caption="name", directed=True, weight=1.5),
        ],
        height=650,
        key="cv_graph",
    )

    if selected:
        st.caption(f"Selected graph data: `{json.dumps(selected, ensure_ascii=False)[:500]}`")


def count_cv_sections(result: Dict[str, Any]) -> int:
    return sum(
        1
        for section_data in result.values()
        if isinstance(section_data, dict) and isinstance(section_data.get("items"), list) and section_data["items"]
    )


def save_parsed_cv_json(result: Dict[str, Any]) -> None:
    CV_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    CV_JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def load_parsed_cv_json() -> Dict[str, Any]:
    if not CV_JSON_PATH.exists():
        raise FileNotFoundError("db/CV.json does not exist yet. Parse a CV first.")

    parsed = json.loads(CV_JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("db/CV.json must be a JSON object.")
    return parsed


def cv_context_text(result: Dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def display_cv_messages(container_obj) -> None:
    for message in st.session_state.setdefault("cv_messages", []):
        role = message.get("role", "user")
        content = message.get("content", "")
        avatar = message.get("avatar")

        if avatar:
            container_obj.chat_message(role, avatar=avatar).markdown(content)
        else:
            container_obj.chat_message(role).markdown(content)


def render_cv_message(container_obj, role: str, content: str, avatar=None) -> None:
    message = {"role": role, "content": content}
    if avatar:
        message["avatar"] = avatar

    st.session_state.setdefault("cv_messages", []).append(message)

    if avatar:
        container_obj.chat_message(role, avatar=avatar).write(content)
    else:
        container_obj.chat_message(role).write(content)


def generate_cv_answer(prompt: str, cv_data: Dict[str, Any], language: str) -> str:
    if not OPEN_API_KEY:
        return "OPENAI_API_KEY is not set. Please add it to your .env file and restart Streamlit."

    assistant = AssistantAgent(
        name="CV_Agent",
        system_message=(
            "You are CV_Agent, a classroom demo agent that answers questions using only the loaded CV JSON. "
            "Use the CV JSON as your only source of truth. "
            "If the answer is not supported by the CV JSON, say that the CV does not provide that information. "
            "Be concise, factual, and cite the relevant section names when useful. "
            f"Please output in {language}. "
            "End your final response with 'ALL DONE'.\n\n"
            f"Loaded CV JSON:\n{cv_context_text(cv_data)}"
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

    result = user_proxy.initiate_chat(
        recipient=assistant,
        message=prompt,
        max_turns=2,
    )
    return result.summary.replace("ALL DONE", "").strip()


def render_cv_chat(result: Dict[str, Any], language: str) -> None:
    chat_container = st.container(border=True)
    display_cv_messages(chat_container)

    if not OPEN_API_KEY:
        st.warning("OPENAI_API_KEY is not set. Add it to your .env file before using the assistant.")

    if prompt := st.chat_input("Ask a question about this CV", key="cv_agent_chat"):
        render_cv_message(chat_container, "user", prompt, avatar=USER_IMAGE)
        answer = generate_cv_answer(prompt, result, language)
        render_cv_message(chat_container, "assistant", answer)


@st.dialog("Upload and Parse CV", width="large")
def parse_cv_dialog() -> None:
    uploaded_file = st.file_uploader(
        "Upload CV",
        type=["pdf", "docx", "txt", "md", "json"],
        accept_multiple_files=False,
        key="cv_parse_upload",
    )

    if not uploaded_file:
        return

    try:
        cv_data = load_uploaded_cv(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        return

    with st.expander("Extracted text preview", expanded=False):
        st.json(cv_data)

    if st.button("Parse CV", type="primary", use_container_width=True):
        with st.status("CV Agent is parsing the file...", expanded=True) as status:
            st.write("Reading extracted text.")
            try:
                result = refine_cv_content(cv_data)
            except Exception as exc:
                status.update(label="CV parsing failed.", state="error", expanded=True)
                st.error(str(exc))
                return

            st.write("Structuring CV sections.")
            save_parsed_cv_json(result)
            st.write("Saved parsed result to `db/CV.json`.")
            status.update(label="CV parsing finished.", state="complete", expanded=False)

        st.session_state["cv_parse_result"] = result
        st.session_state["cv_messages"] = []
        st.rerun()


@st.dialog("Load CV", width="small")
def load_cv_dialog() -> None:
    st.caption("Load the saved parsed CV from `db/CV.json`.")

    if not CV_JSON_PATH.exists():
        st.warning("db/CV.json does not exist yet. Parse a CV first.")
        return

    if st.button("Load db/CV.json", type="primary", use_container_width=True):
        try:
            st.session_state["cv_parse_result"] = load_parsed_cv_json()
            st.session_state["cv_messages"] = []
        except Exception as exc:
            st.error(str(exc))
            return

        st.rerun()


@st.dialog("CV Graph", width="large")
def cv_graph_dialog() -> None:
    result = st.session_state.get("cv_parse_result")
    if not result:
        st.warning("No CV is loaded.")
        return

    st.caption(f"Source: `{CV_JSON_PATH.relative_to(Path(__file__).resolve().parents[1])}`")
    render_cv_graph(result)

    with st.expander("Parsed CV JSON", expanded=False):
        st.json(result)


def main():
    st.set_page_config(
        page_title="K-Assistant - CV Agent",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://streamlit.io/",
            "Report a bug": "https://github.com",
            "About": "CV parsing classroom demo",
        },
        page_icon="img/favicon.ico",
    )

    st.title("CV Agent")
    st.caption("Ask questions about a parsed CV stored in `db/CV.json`.")

    with st.sidebar:
        paging()
        selected_lang = st.selectbox(
            "Language",
            ["English", "Traditional Chinese"],
            index=0,
            key="cv_language_select",
        )
        st.image(CV_AGENT_IMAGE, caption="CV Agent")

    if "cv_parse_result" not in st.session_state and CV_JSON_PATH.exists():
        try:
            st.session_state["cv_parse_result"] = load_parsed_cv_json()
        except Exception:
            pass

    result = st.session_state.get("cv_parse_result")

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Upload / Parse", type="primary", use_container_width=True):
            parse_cv_dialog()
    with action_cols[1]:
        if st.button("Load CV", use_container_width=True, disabled=not CV_JSON_PATH.exists()):
            load_cv_dialog()
    with action_cols[2]:
        if st.button("CV Graph", use_container_width=True, disabled=not bool(result)):
            cv_graph_dialog()

    if result:
        st.caption(
            f"Loaded `db/CV.json` with {count_cv_sections(result)} sections. "
            "Use the buttons above to replace, reload, or inspect the graph."
        )
    else:
        st.info("No CV is loaded. Upload and parse a CV, or load `db/CV.json` if it exists.")
        return

    st.divider()
    st.subheader("Ask CV Agent")
    st.caption("The agent answers only from the loaded CV JSON.")

    if result:
        render_cv_chat(result, selected_lang)


if __name__ == "__main__":
    main()
