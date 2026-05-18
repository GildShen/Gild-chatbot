from typing import List, Dict, Optional, Any, Annotated
from coding.tools import search_expert, search_textbook, search_news, fetch_all_news
from datetime import datetime
import streamlit as st


@st.cache_data(ttl=600, show_spinner="Fetching recent news...")
def cached_fetch_news():
    return fetch_all_news(1, 1, list_type="all") ## How many news to fetch? Adjust as needed.


def AG_search_expert(
    name: Annotated[Optional[str], "Expert name."] = None,
    discipline: Annotated[Optional[List[str]], "List of input strings containing disciplines to filter by."] = None,
    interest: Annotated[Optional[List[str]], "List of input strings containing interests to filter by."] = None
):
    """
    Wrapper around search_expert that accepts lists for discipline and interest.
    """
    # If multiple disciplines or interests are provided, return experts matching ANY of them
    matched = []
    for d in (discipline or []):
        matched.extend(search_expert(name=name, discipline=d, interest=None))
    for i in (interest or []):
        matched.extend(search_expert(name=name, discipline=None, interest=i))
    # If no lists provided, fall back to single-value search
    if not discipline and not interest:
        matched = search_expert(name=name)
    # Deduplicate results
    unique = {expert["EMAIL"]: expert for expert in matched if "EMAIL" in expert}
    return list(unique.values())

def AG_search_textbook(
    title: Annotated[Optional[str], "Textbook title."] = None,
    discipline: Annotated[Optional[List[str]], "List of input strings containing disciplines to filter by."] = None,
    related_expert: Annotated[Optional[List[str]], "List of input strings containing related expert names to filter by."] = None
):
    """
    Wrapper around search_textbook that accepts lists for discipline and related_expert.
    """
    matched = []
    for d in (discipline or []):
        matched.extend(search_textbook(title=title, discipline=d, related_expert=None))
    for e in (related_expert or []):
        matched.extend(search_textbook(title=title, discipline=None, related_expert=e))
    if not discipline and not related_expert:
        matched = search_textbook(title=title)
    unique = {tb["TITLE"]: tb for tb in matched if "TITLE" in tb}
    return list(unique.values())

def AG_search_news(
    query: Annotated[
        Optional[str],
        "Keyword or phrase to search for; if None and no other filters, returns all rows"
    ] = None,
    search_columns: Annotated[
        Optional[List[str]],
        "Which text fields to search: any subset of ['ar_head','ar_desc']"
    ] = None,
    sections: Annotated[
        Optional[List[str]],
        "Filter by ar_section values, e.g. ['Taiwan News', 'World News', 'Sports', 'Front Page', 'Features', 'Editorials', 'Business','Bilingual Pages']"
    ] = None,
    date_from: Annotated[
        Optional[str],
        "Start date inclusive, 'YYYY-MM-DD'"
    ] = None,
    date_to: Annotated[
        Optional[str],
        "End date inclusive, 'YYYY-MM-DD'"
    ] = None
) -> List[Dict[str, Any]]:
    """
    Tool wrapper: takes a list-of-dicts, runs search_news, returns list-of-dicts.
    """
    df = cached_fetch_news()

    # Apply search
    result_df = search_news(
        df=df,
        query=query,
        search_columns=search_columns,
        sections=sections,
        date_from=date_from,
        date_to=date_to
    )
    # Return as plain JSON-serializable list
    return result_df.to_dict(orient="records")

def get_time() -> str:
        """
        Get the current time formatted as a string.

        Args:
            currentdatetime (Annotated[str, "The current date and time we are in"]): The current datetime string.

        Returns:
            str: A formatted string with the current time.
        """
        try:
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception as e:
            current_time = "2024-10-01 12:00:00"

        return f"Current time in your location: {current_time}"
