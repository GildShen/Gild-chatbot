from __future__ import annotations

import os
import warnings
from typing import Callable, Literal, Optional, Union

import streamlit as st
import streamlit.components.v1 as components

from .events import Event
from .layouts import LAYOUTS
from .styles import EdgeStyle, NodeStyle


class LinkAnalysisDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter("once", LinkAnalysisDeprecationWarning)


parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "../frontend/build")

_component_func = components.declare_component(
    "st_link_analysis_compat",
    path=build_dir,
)

# @st.fragment
def st_link_analysis(
    elements: dict,
    layout: Union[str, dict] = "cose",
    node_styles: list[NodeStyle] = [],
    edge_styles: list[EdgeStyle] = [],
    height: int = 500,
    key: Optional[str] = None,
    on_change: Optional[Callable[..., None]] = None,
    node_actions: list[Literal["remove", "expand"]] = [],
    enable_node_actions: Optional[bool] = None,  # deprecated
    events: list[Event] = [],
    legend_items: Optional[list[dict]] = None,
    selection_info_config: Optional[dict] = None,
):
    """A compatible drop-in API for st_link_analysis.

    Notes
    -----
    - This component is intended to be API-compatible with `st_link_analysis`.
    - The return value is the latest event payload sent from the frontend:
      `{action: str, data: dict, timestamp: int}` or `None`.
    """

    node_styles_dumped = [n.dump() for n in node_styles]
    edge_styles_dumped = [e.dump() for e in edge_styles]
    style = node_styles_dumped + edge_styles_dumped

    if isinstance(layout, str):
        layout = LAYOUTS.get(layout, LAYOUTS["cose"])

    events_dumped = [e.dump() for e in events]

    # deprecated compatibility
    if enable_node_actions is not None:
        warnings.warn(
            "Paramter `enable_node_actions` is deprecated and will be removed in a future release. "
            "Please use the `node_actions` parameter instead.",
            LinkAnalysisDeprecationWarning,
        )
    if enable_node_actions and not node_actions:
        node_actions = ["remove", "expand"]

    return _component_func(
        elements=elements,
        style=style,
        layout=layout,
        height=f"{int(height)}px",
        key=key,
        on_change=on_change,
        nodeActions=node_actions,
        events=events_dumped,
        legendItems=legend_items,
        selectionInfoConfig=selection_info_config,
    )
