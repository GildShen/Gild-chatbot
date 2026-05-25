"""Event config passed to the frontend.

For more details refer to https://js.cytoscape.org/#events
"""

from __future__ import annotations

RESERVED_NAMES = ["remove", "expand"]


class Event:
    def __init__(
        self,
        name: str,
        event_type: str,
        selector: str,
        debounce: int = 100,
    ) -> None:
        """Define an event to listen to on the Cytoscape instance.

        Parameters
        ----------
        name:
            User-defined name returned in the event payload under `action`.
        event_type:
            Space-separated list of Cytoscape event types (e.g. "click tap").
        selector:
            Cytoscape selector (e.g. "node", "edge", "node[label='PERSON']").
        debounce:
            Debounce time in milliseconds applied in the frontend.
        """
        self.name = name
        self.event_type = event_type
        self.selector = selector
        self.debounce = debounce

        if name in RESERVED_NAMES:
            raise ValueError(f"{RESERVED_NAMES} are reserved action names")

    def dump(self) -> dict:
        return {
            "name": self.name,
            "event_type": self.event_type,
            "selector": self.selector,
            "debounce": self.debounce,
        }
