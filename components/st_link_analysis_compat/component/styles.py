from __future__ import annotations

import warnings
from typing import Optional


def _hex_to_rgb(color: str) -> Optional[tuple[int, int, int]]:
    if not color:
        return None
    s = color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _lighten_hex(color: str, amount: float = 0.86) -> Optional[str]:
    """Blend a hex color toward white by `amount` (0..1).

    amount=0 -> original color, amount=1 -> white.
    """
    rgb = _hex_to_rgb(color)
    if rgb is None:
        return None
    amount = max(0.0, min(1.0, float(amount)))
    r, g, b = rgb
    nr = r + (255 - r) * amount
    ng = g + (255 - g) * amount
    nb = b + (255 - b) * amount
    return _rgb_to_hex((int(round(nr)), int(round(ng)), int(round(nb))))


class LinkAnalysisDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter("once", LinkAnalysisDeprecationWarning)


class NodeStyle:
    def __init__(
        self,
        label: str,
        color: Optional[str] = None,
        fill: Optional[str] = None,
        caption: Optional[str] = None,
        icon: Optional[str] = None,
        image_field: Optional[str] = None,
        image_fit: Optional[str] = None,
        image_clip: Optional[str] = None,
        image_opacity: Optional[float] = None,
        size: Optional[float] = None,
        size_field: Optional[str] = None,
        border: Optional[float] = None,
        shape: Optional[str] = None,
        corner_radius: Optional[float] = None,
        text_wrap: Optional[str] = None,
        text_max_width: Optional[float] = None,
        text_halign: Optional[str] = None,
        text_valign: Optional[str] = None,
        text_margin_y: Optional[float] = None,
        font_size: Optional[float] = None,
    ) -> None:
        self.label = label
        self.color = color
        self.fill = fill
        self.caption = caption
        self.icon = icon
        self.image_field = image_field
        self.image_fit = image_fit
        self.image_clip = image_clip
        self.image_opacity = image_opacity
        self.size = size
        self.size_field = size_field
        self.border = border
        self.shape = shape
        self.corner_radius = corner_radius
        self.text_wrap = text_wrap
        self.text_max_width = text_max_width
        self.text_halign = text_halign
        self.text_valign = text_valign
        self.text_margin_y = text_margin_y
        self.font_size = font_size

    def dump(self) -> dict:
        selector = f"node[label='{self.label}']"
        style: dict[str, object] = {}

        if self.shape:
            style["shape"] = self.shape
        if self.corner_radius is not None:
            style["corner-radius"] = float(self.corner_radius)
        if self.text_wrap:
            style["text-wrap"] = self.text_wrap
        if self.text_max_width is not None:
            style["text-max-width"] = float(self.text_max_width)
        if self.text_halign:
            style["text-halign"] = self.text_halign
        if self.text_valign:
            style["text-valign"] = self.text_valign
        if self.text_margin_y is not None:
            style["text-margin-y"] = float(self.text_margin_y)
        if self.font_size is not None:
            style["font-size"] = float(self.font_size)
        # Backward-compatible aesthetic default:
        # If only `color` is provided (historically used as border color),
        # auto-generate a soft background fill derived from it.
        if self.fill:
            style["background-color"] = self.fill
        elif self.color:
            auto_fill = _lighten_hex(self.color, amount=0.86)
            if auto_fill:
                style["background-color"] = auto_fill
        if self.size_field:
            style["width"] = f"data({self.size_field})"
            style["height"] = f"data({self.size_field})"
        elif self.size is not None:
            style["width"] = float(self.size)
            style["height"] = float(self.size)
        if self.border is not None:
            style["border-width"] = float(self.border)
        if self.color:
            style["border-color"] = self.color
        if self.caption:
            style["label"] = f"data({self.caption})"

        # Per-node images (preferred): background-image can be driven by node data.
        # Example: NodeStyle("FRIEND", image_field="image_url", image_fit="cover")
        if self.image_field:
            style["background-image"] = f"data({self.image_field})"
            if self.image_fit:
                style["background-fit"] = self.image_fit
            if self.image_clip:
                style["background-clip"] = self.image_clip
            if self.image_opacity is not None:
                style["background-image-opacity"] = float(self.image_opacity)
        elif self.icon:
            # Static icon (same for all nodes of this label)
            if not self.icon.startswith("url"):
                icon_path = f"./icons/{self.icon.lower()}.svg"
                style["background-image"] = icon_path
            else:
                style["background-image"] = self.icon

        return {"selector": selector, "style": style}


class EdgeStyle:
    def __init__(
        self,
        label: str,
        color: Optional[str] = None,
        caption: Optional[str] = None,
        labeled: Optional[bool] = None,  # deprecated
        directed: bool = False,
        curve_style: Optional[str] = None,
        weight: Optional[float] = None,
        weight_field: Optional[str] = None,
        stacktype: Optional[str] = None,
    ) -> None:
        self.label = label
        self.color = color
        self.caption = caption
        self.directed = directed
        self.curve_style = curve_style
        self.weight = weight
        self.weight_field = weight_field
        self.stacktype = stacktype

        if labeled is not None:
            warnings.warn(
                "Parameter `labeled` is deprecated and will be removed in a future release. "
                "Please use the `caption` parameter instead.",
                LinkAnalysisDeprecationWarning,
            )
        if labeled and not caption:
            self.caption = "label"

    def dump(self) -> dict:
        selector = f"edge[label='{self.label}']"
        style: dict[str, object] = {}

        if self.color:
            style["line-color"] = self.color
            style["background-color"] = self.color
            style["text-background-color"] = self.color
            style["target-arrow-color"] = self.color
        if self.caption:
            style["label"] = f"data({self.caption})"
        if self.weight_field:
            style["width"] = f"data({self.weight_field})"
        elif self.weight is not None:
            style["width"] = float(self.weight)
        if self.directed:
            style["target-arrow-shape"] = "triangle"
        if self.stacktype:
            # Edge stacking (parallel/multi-edge readability)
            # - "stack" -> bezier with a reasonable step size
            # - otherwise, treat as Cytoscape `curve-style` value
            if self.stacktype == "stack":
                style["curve-style"] = "bezier"
                style["control-point-step-size"] = 40
            else:
                style["curve-style"] = self.stacktype
        elif self.curve_style:
            style["curve-style"] = self.curve_style

        return {"selector": selector, "style": style}
