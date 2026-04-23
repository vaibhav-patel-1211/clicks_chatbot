"""
context_formatter.py – Formats a property dict into a string for the LLM prompt.

Django note
-----------
This file REPLACES property_data.py in the Django package.

property_data.py (the original) had two responsibilities:
  1. Load and parse property.txt files from disk  ← terminal demo only, NOT needed in Django
  2. format_property_context(dict) → str           ← still needed in Django

This file contains ONLY the formatting function.
Property data in Django always comes from the DB via consumers.py → _property_to_dict().
"""

from __future__ import annotations

from typing import Dict, List


# Keys excluded from the LLM context to save tokens
_SKIP_IN_CONTEXT = {
    "property_map_embed",
    "office_map_embed",
    "map_link",
    "office_map_link",
    "similar_map_link",
}


def format_property_context(p: Dict) -> str:
    """
    Convert a property dict into a human-readable string for the LLM system prompt.

    Parameters
    ----------
    p : dict — property data dict as built by consumers._property_to_dict()

    Returns
    -------
    str — formatted text injected into the LLM system prompt
    """
    lines: List[str] = []

    for key, value in p.items():
        if key in _SKIP_IN_CONTEXT or key == "highlights":
            continue
        display_key = key.replace("_", " ").title()
        lines.append(f"{display_key}: {value}")

    if p.get("highlights"):
        lines.append("")
        lines.append("Key Highlights:")
        for h in p["highlights"]:
            lines.append(f"  • {h}")

    return "\n".join(lines)
