"""
graph_state.py – Typed state schema for the LangGraph conversation graph.

Django integration additions vs original terminal version
----------------------------------------------------------
- liked_properties   : all properties the buyer swiped right on
- active_property_id : which property the current conversation is about
- property_data      : always the active property dict, populated from DB
                       by consumers.py — never loaded from property.txt
"""

from __future__ import annotations

import operator
from typing import Dict, List, Optional
from typing_extensions import TypedDict, Annotated


class ConversationState(TypedDict, total=False):
    # ── Message history ───────────────────────────────────────────────────────
    # Plain dicts: {"role": "user"|"assistant"|"tool", "content": str}
    messages: Annotated[List[Dict[str, str]], operator.add]

    # ── Active property (loaded from DB by consumers.py) ──────────────────────
    property_data:    Dict   # full dict for the property this chatroom is about
    property_context: str    # formatted string injected into LLM system prompt

    # ── All liked / swiped-right properties ───────────────────────────────────
    # Populated once on WebSocket connect from the buyer's liked list in the DB.
    # Each entry: {"id": str, "title": str, "data": dict}
    liked_properties:   List[Dict]
    active_property_id: str

    # ── Per-turn I/O ──────────────────────────────────────────────────────────
    user_input:    str
    response:      str
    quick_replies: List[str]

    # ── Booking state (persisted across turns via Redis cache) ─────────────────
    pending_time:   Optional[str]
    pending_hour:   Optional[int]   # hour remembered when user gives time before date
    confirmed_time: Optional[str]
    visitor:        Dict
    booking_step:   Optional[str]   # None | "ask_time" | "ask_name" | "ask_phone"
                                    # | "confirm" | "cancel_confirm"
                                    # | "reschedule_old" | "reschedule_new"
                                    # | "switch_select"
    reschedule_old: Optional[str]

    # ── Error propagation ─────────────────────────────────────────────────────
    error: Optional[str]
