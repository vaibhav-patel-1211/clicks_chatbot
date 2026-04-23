"""
nodes.py  –  LangGraph nodes for the property chatbot.

Root cause of "generic response" bug
-------------------------------------
property_data was arriving as {} because consumers.py wasn't injecting it into
state before calling the graph.  nodes.py is now fully defensive:

  • _get_prop()    — safe accessor, never returns None
  • _ensure_ctx() — builds property_context from property_data if missing
  • Every LLM call validates property_data is non-empty before proceeding
  • Greeting is 100% deterministic (no LLM)
  • All booking steps are 100% deterministic (no LLM)
  • LLM is called ONLY for general Q&A, with full property context inline

NIM / Llama compatibility
--------------------------
NVIDIA NIM ignores SystemMessage.  The system prompt is prepended directly
into the first HumanMessage inside _call_llm().
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .appointment_manager import (
    book_appointment,
    cancel_appointment,
    check_availability,
    get_available_slots,
    list_upcoming,
    reschedule_appointment,
)
from .config import BUSINESS_HOURS_END, BUSINESS_HOURS_START, model as _llm_model
from .context_formatter import format_property_context
from .graph_state import ConversationState

logger = logging.getLogger(__name__)

# ── Quick-reply presets ───────────────────────────────────────────────────────
_DEFAULT_QR = ["Book a visit", "Property highlights", "Pricing & payment"]
_BOOKING_QR = ["Reschedule visit", "Cancel visit", "Get directions"]
_CANCEL_QR  = ["Book a visit", "Property highlights", "Contact agent"]
_GENERAL_QR = ["Book a visit", "Property highlights", "Pricing & payment"]


# ─────────────────────────────────────────────────────────────────────────────
# Safe state accessors
# ─────────────────────────────────────────────────────────────────────────────

def _get_prop(state: ConversationState) -> Dict:
    return state.get("property_data") or {}


def _ensure_ctx(state: ConversationState) -> str:
    ctx = state.get("property_context") or ""
    if not ctx:
        p = _get_prop(state)
        if p:
            ctx = format_property_context(p)
    return ctx


def _prop_title(state: ConversationState) -> str:
    return _get_prop(state).get("title") or "this property"


def _prop_id(state: ConversationState) -> str:
    return state.get("active_property_id") or ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM wrapper  (NIM-safe: system text prepended to first HumanMessage)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(messages: List[Dict]) -> str:
    from langchain_core.messages import AIMessage, HumanMessage

    system_parts: List[str] = []
    turns: List[Dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(str(m.get("content", "")))
        else:
            turns.append(m)

    sys_text = "\n\n".join(system_parts)
    lc: List = []
    injected = False

    for m in turns:
        content = str(m.get("content", ""))
        if m.get("role") == "assistant":
            lc.append(AIMessage(content=content))
        else:
            if sys_text and not injected:
                content  = f"{sys_text}\n\n---\n\n{content}"
                injected = True
            lc.append(HumanMessage(content=content))

    if not lc:
        lc.append(HumanMessage(content=sys_text or "Hello"))

    return _llm_model.invoke(lc).content


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_qr(text: str) -> Tuple[str, List[str]]:
    m = re.search(r"<quick_replies>(.*?)</quick_replies>", text, re.DOTALL)
    if not m:
        return text.strip(), []
    clean = text[: m.start()].strip()
    try:
        replies = json.loads(m.group(1).strip())
        if isinstance(replies, list):
            return clean, [r for r in replies if isinstance(r, str)][:4]
    except Exception:
        pass
    return clean, []


def _fmt_slot(slot: str) -> str:
    try:
        dt  = datetime.strptime(slot, "%Y-%m-%d %H:%M")
        h12 = dt.strftime("%I").lstrip("0") or "12"
        return dt.strftime(f"%A, %d %b at {h12}:%M %p")
    except Exception:
        return slot


def _slot_str(date_str: str, hour: int) -> str:
    return f"{date_str} {hour:02d}:00"


def _in_hours(hour: int) -> bool:
    return BUSINESS_HOURS_START <= hour < BUSINESS_HOURS_END


def _hours_label() -> str:
    end12 = BUSINESS_HOURS_END - 12 if BUSINESS_HOURS_END > 12 else BUSINESS_HOURS_END
    return f"{BUSINESS_HOURS_START} AM – {end12} PM"


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

_GREETINGS = {
    "hello", "hi", "hey", "hiya", "howdy", "yo", "sup", "greetings",
    "good morning", "good afternoon", "good evening", "good day",
    "hi there", "hello there",
}
_BOOK_KW = (
    "book a visit", "book visit", "book a slot", "book an appointment",
    "book appointment", "make an appointment", "schedule a visit",
    "schedule visit", "i want to visit", "i'd like to visit",
    "i would like to visit", "arrange a visit", "fix a visit",
    "set up a visit", "i want to book", "want to schedule",
    "book a meeting", "schedule a meeting", "book a tour","visit","book",
)
_CANCEL_KW = (
    "cancel my visit", "cancel my appointment", "cancel my booking",
    "cancel the appointment", "cancel the visit", "cancel booking",
    "cancel appointment", "remove my appointment", "delete my appointment",
    "cancel my slot", "cancel visit", "cancel", # quick-reply chip
)
_RESCHEDULE_KW = (
    "reschedule", "move my appointment", "change my appointment",
    "shift my appointment", "change the time", "change my visit",
    "move my visit", "postpone my appointment", "postpone my visit",
    "change my booking","change booking",
)
_LIST_KW = (
    "my bookings", "my appointments", "upcoming appointments",
    "list appointments", "show appointments", "show my bookings",
    "view my bookings", "my upcoming",
)
_SWITCH_KW = (
    "other property", "another property", "other listing",
    "different property", "show me more", "other properties",
    "alternatives", "show other", "see other",
)
_DISINTEREST_KW = (
    "not interested", "don't like this", "dont like this", "not for me",
    "too expensive", "out of budget", "too small", "too big",
    "not suitable", "move on", "next property", "skip this", "not my type",
)


def _detect_intent(text: str) -> str:
    clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if clean in _GREETINGS:
        return "greeting"
    t = text.lower()
    if any(k in t for k in _RESCHEDULE_KW): return "reschedule"
    if any(k in t for k in _CANCEL_KW):     return "cancel"
    if any(k in t for k in _BOOK_KW):       return "book"
    if any(k in t for k in _LIST_KW):       return "list"
    if any(k in t for k in _SWITCH_KW):     return "switch_property"
    if any(k in t for k in _DISINTEREST_KW):return "switch_property"
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Date/time parser
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday",
             "friday", "saturday", "sunday"]


def _parse_dt(text: str) -> Tuple[Optional[str], Optional[int]]:
    t   = text.lower().strip()
    now = datetime.now()
    date: Optional[datetime] = None

    if "today" in t:
        date = now
    elif "day after tomorrow" in t:
        date = now + timedelta(days=2)
    elif "tomorrow" in t:
        date = now + timedelta(days=1)
    else:
        for i, wd in enumerate(_WEEKDAYS):
            if wd in t:
                delta = (i - now.weekday()) % 7 or 7
                date  = now + timedelta(days=delta)
                break

    if date is None:
        m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
        if m:
            try: date = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError: pass

    if date is None:
        m = re.search(r"\b(\d{1,2})[\/\-](\d{1,2})\b", t)
        if m:
            try: date = datetime(now.year, int(m.group(2)), int(m.group(1)))
            except ValueError: pass

    if date is None:
        for mon, num in _MONTHS.items():
            m2 = re.search(rf"\b(\d{{1,2}})\s+{mon}\b|\b{mon}\s+(\d{{1,2}})\b", t)
            if m2:
                try: date = datetime(now.year, num, int(m2.group(1) or m2.group(2)))
                except ValueError: pass
                break

    hour: Optional[int] = None
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        hour = int(m.group(1))
    else:
        m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", t)
        if m:
            h = int(m.group(1))
            if m.group(2) == "pm" and h != 12: h += 12
            elif m.group(2) == "am" and h == 12: h = 0
            hour = h
        else:
            m = re.search(r"\bat\s+(\d{1,2})\b", t)
            if m: hour = int(m.group(1))

    return (date.strftime("%Y-%m-%d") if date else None), hour


# ─────────────────────────────────────────────────────────────────────────────
# Greeting  (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def _handle_greeting(state: ConversationState) -> Tuple[str, List[str], Dict]:
    p          = _get_prop(state)
    title      = p.get("title") or "this property"
    agent_name = p.get("agent_name") or "Priya"
    location   = p.get("location") or p.get("city") or ""
    loc_str    = f" in {location}" if location else ""

    msg = (
        f"Hi! I'm {agent_name}, your personal property advisor. 👋\n\n"
        f"You're looking at **{title}**{loc_str}.\n"
        f"I can help you with property details, pricing, amenities, "
        f"or booking a visit.\n\nWhat would you like to know?"
    )
    return msg, _DEFAULT_QR, {}


# ─────────────────────────────────────────────────────────────────────────────
# Booking flow  (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

_CHIP_SET = {
    "book a visit", "book visit", "proceed", "choose a different time",
    "property highlights", "pricing & payment", "property details",
    "amenities", "get directions", "contact agent", "view my bookings",
    "reschedule visit", "cancel visit", "yes, book",
}


def _handle_ask_time(user_text: str, state: ConversationState):
    if user_text.lower().strip() in _CHIP_SET:
        return (
            f"What date and time would you like to visit?\n"
            f"We're available Mon–Sat, {_hours_label()}.",
            ["Tomorrow 11 AM", "This Saturday 2 PM", "Next Monday 3 PM"],
            {"pending_time": None, "pending_hour": None},
        )

    date_str, hour = _parse_dt(user_text)
    pid = _prop_id(state)

    # If user gave only a date this turn, check if we have a pending_hour from last turn
    if date_str and hour is None:
        saved_hour = state.get("pending_hour")
        if saved_hour is not None:
            hour = saved_hour  # combine: we now have both date and time

    # ── Only date, no time ────────────────────────────────────────────────────
    if date_str and hour is None:
        slots = get_available_slots(date_str, 4, property_id=pid)
        if not slots:
            return (
                f"No slots on {date_str}. Try a different date?",
                ["Tomorrow", "This Saturday", "Next Monday"],
                {"pending_hour": None},
            )
        times = [s.split(" ")[1] for s in slots]
        return (
            f"Available times on {date_str}: {', '.join(times)}. Which works?",
            times[:4],
            {"pending_time": date_str, "pending_hour": None},
        )

    # ── Only time, no date ────────────────────────────────────────────────────
    if hour is not None and not date_str:
        if not _in_hours(hour):
            return (
                f"{hour:02d}:00 is outside visiting hours ({_hours_label()}, Mon–Sat).",
                ["11 AM", "2 PM", "5 PM"],
                {"pending_hour": None},
            )
        # If we already have a pending date from showing available slots, combine now
        pending = state.get("pending_time") or ""
        if pending and " " not in pending:  # pending_time is a date-only string
            date_str = pending
            # fall through to "both date and time" logic below
        else:
            return (
                f"Got it — {hour:02d}:00. Which date?",
                ["Today", "Tomorrow", "This Saturday"],
                {"pending_hour": hour},
            )

    # ── Neither parsed ────────────────────────────────────────────────────────
    if not date_str and hour is None:
        return (
            'Please share a date and time — e.g. "tomorrow at 2 PM" or "Saturday at 11 AM".',
            ["Tomorrow 11 AM", "This Saturday 2 PM", "Next Monday 3 PM"],
            {},
        )

    # ── Both date and time available ──────────────────────────────────────────
    if not _in_hours(hour):
        return (
            f"{hour:02d}:00 is outside visiting hours ({_hours_label()}, Mon–Sat).",
            ["11 AM", "2 PM", "5 PM"],
            {"pending_hour": None},
        )

    slot  = _slot_str(date_str, hour)
    avail = check_availability(slot, property_id=pid)
    if avail.get("available"):
        return (
            f"Great — {_fmt_slot(slot)} is available! "
            "May I have your full name for the booking?",
            [],
            {"pending_time": slot, "pending_hour": None, "booking_step": "ask_name"},
        )

    alts = get_available_slots(date_str, 4, property_id=pid)
    if alts:
        times = [s.split(" ")[1] for s in alts]
        return (
            f"Sorry, {_fmt_slot(slot)} is taken.\n"
            f"Available on {date_str}: {', '.join(times)}. Which works?",
            times[:4],
            {"pending_time": date_str, "pending_hour": None},
        )
    return (
        f"No slots on {date_str}. Try a different date?",
        ["Tomorrow", "This Saturday", "Next Monday"],
        {"pending_hour": None},
    )


_NAME_CHIPS = {
    "proceed", "yes", "no", "confirm", "cancel", "book a visit", "book visit",
    "reschedule visit", "cancel visit", "get directions", "property highlights",
    "pricing & payment", "contact agent", "today", "tomorrow", "this saturday",
}


def _handle_ask_name(user_text: str, state: ConversationState):
    name = user_text.strip().title()
    if len(name) < 2 or name.lower() in _NAME_CHIPS:
        return (
            "Please share your full name so I can complete the booking.",
            [],
            {},
        )
    visitor         = dict(state.get("visitor") or {})
    visitor["name"] = name
    return (
        f"Thanks, {name}! What's your phone number?",
        [],
        {"visitor": visitor, "booking_step": "ask_phone"},
    )

try:
    import phonenumbers
    _PHONENUMBERS_AVAILABLE = True
except ImportError:
    _PHONENUMBERS_AVAILABLE = False


def _validate_phone(raw: str) -> tuple[bool, str]:
    """
    Returns (is_valid, cleaned_phone).
    Uses is_possible_number() instead of is_valid_number() so that
    test/dummy numbers pass format checks without needing real carrier ranges.
    """
    phone  = re.sub(r"[^\d\+\-\s\(\)]", "", raw).strip()
    digits = re.sub(r"\D", "", phone)

    if not _PHONENUMBERS_AVAILABLE:
        return (7 <= len(digits) <= 15), phone

    def _try_parse(number: str, region: Optional[str]) -> tuple[bool, str]:
        try:
            parsed = phonenumbers.parse(number, region)
            # is_possible_number = correct length/format for region
            # is_valid_number    = real carrier range (too strict for test numbers)
            if phonenumbers.is_possible_number(parsed):
                cleaned = phonenumbers.format_number(
                    parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                )
                return True, cleaned
        except Exception:
            pass
        return False, phone

    # ── Explicit + prefix: parse directly ────────────────────────────────
    if phone.startswith("+"):
        return _try_parse(phone, None)

    # ── No country code: detect by digit length and prefix ───────────────
    if len(digits) == 8:
        # Could be Costa Rica (8 digits) — try CR first
        ok, cleaned = _try_parse(phone, "CR")
        if ok:
            return True, cleaned

    elif len(digits) == 10:
        # India bare (10 digits) — try IN first, then CR (unlikely but safe)
        for region in ["IN", "CR"]:
            ok, cleaned = _try_parse(phone, region)
            if ok:
                return True, cleaned

    elif len(digits) == 11:
        if digits.startswith("0"):
            return _try_parse(phone, "IN")       # 0XXXXXXXXXX India style
        elif digits.startswith("506"):
            return _try_parse("+" + digits, None) # 506XXXXXXXX → +506XXXXXXXX
        elif digits.startswith("91"):
            return _try_parse("+" + digits, None) # 91XXXXXXXXXX → +91XXXXXXXXXX

    elif len(digits) == 12:
        if digits.startswith("91"):
            return _try_parse("+" + digits, None) # 91XXXXXXXXXX
        elif digits.startswith("506"):
            return _try_parse("+" + digits, None) # 506XXXXXXXX

    # ── Last resort: try IN and CR as-is ────────────────────────────────
    for region in ["IN", "CR"]:
        ok, cleaned = _try_parse(phone, region)
        if ok:
            return True, cleaned

    return False, phone


def _handle_ask_phone(user_text: str, state: ConversationState):
    is_valid, phone = _validate_phone(user_text)

    if not is_valid:
        return (
            "Please share a valid phone number.\n"
            "• India: 10-digit number or with +91\n"
            "• Costa Rica: 8-digit number or with +506",
            [],
            {},
        )

    visitor          = dict(state.get("visitor") or {})
    visitor["phone"] = phone
    pending          = state.get("pending_time") or ""
    name             = visitor.get("name", "")
    title            = _get_prop(state).get("title") or "the property"
    fmt_date         = _fmt_slot(pending) if " " in (pending or "") else (pending or "TBD")

    return (
        f"Please confirm your booking:\n\n"
        f"  🏠 Property : {title}\n"
        f"  👤 Name     : {name}\n"
        f"  📞 Phone    : {phone}\n"
        f"  📅 Visit    : {fmt_date}\n\n"
        "Shall I confirm this?",
        ["Yes, confirm", "No, change time"],
        {"visitor": visitor, "booking_step": "confirm"},
    ) 
 


_YES = {"yes", "yep", "yeah", "yup", "sure", "ok", "okay", "confirm",
        "go ahead", "book it", "proceed", "correct", "right", "do it",
        "yes confirm", "yes, confirm"}


def _handle_confirm(user_text: str, state: ConversationState):
    if not any(w in user_text.lower() for w in _YES):
        return (
            "No problem! What date and time would you prefer?",
            ["Tomorrow 11 AM", "This Saturday 2 PM", "Next Monday 3 PM"],
            {"booking_step": "ask_time"},
        )
    visitor = state.get("visitor") or {}
    pending = state.get("pending_time") or ""
    name    = visitor.get("name", "")
    phone   = visitor.get("phone", "")
    pid     = _prop_id(state)
    prop    = _get_prop(state)
    title   = prop.get("title") or "Property"
    contact = prop.get("contact") or prop.get("phone") or ""

    # pending_time must be a full "YYYY-MM-DD HH:MM" string — if it's only a
    # date (no time) the booking would be saved incorrectly. Ask again.
    if not pending or " " not in pending:
        return (
            "I still need a time for your visit. What time works?\n"
            f"Available Mon–Sat, {_hours_label()}.",
            ["11 AM", "2 PM", "4 PM"],
            {"booking_step": "ask_time", "pending_time": pending or None},
        )

    result = book_appointment(
        summary     = f"Property Visit – {name} | {title}",
        start_time  = pending,
        description = f"Property: {title} | Visitor: {name} | Phone: {phone}",
        property_id = pid,
    )
    logger.info("book_appointment(%s, pid=%s) → %s", pending, pid, result)
    if result["success"]:
        a    = result["appointment"]
        note = f"\n\nFor queries call: {contact}" if contact else ""
        return (
            f"✅ Booking confirmed!\n\n"
            f"  🏠 Property : {title}\n"
            f"  👤 Name     : {name}\n"
            f"  📞 Phone    : {phone}\n"
            f"  📅 Visit    : {_fmt_slot(a['start_time'])}\n"
            f"  ⏱  Duration : {a['duration_minutes']} min{note}",
            _BOOKING_QR,
            {"booking_step": "", "confirmed_time": pending, "visitor": {}, "pending_hour": None},
        )
    return (
        f"Booking failed: {result['message']}. Try a different time?",
        ["Try a different time", "Contact agent"],
        {"booking_step": "ask_time"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cancel flow
# ─────────────────────────────────────────────────────────────────────────────

_AFFIRM = {"yes", "y", "yep", "yeah", "yup", "sure", "ok", "okay",
           "confirm", "cancel it", "go ahead", "proceed"}
_DENY   = {"no", "nope", "nah", "keep", "n", "keep it"}


def _my_appts(state: ConversationState) -> List[Dict]:
    """
    Return only the appointments that belong to the current session's visitor.
    Matches on visitor name or phone stored in state["visitor"] or
    confirmed_time. Falls back to all appointments if no visitor info present.
    The appointment summary format is:  "Property Visit – {name} | {title}"
    """
    pid     = _prop_id(state)
    all_    = list_upcoming(30, property_id=pid).get("appointments", [])
    visitor = state.get("visitor") or {}
    name    = (visitor.get("name") or "").strip().lower()
    phone   = re.sub(r"\D", "", visitor.get("phone") or "")

    if not name and not phone:
        return all_  # no visitor info yet — edge case fallback

    filtered = []
    for a in all_:
        summary = (a.get("summary") or "").lower()
        desc    = (a.get("description") or "").lower()
        name_match  = bool(name  and (f"– {name} |" in summary or f"– {name}|" in summary))
        phone_match = bool(phone and phone in re.sub(r"\D", "", desc))
        if name_match or phone_match:
            filtered.append(a)

    return filtered if filtered else all_


def _match_appt(user_text: str, appts: List[Dict]) -> Optional[str]:
    t = user_text.strip()
    if t.isdigit():
        idx = int(t) - 1
        if 0 <= idx < len(appts):
            return appts[idx]["start_time"]
    date_str, hour = _parse_dt(user_text)
    if date_str and hour is not None:
        return _slot_str(date_str, hour)
    if date_str:
        for a in appts:
            if a["start_time"].startswith(date_str):
                return a["start_time"]
    tl = user_text.lower()
    for a in appts:
        if tl in a["start_time"].lower() or tl in a.get("summary", "").lower():
            return a["start_time"]
    return None


def _handle_cancel_start(user_text: str, state: ConversationState):
    appts = _my_appts(state)
    if not appts:
        return ("You have no upcoming appointments to cancel.", _GENERAL_QR, {"booking_step": ""})
    if len(appts) == 1:
        a = appts[0]
        return (
            f"Found your appointment:\n\n"
            f"  📅 {_fmt_slot(a['start_time'])}\n"
            f"  🏠 {a['summary']}\n\n"
            "Would you like to cancel it?",
            ["Yes, cancel it", "No, keep it"],
            {"booking_step": "cancel_confirm"},
        )
    lines = ["Which appointment would you like to cancel?\n"]
    for i, a in enumerate(appts[:5], 1):
        lines.append(f"  {i}. {_fmt_slot(a['start_time'])}  –  {a['summary']}")
    lines.append("\nReply with the number.")
    return (
        "\n".join(lines),
        [str(i) for i in range(1, min(5, len(appts)) + 1)],
        {"booking_step": "cancel_confirm"},
    )


def _handle_cancel_confirm(user_text: str, state: ConversationState):
    pid   = _prop_id(state)
    appts = _my_appts(state)
    t     = user_text.strip().lower()
    if not appts:
        return ("No appointments found.", _GENERAL_QR, {"booking_step": ""})

    slot: Optional[str] = None
    if len(appts) == 1:
        if t in _AFFIRM or any(w in t for w in ("yes", "cancel", "confirm")):
            slot = appts[0]["start_time"]
        elif t in _DENY or any(w in t for w in ("no", "keep")):
            return ("Your appointment is kept as is.", _GENERAL_QR, {"booking_step": ""})

    if slot is None:
        slot = _match_appt(user_text, appts)

    if not slot:
        return (
            "I couldn't match that. Reply with the number or date/time.",
            [str(i) for i in range(1, min(5, len(appts)) + 1)],
            {},
        )
    res = cancel_appointment(slot, property_id=pid)
    if res["success"]:
        return (
            f"✅ Appointment on {_fmt_slot(slot)} cancelled.",
            _CANCEL_QR,
            {"booking_step": "", "pending_time": None, "confirmed_time": None},
        )
    return (f"Couldn't cancel: {res['message']}", _GENERAL_QR, {"booking_step": ""})


# ─────────────────────────────────────────────────────────────────────────────
# Reschedule flow
# ─────────────────────────────────────────────────────────────────────────────

def _handle_reschedule_start(user_text: str, state: ConversationState):
    appts = _my_appts(state)
    if not appts:
        return ("You have no upcoming appointments to reschedule.", _GENERAL_QR, {"booking_step": ""})
    if len(appts) == 1:
        slot = appts[0]["start_time"]
        return (
            f"Your appointment is on {_fmt_slot(slot)}. What new time would you prefer?",
            ["Tomorrow 11 AM", "This Saturday 2 PM", "Next Monday 3 PM"],
            {"booking_step": "reschedule_new", "reschedule_old": slot},
        )
    lines = ["Which appointment would you like to reschedule?\n"]
    for i, a in enumerate(appts[:5], 1):
        lines.append(f"  {i}. {_fmt_slot(a['start_time'])}  –  {a['summary']}")
    lines.append("\nReply with the number.")
    return (
        "\n".join(lines),
        [str(i) for i in range(1, min(5, len(appts)) + 1)],
        {"booking_step": "reschedule_old"},
    )


def _handle_reschedule_old(user_text: str, state: ConversationState):
    appts = _my_appts(state)
    slot  = _match_appt(user_text, appts)
    if not slot:
        return (
            "Couldn't match that. Reply with the number or date/time.",
            [str(i) for i in range(1, min(5, len(appts)) + 1)],
            {},
        )
    return (
        f"Rescheduling {_fmt_slot(slot)}. What new date and time?",
        ["Tomorrow 11 AM", "This Saturday 2 PM", "Next Monday 3 PM"],
        {"booking_step": "reschedule_new", "reschedule_old": slot},
    )


def _handle_reschedule_new(user_text: str, state: ConversationState):
    old_slot = state.get("reschedule_old") or ""
    pid      = _prop_id(state)
    date_str, hour = _parse_dt(user_text)

    if hour is not None and not date_str and old_slot:
        date_str = old_slot[:10]

    if not date_str or hour is None:
        return (
            'Please give a date and time, e.g. "March 5 at 2 PM".',
            ["Tomorrow 11 AM", "This Saturday 2 PM"],
            {},
        )
    if not _in_hours(hour):
        return (
            f"{hour:02d}:00 is outside visiting hours ({_hours_label()}, Mon–Sat).",
            ["11 AM", "2 PM", "5 PM"],
            {},
        )

    new_slot = _slot_str(date_str, hour)
    if not check_availability(new_slot, property_id=pid).get("available"):
        alts = get_available_slots(date_str, 4, property_id=pid)
        if alts:
            times = [s.split(" ")[1] for s in alts]
            return (
                f"{_fmt_slot(new_slot)} is not available.\n"
                f"Available on {date_str}: {', '.join(times)}.",
                times[:4],
                {},
            )
        return (f"No slots on {date_str}. Try a different date?", ["Tomorrow", "This Saturday"], {})

    res = reschedule_appointment(old_slot, new_slot, property_id=pid)
    if res["success"]:
        a = res["appointment"]
        return (
            f"✅ Rescheduled!\n\n"
            f"  From : {_fmt_slot(old_slot)}\n"
            f"  To   : {_fmt_slot(a['start_time'])}",
            _BOOKING_QR,
            {"booking_step": "", "reschedule_old": None, "confirmed_time": new_slot},
        )
    return (f"Reschedule failed: {res['message']}", _GENERAL_QR, {"booking_step": ""})


# ─────────────────────────────────────────────────────────────────────────────
# Switch property
# ─────────────────────────────────────────────────────────────────────────────

def _handle_switch_property(state: ConversationState) -> Tuple[str, List[str], Dict]:
    liked  = state.get("liked_properties") or []
    active = state.get("active_property_id") or ""
    others = [p for p in liked if str(p.get("id", "")) != str(active)]
    if not others:
        return (
            "This is the only property you've saved. Explore more in the app!",
            _GENERAL_QR,
            {},
        )

    # ── Only one other property — switch directly, no list needed ─────────────
    if len(others) == 1:
        chosen   = others[0]
        new_data = chosen.get("data") or {}
        title    = chosen.get("title") or "another property"
        location = new_data.get("location") or ""
        price    = new_data.get("price") or 0
        price_str = f"₹{price:,}" if isinstance(price, (int, float)) and price > 0 else "available on request"
        loc_str  = f" in {location}" if location else ""
        return (
            f"Sure! You also saved **{title}**{loc_str}. "
            f"Price is {price_str}. Would you like to know more about it or book a visit?",
            _DEFAULT_QR,
            {
                "active_property_id": str(chosen["id"]),
                "property_data":      new_data,
                "property_context":   format_property_context(new_data),
                "booking_step":       "",
                "pending_time":       None,
                "pending_hour":       None,
                "confirmed_time":     None,
                "visitor":            {},
            },
        )

    # ── Multiple others — show numbered list ──────────────────────────────────
    lines = ["Here are your other saved properties:\n"]
    for i, p in enumerate(others[:5], 1):
        ttl   = p.get("title") or f"Property {p.get('id', i)}"
        price = (p.get("data") or {}).get("price")
        p_str = (f"  –  ₹{price:,}" if isinstance(price, (int, float)) and price > 0 else "")
        lines.append(f"  {i}. {ttl}{p_str}")
    lines.append("\nReply with the number to switch.")
    qr = [f"{i}. {p.get('title', f'Option {i}')}"[:35] for i, p in enumerate(others[:3], 1)]
    return ("\n".join(lines), qr, {"booking_step": "switch_select"})


def _handle_switch_select(user_text: str, state: ConversationState) -> Tuple[str, List[str], Dict]:
    liked  = state.get("liked_properties") or []
    active = state.get("active_property_id") or ""
    others = [p for p in liked if str(p.get("id", "")) != str(active)]
    t      = user_text.strip()
    chosen: Optional[Dict] = None

    # Handle "1. Title" formatted quick replies — extract leading number
    num_match = re.match(r"^(\d+)[.\s]", t)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(others):
            chosen = others[idx]

    if chosen is None and t.isdigit():
        idx = int(t) - 1
        if 0 <= idx < len(others):
            chosen = others[idx]

    if chosen is None:
        tl = t.lower()
        # First try: text is contained in a title
        for p in others:
            if tl in (p.get("title") or "").lower():
                chosen = p
                break
        # Second try (for truncated quick replies): title starts with or contains user text
        if chosen is None:
            for p in others:
                prop_title_lower = (p.get("title") or "").lower()
                if prop_title_lower.startswith(tl[:15]):  # match first 15 chars
                    chosen = p
                    break

    if not chosen:
        return (
            "Couldn't match that. Reply with the number from the list.",
            [str(i) for i in range(1, min(4, len(others)) + 1)],
            {},
        )
    new_data = chosen.get("data") or {}
    title    = chosen.get("title") or "the property"
    location = new_data.get("location") or ""
    price    = new_data.get("price") or 0
    price_str = f"₹{price:,}" if isinstance(price, (int, float)) and price > 0 else "available on request"
    loc_str  = f" in {location}" if location else ""
    return (
        f"Sure! Here's **{title}**{loc_str}. "
        f"Price is {price_str}. Would you like to know more about it or book a visit?",
        _DEFAULT_QR,
        {
            "active_property_id": str(chosen["id"]),
            "property_data":      new_data,
            "property_context":   format_property_context(new_data),
            "booking_step":       "",
            "pending_time":       None,
            "pending_hour":       None,
            "confirmed_time":     None,
            "visitor":            {},
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Q&A
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(state: ConversationState) -> str:
    prop       = _get_prop(state)
    agent_name = prop.get("agent_name") or "Priya"
    ctx        = _ensure_ctx(state)
    now        = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

    return (
        f"You are {agent_name}, a friendly and knowledgeable real-estate assistant.\n"
        f"Today is {now}.\n\n"
        "=== THE PROPERTY YOU ARE HELPING WITH ===\n"
        f"{ctx}\n\n"
        "=== YOUR RULES ===\n"
        "1. Answer ONLY the question asked. Be concise: 2-4 sentences maximum.\n"
        "2. ONLY discuss the property above. NEVER mention any other properties.\n"
        "3. Do NOT greet or introduce yourself — answer the question directly.\n"
        "4. Do NOT use bullet points unless the user explicitly asks for a list.\n"
        "5. Do NOT invent data. Only use information from THE PROPERTY section above.\n"
        "6. If price is 0 or missing, say: 'Price is available on request.'\n"
        "7. If you don't have the info, say: 'I don't have that detail — please contact the agent.'\n\n"
        "8. If you can't understand user language, say :'''I'm sorry, I don't understand'''.\n"
        "9. Never say please contact agent"
        "=== QUICK REPLIES ===\n"
        "End EVERY reply with exactly this tag containing 3 relevant options:\n"
        "<quick_replies>[\"Option A\", \"Option B\", \"Option C\"]</quick_replies>\n"
        "Examples:\n"
        "  After price    → [\"Book a visit\", \"Payment plan\", \"Amenities\"]\n"
        "  After location → [\"Book a visit\", \"Contact agent\", \"Property highlights\"]\n"
        "  After amenities → [\"Book a visit\", \"Pricing & payment\", \"Get directions\"]\n"
        "  Default        → [\"Book a visit\", \"Property highlights\", \"Pricing & payment\"]\n"
    )


def _llm_qa(user_text: str, state: ConversationState) -> Tuple[str, List[str]]:
    if not _get_prop(state):
        return (
            "I'm having trouble loading the property details. Please try again.",
            _DEFAULT_QR,
        )

    messages_raw = list(state.get("messages") or [])
    history: List[Dict] = [{"role": "system", "content": _build_system_prompt(state)}]

    prior = [m for m in messages_raw if m.get("role") in ("user", "assistant")]
    if prior and prior[-1].get("role") == "user":
        prior = prior[:-1]
    for m in prior[-8:]:
        history.append({"role": m["role"], "content": str(m.get("content", ""))})
    history.append({"role": "user", "content": user_text})

    try:
        raw       = _call_llm(history)
        clean, qr = _strip_qr(raw)
        return clean or "I don't have that detail — please contact the agent.", (qr or _DEFAULT_QR)
    except Exception:
        return ("Sorry, I couldn't get that right now. Please try again.", _DEFAULT_QR)


# ─────────────────────────────────────────────────────────────────────────────
# agent_node
# ─────────────────────────────────────────────────────────────────────────────

def agent_node(state: ConversationState) -> Dict:
    messages  = list(state.get("messages") or [])
    user_text = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    ).strip()

    step          = state.get("booking_step") or ""
    response      = ""
    quick_replies = _DEFAULT_QR
    state_updates: Dict = {}

    if   step == "ask_time":       response, quick_replies, state_updates = _handle_ask_time(user_text, state)
    elif step == "ask_name":       response, quick_replies, state_updates = _handle_ask_name(user_text, state)
    elif step == "ask_phone":      response, quick_replies, state_updates = _handle_ask_phone(user_text, state)
    elif step == "confirm":        response, quick_replies, state_updates = _handle_confirm(user_text, state)
    elif step == "cancel_confirm": response, quick_replies, state_updates = _handle_cancel_confirm(user_text, state)
    elif step == "reschedule_old": response, quick_replies, state_updates = _handle_reschedule_old(user_text, state)
    elif step == "reschedule_new": response, quick_replies, state_updates = _handle_reschedule_new(user_text, state)
    elif step == "switch_select":  response, quick_replies, state_updates = _handle_switch_select(user_text, state)
    else:
        intent = _detect_intent(user_text)

        if intent == "greeting":
            response, quick_replies, state_updates = _handle_greeting(state)

        elif intent == "book":
            response = (
                f"I'd love to help you book a visit to **{_prop_title(state)}**! 🏠\n"
                f"What date and time works? We're available Mon–Sat, {_hours_label()}."
            )
            quick_replies = ["Today", "Tomorrow", "This Saturday"]
            state_updates = {"booking_step": "ask_time"}

        elif intent == "cancel":
            response, quick_replies, state_updates = _handle_cancel_start(user_text, state)

        elif intent == "reschedule":
            response, quick_replies, state_updates = _handle_reschedule_start(user_text, state)

        elif intent == "list":
            appts = _my_appts(state)
            if not appts:
                response = "You have no upcoming appointments for this property."
            else:
                lines = [f"Your upcoming appointments ({len(appts)}):\n"]
                for a in appts:
                    lines.append(f"  • {_fmt_slot(a['start_time'])}  |  {a['summary']}")
                response = "\n".join(lines)
            quick_replies = _BOOKING_QR

        elif intent == "switch_property":
            response, quick_replies, state_updates = _handle_switch_property(state)

        else:
            response, quick_replies = _llm_qa(user_text, state)

    result = {
        "messages":      [{"role": "assistant", "content": response}],
        "response":      response,
        "quick_replies": quick_replies,
    }
    result.update(state_updates)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# output_node
# ─────────────────────────────────────────────────────────────────────────────

def output_node(state: ConversationState) -> Dict:
    response      = state.get("response") or ""
    quick_replies = state.get("quick_replies") or []
    clean, qr_txt = _strip_qr(response)
    return {
        "response":      clean or "How can I help you?",
        "quick_replies": quick_replies if quick_replies else (qr_txt or _DEFAULT_QR),
    }
