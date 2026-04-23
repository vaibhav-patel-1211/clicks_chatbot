"""
appointment_manager.py – Appointment storage and retrieval (JSON-backed).

Django integration change vs original terminal version
-------------------------------------------------------
All public functions accept an optional `property_id` keyword argument.
When provided, the JSON file is named appointments_<property_id>.json so
bookings are scoped per property listing and never collide across properties.

Default behaviour (no property_id) is 100% backward-compatible.

Production swap: replace _load() and _save() with Django ORM or Google
Calendar API calls. All public function signatures stay the same.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .config import (
    CALENDAR_FILE,
    APPOINTMENT_DURATION_MINUTES,
    BUSINESS_HOURS_START,
    BUSINESS_HOURS_END,
    MAX_SLOTS_PER_DAY,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _calendar_file(property_id: Optional[str] = None) -> str:
    if not property_id:
        return CALENDAR_FILE
    base, ext = os.path.splitext(CALENDAR_FILE)
    return f"{base}_{property_id}{ext}"


def _load(property_id: Optional[str] = None) -> Dict:
    path = _calendar_file(property_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data: Dict, property_id: Optional[str] = None) -> None:
    path = _calendar_file(property_id)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[appointment_manager] ERROR saving calendar: {exc}")


def _parse_dt(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised datetime format: '{value}'")


def _overlaps(new_start: datetime, new_dur: int, existing: Dict[str, dict]) -> bool:
    new_end = new_start + timedelta(minutes=new_dur)
    for appt in existing.values():
        a_start = _parse_dt(appt["start_time"])
        a_end   = a_start + timedelta(minutes=appt["duration_minutes"])
        if new_start < a_end and new_end > a_start:
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def check_availability(slot: str, property_id: Optional[str] = None) -> Dict:
    data  = _load(property_id)
    is_dt = " " in slot
    try:
        dt = _parse_dt(slot)
    except ValueError as exc:
        return {"available": False, "message": str(exc)}

    date_key = dt.strftime("%Y-%m-%d")
    day_data = data.get(date_key, {})

    if is_dt:
        time_key = dt.strftime("%H:%M")
        if not (BUSINESS_HOURS_START <= dt.hour < BUSINESS_HOURS_END):
            return {
                "available": False,
                "message": (
                    f"Outside business hours. Visits available "
                    f"{BUSINESS_HOURS_START:02d}:00–{BUSINESS_HOURS_END:02d}:00."
                ),
            }
        if time_key in day_data:
            return {"available": False, "message": f"{time_key} on {date_key} is already booked."}
        return {"available": True, "message": f"{time_key} on {date_key} is available."}

    booked = list(day_data.keys())
    return {
        "available":    len(booked) < MAX_SLOTS_PER_DAY,
        "message":      f"{date_key} has {len(booked)} appointment(s) booked.",
        "booked_slots": booked,
    }


def book_appointment(
    summary: str,
    start_time: str,
    duration_minutes: int = APPOINTMENT_DURATION_MINUTES,
    description: str = "",
    property_id: Optional[str] = None,
) -> Dict:
    try:
        start_dt = _parse_dt(start_time)
    except ValueError as exc:
        return {"success": False, "message": str(exc), "appointment": None}

    date_key = start_dt.strftime("%Y-%m-%d")
    time_key = start_dt.strftime("%H:%M")
    data     = _load(property_id)
    day_data = data.get(date_key, {})

    if _overlaps(start_dt, duration_minutes, day_data):
        return {
            "success":     False,
            "message":     f"Slot {time_key} on {date_key} conflicts with an existing booking.",
            "appointment": None,
        }

    end_dt = start_dt + timedelta(minutes=duration_minutes)
    appt = {
        "summary":          summary,
        "start_time":       start_time,
        "duration_minutes": duration_minutes,
        "description":      description,
        "end_time":         end_dt.strftime("%Y-%m-%d %H:%M"),
        "property_id":      property_id or "",
    }

    data.setdefault(date_key, {})[time_key] = appt
    _save(data, property_id)
    return {"success": True, "message": f"Appointment confirmed for {start_time}.", "appointment": appt}


def cancel_appointment(start_time: str, property_id: Optional[str] = None) -> Dict:
    try:
        dt = _parse_dt(start_time)
    except ValueError as exc:
        return {"success": False, "message": str(exc)}

    date_key = dt.strftime("%Y-%m-%d")
    time_key = dt.strftime("%H:%M")
    data     = _load(property_id)

    if date_key not in data or time_key not in data[date_key]:
        return {"success": False, "message": f"No appointment found at {start_time}."}

    del data[date_key][time_key]
    if not data[date_key]:
        del data[date_key]
    _save(data, property_id)
    return {"success": True, "message": f"Appointment at {start_time} cancelled successfully."}


def reschedule_appointment(
    old_start_time: str,
    new_start_time: str,
    duration_minutes: int = APPOINTMENT_DURATION_MINUTES,
    property_id: Optional[str] = None,
) -> Dict:
    data = _load(property_id)

    try:
        old_dt = _parse_dt(old_start_time)
    except ValueError as exc:
        return {"success": False, "message": str(exc), "appointment": None}

    old_date = old_dt.strftime("%Y-%m-%d")
    old_time = old_dt.strftime("%H:%M")

    if old_date not in data or old_time not in data[old_date]:
        return {"success": False, "message": f"No appointment found at {old_start_time}.", "appointment": None}

    old_appt = data[old_date][old_time]

    try:
        new_dt = _parse_dt(new_start_time)
    except ValueError as exc:
        return {"success": False, "message": str(exc), "appointment": None}

    if not (BUSINESS_HOURS_START <= new_dt.hour < BUSINESS_HOURS_END):
        return {
            "success":     False,
            "message":     f"New time outside business hours ({BUSINESS_HOURS_START:02d}:00–{BUSINESS_HOURS_END:02d}:00).",
            "appointment": None,
        }

    new_date = new_dt.strftime("%Y-%m-%d")
    new_time = new_dt.strftime("%H:%M")

    temp_day = {
        k: v for k, v in data.get(new_date, {}).items()
        if not (new_date == old_date and k == old_time)
    }

    if _overlaps(new_dt, duration_minutes, temp_day):
        return {"success": False, "message": f"New slot {new_time} on {new_date} is already taken.", "appointment": None}

    del data[old_date][old_time]
    if not data[old_date]:
        del data[old_date]

    new_end  = new_dt + timedelta(minutes=duration_minutes)
    new_appt = {
        **old_appt,
        "start_time":       new_start_time,
        "end_time":         new_end.strftime("%Y-%m-%d %H:%M"),
        "duration_minutes": duration_minutes,
    }
    data.setdefault(new_date, {})[new_time] = new_appt
    _save(data, property_id)

    return {"success": True, "message": f"Rescheduled from {old_start_time} to {new_start_time}.", "appointment": new_appt}


def list_upcoming(days: int = 7, property_id: Optional[str] = None) -> Dict:
    data  = _load(property_id)
    now   = datetime.now()
    limit = now + timedelta(days=days)
    appts: List[dict] = []

    current = now
    while current <= limit:
        dk = current.strftime("%Y-%m-%d")
        for appt in data.get(dk, {}).values():
            try:
                if _parse_dt(appt["start_time"]) >= now:
                    appts.append(appt)
            except ValueError:
                pass
        current += timedelta(days=1)

    appts.sort(key=lambda a: a["start_time"])
    return {"success": True, "appointments": appts, "count": len(appts)}


def get_available_slots(date: str, count: int = 5, property_id: Optional[str] = None) -> List[str]:
    data     = _load(property_id)
    day_data = data.get(date, {})
    slots: List[str] = []

    for hour in range(BUSINESS_HOURS_START, BUSINESS_HOURS_END):
        time_key = f"{hour:02d}:00"
        if time_key not in day_data:
            slots.append(f"{date} {time_key}")
        if len(slots) >= count:
            break

    return slots
