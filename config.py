"""
config.py – Model configuration and shared constants for the property chatbot.

This is the SINGLE place to change:
  - The LLM model / provider
  - Business hours and booking constraints
  - The path to the appointments JSON file

To swap providers replace the ChatNVIDIA block with ChatOpenAI / ChatAnthropic etc.
"""

from __future__ import annotations

import os
import sys
from django.conf import settings

# ── Environment ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("NVIDIA_API_KEY"):
    print("Warning: NVIDIA_API_KEY environment variable is not set.")
    print("Please set it or create a .env file with NVIDIA_API_KEY=nvapi-...")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Booking / calendar constants ───────────────────────────────────────────────
# Anchor the calendar file to this package's directory so it always resolves
# correctly regardless of Django's working directory.
_CHATBOT_DIR                 = os.path.dirname(os.path.abspath(__file__))
CALENDAR_FILE                = os.path.join(settings.BASE_DIR, "appointments.json")
APPOINTMENT_DURATION_MINUTES = 60
BUSINESS_HOURS_START         = 11   # 11 AM inclusive
BUSINESS_HOURS_END           = 18   # 6 PM exclusive
MAX_SLOTS_PER_DAY            = 8

# ── LLM – NVIDIA NIM (Llama 3.1 70B) ──────────────────────────────────────────
# To swap providers replace this block only, e.g.:
#   from langchain_openai import ChatOpenAI
#   model = ChatOpenAI(model="gpt-4o", temperature=0.2)
from langchain_nvidia_ai_endpoints import ChatNVIDIA

model = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    temperature=0.2,
    max_tokens=700,
)

 