# Property Chatbot

A conversational AI assistant for real estate, designed for seamless integration into Django applications via WebSockets. This chatbot helps prospective buyers explore property details and manage site visits (bookings) through an intuitive, state-aware interface.

## 🚀 Key Features

-   **Intelligent Q&A**: Answers specific questions about property amenities, pricing, location, and policies using NVIDIA NIM (Llama 3.1 70B).
-   **Appointment Management**: Full-lifecycle support for site visits:
    -   **Book**: Interactive flow to check availability and schedule visits.
    -   **Reschedule**: Easily move existing appointments to a new time.
    -   **Cancel**: Remove upcoming visits.
    -   **List**: View all scheduled appointments for a property.
-   **Deterministic Flow**: Sensitive operations like booking use structured logic rather than LLM guesswork to ensure accuracy and reliability.
-   **LangGraph-Powered**: Uses a state-graph architecture to maintain context across multi-turn conversations.
-   **Quick Replies**: Automatically generates interactive UI chips for common actions, improving user engagement and conversion.
-   **Multi-Property Aware**: Scopes appointments and AI context per property listing, preventing data collisions.
-   **Regional Validation**: Built-in phone number validation for India (+91) and Costa Rica (+506).

## 🛠️ Tech Stack

-   **Logic**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain)
-   **LLM**: NVIDIA AI Endpoints (NIM)
-   **Backend**: Python, designed for [Django Channels](https://channels.readthedocs.io/)
-   **Persistence**: JSON-based (default), architected for easy migration to Django ORM or Google Calendar.

## 📁 Project Structure

```text
D:\click_chatbot\
├── appointment_manager.py  # Appointment CRUD and JSON persistence logic
├── config.py               # Central configuration (LLM, business hours, paths)
├── context_formatter.py    # Helper to convert property data for LLM consumption
├── graph_state.py          # Definition of the conversation state (TypedDict)
├── graph.py                # LangGraph topology and compilation
├── nodes.py                # Core logic: Intent detection, booking flows, and LLM Q&A
└── INTEGRATION_GUIDE.md    # Detailed instructions for Django developers
```

## ⚙️ Configuration

Key settings in `config.py`:
-   `BUSINESS_HOURS_START` / `END`: Default 11 AM – 6 PM.
-   `APPOINTMENT_DURATION_MINUTES`: Default 60 mins.
-   `model`: Currently configured for `meta/llama-3.1-70b-instruct`.

## 📦 Setup

1.  **Dependencies**:
    ```bash
    pip install langgraph langchain langchain-core langchain-nvidia-ai-endpoints phonenumbers
    ```
2.  **API Key**: Add `NVIDIA_API_KEY` to your environment or `.env` file.
3.  **Integration**: Follow the [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) to plug the chatbot into your Django project.
