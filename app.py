from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from datetime import datetime, timedelta
from openai import OpenAI
import dateparser
import pytz
import uuid
import logging
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Timezone
# -----------------------
central = pytz.timezone("America/Chicago")

# -----------------------
# Sessions store
# -----------------------
sessions: dict = {}
MAX_NAME_LENGTH = 60
MAX_MESSAGE_LENGTH = 500
MAX_HISTORY_LENGTH = 20
MAX_NEXT_AVAILABLE_ITERATIONS = 200


# -----------------------
# Models
# -----------------------
class LumeraChatMessage(BaseModel):
    message: str
    session_id: str | None = None


# -----------------------
# Booking helpers
# -----------------------
def reset_booking():
    return {
        "service": None,
        "time": None,
        "name": None,
        "time_suggestion": None,
        "time_confirmed": False,
        "cancelling": False,
        "cancellation_name": None
    }

def get_session(session_id: str | None):
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {
            "booking": reset_booking(),
            "history": []
        }
        logger.info(f"New session created: {session_id}")
    else:
        logger.info(f"Existing session resumed: {session_id}")
    return session_id, sessions[session_id]


# -----------------------
# OpenAI helper
# -----------------------
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


# -----------------------
# Google Calendar helper
# -----------------------
def get_calendar_service():
    sa_json = os.getenv("SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("SERVICE_ACCOUNT_JSON not set")
    creds_info = json.loads(sa_json)
    credentials = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=credentials)

CALENDAR_ID = os.getenv("CALENDAR_ID")


# -----------------------
# Extractors
# -----------------------
def extract_time(text):
    parsed = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.now(),
            "TIMEZONE": "America/Chicago",
            "RETURN_AS_TIMEZONE_AWARE": True
        }
    )
    if parsed:
        parsed = parsed.astimezone(central)
        if parsed.hour == 0 and parsed.minute == 0:
            parsed = parsed.replace(hour=10)
        elif 1 <= parsed.hour <= 8:
            parsed = parsed.replace(hour=parsed.hour + 12)
        return parsed
    return None

def extract_service(text):
    service_map = {
        "deep clean": "Deep Clean",
        "deep cleaning": "Deep Clean",
        "house cleaning": "House Cleaning",
        "cleaning": "House Cleaning",
        "clean": "House Cleaning",
    }
    text_lower = text.lower()
    for phrase in sorted(service_map, key=len, reverse=True):
        if phrase in text_lower:
            return service_map[phrase]
    return None

def extract_name(text):
    skip_words = {"cancel", "appointment", "booking", "schedule", "clean", "cleaning", "yes", "no", "sure", "ok"}
    text = text.strip()
    if len(text) > MAX_NAME_LENGTH:
        return None
    text_lower = text.lower()
    triggers = ["my name is", "name is", "i am", "i'm"]
    for trigger in triggers:
        if trigger in text_lower:
            name_part = text_lower.split(trigger)[1].strip()
            words = name_part.split()
            if len(words) >= 2:
                first, last = words[0], words[1]
                if first not in skip_words and last not in skip_words:
                    return first.capitalize() + " " + last.capitalize()
    words = text.split()
    if len(words) == 2:
        first, last = words[0].lower(), words[1].lower()
        if first not in skip_words and last not in skip_words:
            return words[0].capitalize() + " " + words[1].capitalize()
    return None

def valid_business_hours(dt):
    return 0 <= dt.weekday() <= 4 and 9 <= dt.hour < 17

def find_next_available(start_dt):
    dt = start_dt
    for _ in range(MAX_NEXT_AVAILABLE_ITERATIONS):
        if valid_business_hours(dt):
            return dt
        dt += timedelta(hours=1)
        if dt.hour >= 17:
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
    return None


# -----------------------
# Calendar actions
# -----------------------
def create_calendar_event(service, name, booking_time, service_type):
    end_time = booking_time + timedelta(hours=2)
    event = {
        "summary": f"{service_type} - {name}",
        "description": f"Booked via Lumera AI chatbot.\nService: {service_type}\nClient: {name}",
        "start": {
            "dateTime": booking_time.isoformat(),
            "timeZone": "America/Chicago",
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": "America/Chicago",
        },
    }
    created = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
    return created.get("htmlLink")

def cancel_calendar_event(service, name):
    now = datetime.now(central).isoformat()
    events_result = service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=now,
        maxResults=20,
        singleEvents=True,
        orderBy="startTime"
    ).execute()
    events = events_result.get("items", [])
    for event in events:
        if name.lower() in event.get("summary", "").lower():
            service.events().delete(calendarId=CALENDAR_ID, eventId=event["id"]).execute()
            return True
    return False


# -----------------------
# AI response helper
# -----------------------
def get_ai_response(history, system_prompt):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + history,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# -----------------------
# System prompt
# -----------------------
SYSTEM_PROMPT = """You are Lumera, a friendly AI receptionist for a professional cleaning company.

You help customers with:
1. Booking house cleaning or deep cleaning appointments
2. Cancelling existing appointments
3. Answering questions about services, pricing, and availability

Services offered:
- House Cleaning: Regular maintenance cleaning. $120–$180 depending on home size.
- Deep Clean: Thorough top-to-bottom cleaning. $200–$300 depending on home size.

Business hours: Monday–Friday, 9am–5pm (Central Time)

When booking, you need to collect:
- The service type (house cleaning or deep clean)
- Their preferred date and time
- Their full name

When cancelling, you need their full name to find the appointment.

Be warm, professional, and concise. If a customer asks something outside your scope, politely let them know you can help with bookings and service questions."""


# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "Lumera Automation is running!"}


@app.post("/chat")
async def chat(body: LumeraChatMessage):
    # Validate message length
    if len(body.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail="Message too long.")

    session_id, session = get_session(body.session_id)
    booking = session["booking"]
    history = session["history"]
    user_message = body.message.strip()

    logger.info(f"[{session_id}] User: {user_message}")

    # Add user message to history
    history.append({"role": "user", "content": user_message})

    # Trim history if too long
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]
        session["history"] = history

    reply = None

    # --- Cancellation flow ---
    cancel_keywords = ["cancel", "cancell", "remove my appointment", "delete my booking"]
    if any(kw in user_message.lower() for kw in cancel_keywords) or booking.get("cancelling"):
        booking["cancelling"] = True

        if not booking.get("cancellation_name"):
            name = extract_name(user_message)
            if name:
                booking["cancellation_name"] = name
                try:
                    cal_service = get_calendar_service()
                    cancelled = cancel_calendar_event(cal_service, name)
                    if cancelled:
                        reply = f"Done! I've cancelled the appointment for {name}. Let me know if there's anything else I can help with!"
                    else:
                        reply = f"I couldn't find an upcoming appointment for {name}. Could you double-check the name, or give us a call to sort it out?"
                except Exception as e:
                    logger.error(f"Calendar cancellation error: {e}")
                    reply = "I had trouble accessing the calendar. Please try again in a moment."
                booking["cancelling"] = False
                booking["cancellation_name"] = None
            else:
                reply = "Sure, I can help cancel your appointment. Could you give me the full name the booking is under?"

    # --- Booking flow ---
    if reply is None:
        # Extract info from message
        if not booking["service"]:
            service = extract_service(user_message)
            if service:
                booking["service"] = service

        if not booking["name"]:
            name = extract_name(user_message)
            if name:
                booking["name"] = name

        if not booking["time"]:
            dt = extract_time(user_message)
            if dt:
                if valid_business_hours(dt):
                    booking["time"] = dt
                    booking["time_suggestion"] = dt
                else:
                    next_slot = find_next_available(dt)
                    if next_slot:
                        booking["time_suggestion"] = next_slot
                        reply = (
                            f"That time is outside our business hours (Mon–Fri, 9am–5pm). "
                            f"The next available slot I can offer is "
                            f"{next_slot.strftime('%A, %B %d at %I:%M %p')} CT. Does that work for you?"
                        )

        # Check if we have all info and confirm
        if booking["time_suggestion"] and not booking["time_confirmed"]:
            confirm_words = ["yes", "yeah", "sure", "ok", "okay", "that works", "sounds good", "perfect", "great"]
            if any(w in user_message.lower() for w in confirm_words):
                booking["time"] = booking["time_suggestion"]
                booking["time_confirmed"] = True

        # If all info collected, create the booking
        if booking["service"] and booking["time"] and booking["name"] and booking["time_confirmed"] and reply is None:
            try:
                cal_service = get_calendar_service()
                event_link = create_calendar_event(
                    cal_service,
                    booking["name"],
                    booking["time"],
                    booking["service"]
                )
                time_str = booking["time"].strftime("%A, %B %d at %I:%M %p")
                reply = (
                    f"You're all set, {booking['name']}! 🎉 "
                    f"Your {booking['service']} is booked for {time_str} CT. "
                    f"We'll see you then! Let me know if you need to make any changes."
                )
                # Reset booking after successful booking
                session["booking"] = reset_booking()
            except Exception as e:
                logger.error(f"Calendar booking error: {e}")
                reply = "I had trouble saving your appointment to the calendar. Please try again in a moment."

    # --- Fall back to AI for FAQ / general questions ---
    if reply is None:
        # Build context about current booking state for the AI
        booking_context = ""
        if booking["service"] or booking["name"] or booking["time_suggestion"]:
            booking_context = f"\n\nCurrent booking in progress: service={booking['service']}, name={booking['name']}, time={booking['time_suggestion']}"
            if not booking["service"]:
                booking_context += "\nStill need: service type"
            if not booking["name"]:
                booking_context += ", full name"
            if not booking["time_suggestion"]:
                booking_context += ", preferred date/time"

        reply = get_ai_response(history, SYSTEM_PROMPT + booking_context)

    logger.info(f"[{session_id}] Bot: {reply}")

    # Add bot reply to history
    history.append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "session_id": session_id,
        "booking": session["booking"]
    }