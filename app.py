from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        "name": None,
        "business": None,
        "time": None,
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

def extract_name(text):
    skip_words = {"cancel", "appointment", "booking", "demo", "call", "schedule", "yes", "no", "sure", "ok"}
    text = text.strip()
    if len(text) > MAX_NAME_LENGTH:
        return None
    text_lower = text.lower()
    triggers = ["my name is", "name is", "i am", "i'm", "this is"]
    for trigger in triggers:
        if trigger in text_lower:
            name_part = text_lower.split(trigger)[1].strip()
            words = name_part.split()
            if len(words) >= 2:
                first, last = words[0], words[1]
                if first not in skip_words and last not in skip_words:
                    return words[0].capitalize() + " " + words[1].capitalize()
    words = text.split()
    if len(words) == 2:
        first, last = words[0].lower(), words[1].lower()
        if first not in skip_words and last not in skip_words:
            return words[0].capitalize() + " " + words[1].capitalize()
    return None

def extract_business(text):
    triggers = ["my business is", "i own", "i run", "company is", "business is", "we are", "i work at"]
    text_lower = text.lower()
    for trigger in triggers:
        if trigger in text_lower:
            part = text_lower.split(trigger)[1].strip()
            return part.split(".")[0].split(",")[0].strip().title()
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
def create_demo_event(service, name, business, booking_time):
    end_time = booking_time + timedelta(hours=1)
    event = {
        "summary": f"Lumera Demo Call - {name}",
        "description": (
            f"Demo call booked via Lumera AI chatbot.\n"
            f"Name: {name}\n"
            f"Business: {business or 'Not provided'}\n"
        ),
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

def cancel_demo_event(service, name):
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
SYSTEM_PROMPT = """You are Lumera, a friendly and professional AI sales assistant for Lumera Automation — a company that sells AI chatbot widgets to service businesses.

Your job is to:
1. Answer FAQs about Lumera Automation's product and services
2. Qualify leads by understanding their business type and needs
3. Explain pricing clearly
4. Book free 30-minute strategy/demo calls
5. Cancel existing demo calls when requested

== ABOUT LUMERA AUTOMATION ==
Lumera Automation builds AI-powered chat widgets that help service businesses:
- Respond instantly to leads 24/7
- Book appointments automatically
- Qualify leads and follow up
- Sync with Google Calendar

Target customers: home services, cleaning companies, landscaping, HVAC, medspas, salons, consultants, agencies, and any local service business.

== PRICING ==
- One-Time Setup Fee: $249
- Monthly Subscription: $79/month
- Includes: instant AI responses, smart booking, automated follow-ups, calendar sync

== QUALIFYING QUESTIONS ==
When someone shows interest, naturally ask:
- What type of business do they run?
- Are they currently losing leads or missing follow-ups?
- How are they currently handling bookings?

Use their answers to explain how Lumera solves their specific problem before pushing the demo.

== BOOKING A DEMO ==
When someone wants to book a demo, collect:
- Their full name
- Their business name (optional but helpful)
- Preferred date and time (Mon-Fri, 9am-5pm Central)

Demo calls are 30 minutes and free.

CRITICAL BOOKING RULES — YOU MUST FOLLOW THESE:
- NEVER say "I've booked your demo", "you're confirmed", "booking confirmed", or anything suggesting the appointment is scheduled. The backend system handles the actual booking — you just collect the info.
- NEVER confirm or announce a booking. Only ask for missing info (name, time).
- Once you have their name and time, simply say something like "Got it! Let me confirm — [name] on [date] at [time] CT. Does that work?" and wait for them to confirm with yes/no.
- Only after they say yes will the system actually create the calendar event and send the real confirmation.

== CANCELLING ==
If someone wants to cancel, get their full name to find the booking.

== TONE ==
Be warm, confident, and conversational. Don't be pushy. Focus on understanding their pain points first, then show how Lumera solves them. Keep responses concise — 2-4 sentences max unless explaining pricing or features."""


# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "Lumera Automation is running!"}


@app.post("/chat")
async def chat(body: LumeraChatMessage):
    if len(body.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail="Message too long.")

    session_id, session = get_session(body.session_id)
    booking = session["booking"]
    history = session["history"]
    user_message = body.message.strip()

    logger.info(f"[{session_id}] User: {user_message}")

    history.append({"role": "user", "content": user_message})

    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]
        session["history"] = history

    reply = None

    # --- Cancellation flow ---
    cancel_keywords = ["cancel", "remove my call", "delete my demo", "cancel my booking", "cancel my demo"]
    if any(kw in user_message.lower() for kw in cancel_keywords) or booking.get("cancelling"):
        booking["cancelling"] = True

        if not booking.get("cancellation_name"):
            name = extract_name(user_message)
            if name:
                booking["cancellation_name"] = name
                try:
                    cal_service = get_calendar_service()
                    cancelled = cancel_demo_event(cal_service, name)
                    if cancelled:
                        reply = f"Done! I've cancelled the demo call for {name}. Feel free to rebook anytime — just let me know!"
                    else:
                        reply = f"I couldn't find an upcoming demo for {name}. Could you double-check the name used when booking?"
                except Exception as e:
                    logger.error(f"Calendar cancellation error: {e}")
                    reply = "I had trouble accessing the calendar. Please try again in a moment."
                booking["cancelling"] = False
                booking["cancellation_name"] = None
            else:
                reply = "Sure, I can cancel your demo call. What's the full name it was booked under?"

    # --- Booking flow ---
    if reply is None:
        if not booking["name"]:
            name = extract_name(user_message)
            if name:
                booking["name"] = name

        if not booking["business"]:
            business = extract_business(user_message)
            if business:
                booking["business"] = business

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
                            f"That time is outside our available hours (Mon-Fri, 9am-5pm CT). "
                            f"The next available slot is {next_slot.strftime('%A, %B %d at %I:%M %p')} CT. "
                            f"Does that work for you?"
                        )

        # Confirm suggested time
        if booking["time_suggestion"] and not booking["time_confirmed"]:
            confirm_words = ["yes", "yeah", "sure", "ok", "okay", "that works", "sounds good", "perfect", "great", "confirmed"]
            if any(w in user_message.lower() for w in confirm_words):
                booking["time"] = booking["time_suggestion"]
                booking["time_confirmed"] = True

        # All info collected — create the booking
        if booking["name"] and booking["time"] and booking["time_confirmed"] and reply is None:
            try:
                cal_service = get_calendar_service()
                create_demo_event(
                    cal_service,
                    booking["name"],
                    booking["business"],
                    booking["time"]
                )
                time_str = booking["time"].strftime("%A, %B %d at %I:%M %p")
                reply = (
                    f"You're all booked, {booking['name']}! 🎉 "
                    f"Your free 30-minute demo call is set for {time_str} CT. "
                    f"We'll walk you through everything and show you exactly how Lumera can work for your business. See you then!"
                )
                session["booking"] = reset_booking()
            except Exception as e:
                logger.error(f"Calendar booking error: {e}")
                reply = "I had trouble saving your demo to the calendar. Please try again in a moment."

    # --- AI fallback for FAQs, qualifying, pricing questions ---
    if reply is None:
        booking_context = ""
        if booking["name"] or booking["time_suggestion"]:
            booking_context = f"\n\nDemo booking in progress: name={booking['name']}, business={booking['business']}, time={booking['time_suggestion']}, confirmed={booking['time_confirmed']}"
            if not booking["name"]:
                booking_context += "\nStill need: full name"
            if not booking["time_suggestion"]:
                booking_context += ", preferred date/time"

        reply = get_ai_response(history, SYSTEM_PROMPT + booking_context)

    logger.info(f"[{session_id}] Bot: {reply}")
    history.append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "session_id": session_id,
        "booking": session["booking"]
    }
