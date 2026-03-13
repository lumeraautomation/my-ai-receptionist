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

central = pytz.timezone("America/Chicago")

sessions: dict = {}
MAX_MESSAGE_LENGTH = 500
MAX_HISTORY_LENGTH = 20


class LumeraChatMessage(BaseModel):
    message: str
    session_id: str | None = None


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
        sessions[session_id] = {"booking": reset_booking(), "history": []}
    return session_id, sessions[session_id]


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


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


def extract_time(text):
    parsed = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.now(central).replace(tzinfo=None),
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

def valid_business_hours(dt):
    return 0 <= dt.weekday() <= 4 and 9 <= dt.hour < 17

def find_next_available(start_dt):
    dt = start_dt
    for _ in range(200):
        if valid_business_hours(dt):
            return dt
        dt += timedelta(hours=1)
        if dt.hour >= 17:
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            while dt.weekday() >= 5:
                dt += timedelta(days=1)
    return None


def extract_booking_info_with_ai(message, booking):
    """Use AI to extract name, business, and time from the user message."""
    client = get_openai_client()
    prompt = f"""Extract booking information from this message. Return ONLY valid JSON, no other text.

Message: "{message}"

Current known info:
- name: {booking['name']}
- business: {booking['business']}

Return JSON with these fields (use null if not found):
{{
  "name": "First Last or null",
  "business": "business name or null",
  "time_text": "the time/date mentioned verbatim or null"
}}

Rules:
- name must be a full name (first + last). Do not return single words or generic words like "demo", "call", "yes", "book".
- business is optional
- time_text is the raw date/time string from the message if any"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        logger.error(f"AI extraction error: {e}")
        return {"name": None, "business": None, "time_text": None}


def create_demo_event(service, name, business, booking_time):
    end_time = booking_time + timedelta(hours=1)
    event = {
        "summary": f"Lumera Demo Call - {name}",
        "description": f"Demo call booked via Lumera AI.\nName: {name}\nBusiness: {business or 'Not provided'}",
        "start": {"dateTime": booking_time.isoformat(), "timeZone": "America/Chicago"},
        "end": {"dateTime": end_time.isoformat(), "timeZone": "America/Chicago"},
    }
    created = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
    return created.get("htmlLink")

def cancel_demo_event(service, name):
    now = datetime.now(central).isoformat()
    events_result = service.events().list(
        calendarId=CALENDAR_ID, timeMin=now,
        maxResults=20, singleEvents=True, orderBy="startTime"
    ).execute()
    for event in events_result.get("items", []):
        if name.lower() in event.get("summary", "").lower():
            service.events().delete(calendarId=CALENDAR_ID, eventId=event["id"]).execute()
            return True
    return False


def get_ai_reply(history, booking):
    client = get_openai_client()
    booking_context = ""
    if booking["name"] or booking["time_suggestion"]:
        booking_context = (
            f"\n\nCurrent booking state: name={booking['name']}, "
            f"business={booking['business']}, time={booking['time_suggestion']}, "
            f"confirmed={booking['time_confirmed']}"
        )

    system = """You are Lumera, a friendly AI sales assistant for Lumera Automation — a company that sells AI chatbot widgets to service businesses.

Your job:
1. Answer FAQs about Lumera Automation
2. Qualify leads by asking about their business and pain points
3. Explain pricing
4. Help book free 30-minute demo calls
5. Cancel demo calls when asked

== ABOUT LUMERA AUTOMATION ==
We build AI chat widgets that help service businesses respond instantly 24/7, book appointments automatically, qualify leads, and sync with Google Calendar.
Target customers: home services, cleaning, landscaping, HVAC, medspas, salons, consultants, agencies, local businesses.

== PRICING ==
- One-Time Setup: $249
- Monthly: $79/month
- Includes: instant AI responses, smart booking, automated follow-ups, calendar sync

== BOOKING RULES — CRITICAL ==
- NEVER say the booking is confirmed or scheduled. The backend handles that.
- Collect name, optional business, and preferred time (Mon-Fri 9am-5pm CT).
- Once you have name + time, ask: "Just to confirm — [name] on [date] at [time] CT. Does that work?"
- Wait for yes/no. Do NOT announce the booking yourself.

== CANCELLING ==
Ask for their full name. Do not confirm the cancellation yourself.

== TONE ==
Warm, confident, conversational. 2-4 sentences max.""" + booking_context

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}] + history,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


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
            extracted = extract_booking_info_with_ai(user_message, booking)
            name = extracted.get("name")
            if name:
                booking["cancellation_name"] = name
                try:
                    cal_service = get_calendar_service()
                    cancelled = cancel_demo_event(cal_service, name)
                    reply = (
                        f"Done! I've cancelled the demo for {name}. Feel free to rebook anytime!"
                        if cancelled else
                        f"I couldn't find a demo for {name}. Could you double-check the name?"
                    )
                except Exception as e:
                    logger.error(f"Cancel error: {e}")
                    reply = "I had trouble accessing the calendar. Please try again."
                booking["cancelling"] = False
                booking["cancellation_name"] = None
            else:
                reply = "Sure! What's the full name the demo was booked under?"

    # --- Booking flow ---
    if reply is None:
        # Use AI to extract name, business, time from message
        extracted = extract_booking_info_with_ai(user_message, booking)

        if not booking["name"] and extracted.get("name"):
            booking["name"] = extracted["name"]
            logger.info(f"Extracted name: {booking['name']}")

        if not booking["business"] and extracted.get("business"):
            booking["business"] = extracted["business"]
            logger.info(f"Extracted business: {booking['business']}")

        if not booking["time_suggestion"] and extracted.get("time_text"):
            dt = extract_time(extracted["time_text"])
            if dt:
                if valid_business_hours(dt):
                    booking["time_suggestion"] = dt
                    logger.info(f"Extracted time: {dt}")
                else:
                    next_slot = find_next_available(dt)
                    if next_slot:
                        booking["time_suggestion"] = next_slot
                        reply = (
                            f"That time is outside our hours (Mon-Fri, 9am-5pm CT). "
                            f"Next available: {next_slot.strftime('%A, %B %d at %I:%M %p')} CT. "
                            f"Does that work?"
                        )

        # Check for confirmation
        confirm_words = ["yes", "yeah", "sure", "ok", "okay", "that works", "sounds good", "perfect", "great", "confirmed", "yep", "yup"]
        if booking["time_suggestion"] and not booking["time_confirmed"]:
            if any(w in user_message.lower() for w in confirm_words):
                booking["time"] = booking["time_suggestion"]
                booking["time_confirmed"] = True
                logger.info("Time confirmed")

        # All info collected — create booking
        if booking["name"] and booking["time"] and booking["time_confirmed"] and reply is None:
            try:
                cal_service = get_calendar_service()
                create_demo_event(cal_service, booking["name"], booking["business"], booking["time"])
                time_str = booking["time"].strftime("%A, %B %d at %I:%M %p")
                reply = (
                    f"You're all booked, {booking['name']}! 🎉 "
                    f"Your free 30-minute demo is set for {time_str} CT. "
                    f"We'll show you exactly how Lumera can work for your business. See you then!"
                )
                session["booking"] = reset_booking()
            except Exception as e:
                logger.error(f"Booking error: {e}")
                reply = "I had trouble saving to the calendar. Please try again in a moment."

    # --- AI fallback ---
    if reply is None:
        reply = get_ai_reply(history, booking)

    logger.info(f"[{session_id}] Bot: {reply}")
    history.append({"role": "assistant", "content": reply})

    return {"reply": reply, "session_id": session_id, "booking": session["booking"]}
