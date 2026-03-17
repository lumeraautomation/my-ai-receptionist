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
import sqlite3

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

CLIENT_ID = os.getenv("CLIENT_ID", "lumera_demo")
DB_PATH = os.getenv("ANALYTICS_DB_PATH", "analytics.db")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                client_id  TEXT NOT NULL,
                timestamp  TEXT NOT NULL
            )
        """)
        # NEW: bookings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                client_id  TEXT NOT NULL,
                name       TEXT NOT NULL,
                business   TEXT,
                start_time TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL
            )
        """)
        # NEW: leads table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                client_id  TEXT NOT NULL,
                name       TEXT,
                business   TEXT,
                email      TEXT,
                phone      TEXT,
                source     TEXT DEFAULT 'Chat Widget',
                status     TEXT NOT NULL DEFAULT 'new',
                created_at TEXT NOT NULL
            )
        """)
        # Add email/phone columns if upgrading from older DB
        try:
            conn.execute("ALTER TABLE leads ADD COLUMN email TEXT")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE leads ADD COLUMN phone TEXT")
        except Exception:
            pass
        conn.commit()


def log_event(event_type: str, session_id: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO events (event_type, session_id, client_id, timestamp) VALUES (?, ?, ?, ?)",
                (event_type, session_id, CLIENT_ID, datetime.now(central).isoformat())
            )
            conn.commit()
        logger.info(f"[analytics] {event_type} | client={CLIENT_ID} | session={session_id}")
    except Exception as e:
        logger.error(f"[analytics] Failed to log event: {e}")


def log_booking(session_id: str, name: str, business: str, start_time: datetime):
    """Save a booking record to the bookings table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO bookings (session_id, client_id, name, business, start_time, status, created_at)
                   VALUES (?, ?, ?, ?, ?, 'confirmed', ?)""",
                (session_id, CLIENT_ID, name, business or "",
                 start_time.isoformat(), datetime.now(central).isoformat())
            )
            conn.commit()
        logger.info(f"[booking] saved | name={name} | time={start_time}")
    except Exception as e:
        logger.error(f"[booking] Failed to save: {e}")


def log_lead(session_id: str, name: str, business: str):
    """Upsert a lead record — one row per session."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            existing = conn.execute(
                "SELECT id FROM leads WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE leads SET name=?, business=? WHERE session_id=?",
                    (name, business or "", session_id)
                )
            else:
                conn.execute(
                    """INSERT INTO leads (session_id, client_id, name, business, source, status, created_at)
                       VALUES (?, ?, ?, ?, 'Chat Widget', 'new', ?)""",
                    (session_id, CLIENT_ID, name, business or "",
                     datetime.now(central).isoformat())
                )
            conn.commit()
    except Exception as e:
        logger.error(f"[lead] Failed to save: {e}")


init_db()


# ── Clients table (active paying clients) ─────────────────────────────────────
def init_clients():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL,
                business     TEXT,
                email        TEXT,
                phone        TEXT,
                status       TEXT NOT NULL DEFAULT 'active',
                setup_paid   INTEGER NOT NULL DEFAULT 1,
                monthly_fee  REAL NOT NULL DEFAULT 99.0,
                setup_fee    REAL NOT NULL DEFAULT 499.0,
                start_date   TEXT NOT NULL,
                notes        TEXT,
                client_id    TEXT NOT NULL
            )
        """)
        conn.commit()

init_clients()


class ClientCreate(BaseModel):
    name: str
    business: str | None = None
    email: str | None = None
    phone: str | None = None
    status: str | None = "active"
    setup_paid: int | None = 1
    monthly_fee: float | None = 99.0
    setup_fee: float | None = 499.0
    start_date: str | None = None
    notes: str | None = None

class ClientUpdate(BaseModel):
    name: str | None = None
    business: str | None = None
    email: str | None = None
    phone: str | None = None
    status: str | None = None
    setup_paid: int | None = None
    monthly_fee: float | None = None
    notes: str | None = None

@app.get("/revenue")
def get_revenue():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            clients = [dict(c) for c in conn.execute(
                "SELECT * FROM clients WHERE client_id=? ORDER BY start_date DESC",
                (CLIENT_ID,)
            ).fetchall()]

        active  = [c for c in clients if c['status'] == 'active']
        churned = [c for c in clients if c['status'] == 'churned']
        trial   = [c for c in clients if c['status'] == 'trial']

        mrr         = sum(c['monthly_fee'] for c in active)
        setup_total = sum(c['setup_fee'] for c in clients if c['setup_paid'])

        from collections import defaultdict
        monthly = defaultdict(lambda: {"new_clients": 0, "setup_revenue": 0, "mrr_snapshot": 0})
        for c in clients:
            if not c['start_date']:
                continue
            month = c['start_date'][:7]
            if c['setup_paid']:
                monthly[month]['setup_revenue'] += c['setup_fee']
            if c['status'] in ('active', 'trial'):
                monthly[month]['new_clients'] += 1

        running_mrr = 0
        for m in sorted(monthly.keys()):
            running_mrr += monthly[m]['new_clients'] * 99
            monthly[m]['mrr_snapshot'] = running_mrr

        return {
            "summary": {
                "mrr":            round(mrr, 2),
                "arr":            round(mrr * 12, 2),
                "setup_total":    round(setup_total, 2),
                "total_revenue":  round(mrr + setup_total, 2),
                "active_clients": len(active),
                "trial_clients":  len(trial),
                "churned_clients":len(churned),
                "total_clients":  len(clients),
            },
            "clients": clients,
            "monthly": dict(sorted(monthly.items())),
        }
    except Exception as e:
        logger.error(f"Revenue error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch revenue data.")

@app.post("/revenue/clients")
def create_client(body: ClientCreate):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO clients (name, business, email, phone, status, setup_paid,
                   monthly_fee, setup_fee, start_date, notes, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (body.name, body.business or "", body.email or "", body.phone or "",
                 body.status or "active", 1 if body.setup_paid is None else body.setup_paid,
                 body.monthly_fee or 99.0, body.setup_fee or 499.0,
                 body.start_date or datetime.now(central).strftime("%Y-%m-%d"),
                 body.notes or "", CLIENT_ID)
            )
            conn.commit()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Client create error: {e}")
        raise HTTPException(status_code=500, detail="Could not create client.")

@app.patch("/revenue/clients/{cid}")
def update_client(cid: int, body: ClientUpdate):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for field, val in body.model_dump(exclude_none=True).items():
                conn.execute(f"UPDATE clients SET {field}=? WHERE id=? AND client_id=?",
                             (val, cid, CLIENT_ID))
            conn.commit()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not update client.")

@app.delete("/revenue/clients/{cid}")
def delete_client(cid: int):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM clients WHERE id=? AND client_id=?", (cid, CLIENT_ID))
            conn.commit()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not delete client.")


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
        log_event("widget_view", session_id)
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
- name must be a full name (first + last). Do not return single words or generic words like "strategy", "call", "yes", "book", "cancel".
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


def create_strategy_call_event(service, name, business, booking_time):
    end_time = booking_time + timedelta(hours=1)
    event = {
        "summary": f"Lumera Strategy Call - {name}",
        "description": f"Strategy call booked via Lumera AI.\nName: {name}\nBusiness: {business or 'Not provided'}",
        "start": {"dateTime": booking_time.isoformat(), "timeZone": "America/Chicago"},
        "end": {"dateTime": end_time.isoformat(), "timeZone": "America/Chicago"},
    }
    created = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
    return created.get("htmlLink")


def cancel_strategy_call_event(service, name):
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
4. Help book free 30-minute strategy calls
5. Cancel strategy calls when asked

== ABOUT LUMERA AUTOMATION ==
We build AI chat widgets that help service businesses respond instantly 24/7, book appointments automatically, qualify leads, and sync with Google Calendar.
The chatbot the user is talking to RIGHT NOW is a live example of what we build for clients.
Target customers: home services, cleaning, landscaping, HVAC, medspas, salons, consultants, agencies, local businesses.

== PRICING ==
- One-Time Setup: $499
- Monthly: $99/month
- Includes: instant AI responses, smart booking, automated follow-ups, calendar sync

== QUALIFYING ==
When someone shows interest, naturally ask:
- What type of business do they run?
- Are they losing leads or missing follow-ups?
- How are they currently handling bookings?
Use their answers to show how Lumera solves their specific problem.

== BOOKING A STRATEGY CALL ==
When someone wants to book, collect:
- Their full name
- Their business name (optional)
- Preferred date and time (Mon-Fri, 9am-5pm CT)

Strategy calls are 30 minutes and free. We'll discuss their business and show how Lumera can work for you.

CRITICAL BOOKING RULES:
- NEVER say the call is confirmed or booked. The backend handles that.
- Once you have name + time, ask: "Just to confirm — [name] on [date] at [time] CT. Does that work?"
- Wait for yes/no before anything else.

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


@app.get("/availability")
async def get_availability():
    try:
        cal_service = get_calendar_service()
        now = datetime.now(central)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=14)).isoformat()

        events_result = cal_service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        busy_times = []
        for event in events_result.get("items", []):
            start = event["start"].get("dateTime")
            end = event["end"].get("dateTime")
            if start and end:
                busy_times.append({"start": start, "end": end})

        available_slots = []
        current = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        while current <= now + timedelta(days=14):
            if 0 <= current.weekday() <= 4 and 9 <= current.hour <= 16:
                slot_end = current + timedelta(hours=1)
                is_busy = False
                for busy in busy_times:
                    busy_start = datetime.fromisoformat(busy["start"]).astimezone(central)
                    busy_end = datetime.fromisoformat(busy["end"]).astimezone(central)
                    if current < busy_end and slot_end > busy_start:
                        is_busy = True
                        break
                if not is_busy:
                    available_slots.append({
                        "start": current.isoformat(),
                        "end": slot_end.isoformat(),
                        "display": current.strftime("%A, %B %d at %I:%M %p") + " CT"
                    })
            current += timedelta(hours=1)

        return {"slots": available_slots}

    except Exception as e:
        logger.error(f"Availability error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch availability.")


@app.get("/analytics")
def get_analytics(client_id: str | None = None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            if client_id == "all":
                rows = conn.execute("SELECT * FROM events ORDER BY timestamp").fetchall()
            else:
                target = client_id or CLIENT_ID
                rows = conn.execute(
                    "SELECT * FROM events WHERE client_id = ? ORDER BY timestamp",
                    (target,)
                ).fetchall()

        from collections import defaultdict
        monthly = defaultdict(lambda: {"views": 0, "bookings": 0, "cancellations": 0})
        total_views = 0
        total_bookings = 0
        total_cancellations = 0

        for row in rows:
            month = row["timestamp"][:7]
            et = row["event_type"]
            if et == "widget_view":
                monthly[month]["views"] += 1
                total_views += 1
            elif et == "booking_created":
                monthly[month]["bookings"] += 1
                total_bookings += 1
            elif et == "booking_cancelled":
                monthly[month]["cancellations"] += 1
                total_cancellations += 1

        conv_rate = round((total_bookings / total_views * 100), 1) if total_views else 0
        sorted_monthly = dict(sorted(monthly.items()))

        return {
            "client_id": client_id or CLIENT_ID,
            "total_views": total_views,
            "total_bookings": total_bookings,
            "total_cancellations": total_cancellations,
            "conversion_rate": conv_rate,
            "monthly": sorted_monthly
        }

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch analytics.")


# ── NEW: Bookings endpoint ─────────────────────────────────────────────────────
@app.get("/bookings")
def get_bookings(client_id: str | None = None, limit: int = 50):
    """Return recent bookings for the admin dashboard."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            target = client_id or CLIENT_ID
            rows = conn.execute(
                """SELECT * FROM bookings WHERE client_id = ?
                   ORDER BY start_time ASC LIMIT ?""",
                (target, limit)
            ).fetchall()
        return {"bookings": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Bookings fetch error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch bookings.")


# ── NEW: Leads endpoint ────────────────────────────────────────────────────────
@app.get("/leads")
def get_leads(client_id: str | None = None, limit: int = 100):
    """Return leads for the admin dashboard."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            target = client_id or CLIENT_ID
            rows = conn.execute(
                """SELECT * FROM leads WHERE client_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (target, limit)
            ).fetchall()
        return {"leads": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Leads fetch error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch leads.")


# ── NEW: Update lead status ────────────────────────────────────────────────────
class LeadUpdate(BaseModel):
    status: str  # new | contacted | qualified | lost

@app.patch("/leads/{lead_id}")
def update_lead(lead_id: int, body: LeadUpdate):
    """Let the admin dashboard update a lead's status."""
    valid = {"new", "contacted", "qualified", "lost"}
    if body.status not in valid:
        raise HTTPException(status_code=400, detail=f"status must be one of {valid}")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE leads SET status=? WHERE id=?", (body.status, lead_id))
            conn.commit()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Lead update error: {e}")
        raise HTTPException(status_code=500, detail="Could not update lead.")


@app.delete("/leads/{lead_id}")
def delete_lead(lead_id: int):
    """Delete a single lead."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM leads WHERE id=? AND client_id=?", (lead_id, CLIENT_ID))
            conn.commit()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Lead delete error: {e}")
        raise HTTPException(status_code=500, detail="Could not delete lead.")
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
    cancel_keywords = ["cancel", "remove my call", "delete my call", "cancel my booking", "cancel my strategy"]
    if any(kw in user_message.lower() for kw in cancel_keywords) or booking.get("cancelling"):
        booking["cancelling"] = True
        if not booking.get("cancellation_name"):
            extracted = extract_booking_info_with_ai(user_message, booking)
            name = extracted.get("name")
            if name:
                booking["cancellation_name"] = name
                try:
                    cal_service = get_calendar_service()
                    cancelled = cancel_strategy_call_event(cal_service, name)
                    if cancelled:
                        log_event("booking_cancelled", session_id)
                        # Mark booking as cancelled in DB
                        try:
                            with sqlite3.connect(DB_PATH) as conn:
                                conn.execute(
                                    "UPDATE bookings SET status='cancelled' WHERE name LIKE ? AND client_id=?",
                                    (f"%{name}%", CLIENT_ID)
                                )
                                conn.commit()
                        except Exception:
                            pass
                        reply = (
                            f"Done! I've cancelled the strategy call for {name}. Feel free to rebook anytime!"
                        )
                    else:
                        reply = f"I couldn't find a strategy call for {name}. Could you double-check the name?"
                except Exception as e:
                    logger.error(f"Cancel error: {e}")
                    reply = "I had trouble accessing the calendar. Please try again."
                booking["cancelling"] = False
                booking["cancellation_name"] = None
            else:
                reply = "Sure! What's the full name the strategy call was booked under?"

    # --- Booking flow ---
    if reply is None:
        extracted = extract_booking_info_with_ai(user_message, booking)

        if not booking["name"] and extracted.get("name"):
            booking["name"] = extracted["name"]
            logger.info(f"Extracted name: {booking['name']}")

        if not booking["business"] and extracted.get("business"):
            booking["business"] = extracted["business"]
            logger.info(f"Extracted business: {booking['business']}")

        # Save as lead as soon as we have a name
        if booking["name"]:
            log_lead(session_id, booking["name"], booking.get("business", ""))

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

        confirm_words = ["yes", "yeah", "sure", "ok", "okay", "that works", "sounds good", "perfect", "great", "confirmed", "yep", "yup"]
        if booking["time_suggestion"] and not booking["time_confirmed"]:
            if any(w in user_message.lower() for w in confirm_words):
                booking["time"] = booking["time_suggestion"]
                booking["time_confirmed"] = True

        # All info collected — create booking
        if booking["name"] and booking["time"] and booking["time_confirmed"] and reply is None:
            try:
                cal_service = get_calendar_service()
                create_strategy_call_event(cal_service, booking["name"], booking["business"], booking["time"])
                log_event("booking_created", session_id)
                # Save to bookings table
                log_booking(session_id, booking["name"], booking["business"], booking["time"])
                # Update lead status to qualified
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                            "UPDATE leads SET status='qualified' WHERE session_id=? AND client_id=?",
                            (session_id, CLIENT_ID)
                        )
                        conn.commit()
                except Exception:
                    pass

                time_str = booking["time"].strftime("%A, %B %d at %I:%M %p")
                reply = (
                    f"You're all set, {booking['name']}! 🎉 "
                    f"Your free 30-minute strategy call is booked for {time_str} CT. "
                    f"We'll walk through your business and show exactly how Lumera can work for you. See you then!"
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


# ── NEW: Manually add a lead ───────────────────────────────────────────────────
class LeadCreate(BaseModel):
    name: str
    business: str | None = None
    email: str | None = None
    phone: str | None = None
    source: str | None = "Manual Entry"
    status: str | None = "new"

@app.post("/leads")
def create_lead(body: LeadCreate):
    """Manually add a lead from the admin dashboard."""
    valid = {"new", "contacted", "qualified", "lost"}
    if body.status not in valid:
        raise HTTPException(status_code=400, detail=f"status must be one of {valid}")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO leads (session_id, client_id, name, business, email, phone, source, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), CLIENT_ID, body.name, body.business or "",
                 body.email or "", body.phone or "",
                 body.source or "Manual Entry", body.status,
                 datetime.now(central).isoformat())
            )
            conn.commit()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Lead create error: {e}")
        raise HTTPException(status_code=500, detail="Could not create lead.")


class BulkLeadCreate(BaseModel):
    leads: list[LeadCreate]

@app.post("/leads/bulk")
def create_leads_bulk(body: BulkLeadCreate):
    """Import multiple leads at once (CSV import from admin dashboard)."""
    if len(body.leads) > 5000:
        raise HTTPException(status_code=400, detail="Max 5000 leads per import.")
    inserted = 0
    skipped = 0
    now = datetime.now(central).isoformat()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for lead in body.leads:
                if not lead.name or not lead.name.strip():
                    skipped += 1
                    continue
                valid = {"new", "contacted", "qualified", "lost"}
                status = lead.status if lead.status in valid else "new"
                conn.execute(
                    """INSERT INTO leads (session_id, client_id, name, business, email, phone, source, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), CLIENT_ID, lead.name.strip(),
                     lead.business or "", lead.email or "", lead.phone or "",
                     lead.source or "CSV Import", status, now)
                )
                inserted += 1
            conn.commit()
        return {"ok": True, "inserted": inserted, "skipped": skipped}
    except Exception as e:
        logger.error(f"Bulk lead create error: {e}")
        raise HTTPException(status_code=500, detail="Could not import leads.")


class BulkDeleteRequest(BaseModel):
    ids: list[int]

@app.post("/leads/delete-bulk")
def delete_leads_bulk(body: BulkDeleteRequest):
    """Delete multiple leads by ID."""
    if not body.ids:
        raise HTTPException(status_code=400, detail="No IDs provided.")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            placeholders = ','.join('?' * len(body.ids))
            conn.execute(
                f"DELETE FROM leads WHERE id IN ({placeholders}) AND client_id=?",
                (*body.ids, CLIENT_ID)
            )
            conn.commit()
        return {"ok": True, "deleted": len(body.ids)}
    except Exception as e:
        logger.error(f"Bulk lead delete error: {e}")
        raise HTTPException(status_code=500, detail="Could not delete leads.")


import hashlib
import secrets

# ── Users table ───────────────────────────────────────────────────────────────
def init_users():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                name          TEXT,
                role          TEXT NOT NULL DEFAULT 'staff',
                client_id     TEXT NOT NULL DEFAULT 'lumera_demo',
                active        INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT NOT NULL
            )
        """)
        conn.commit()
        # Migrate: add client_id if upgrading older DB
        try:
            conn.execute("ALTER TABLE admin_users ADD COLUMN client_id TEXT NOT NULL DEFAULT 'lumera_demo'")
            conn.commit()
        except Exception:
            pass
        # Seed default super-admin
        default_email = os.getenv("ADMIN_EMAIL", "kory@lumeraautomation.com")
        default_pass  = os.getenv("ADMIN_PASSWORD", "lumera2026")
        existing = conn.execute(
            "SELECT id FROM admin_users WHERE email=?", (default_email,)
        ).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO admin_users (email, password_hash, name, role, client_id, active, created_at)
                   VALUES (?, ?, ?, 'superadmin', 'lumera_demo', 1, ?)""",
                (default_email, _hash(default_pass), "Kory",
                 datetime.now(central).isoformat())
            )
            conn.commit()
            logger.info(f"[users] Seeded superadmin: {default_email}")
        else:
            # Upgrade existing Kory account to superadmin
            conn.execute(
                "UPDATE admin_users SET role='superadmin', name='Kory' WHERE email=?",
                (default_email,)
            )
            conn.commit()

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

init_users()


class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/login")
def login(body: LoginRequest):
    """Validate admin credentials. Returns user info on success."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            user = conn.execute(
                "SELECT * FROM admin_users WHERE email=? AND active=1",
                (body.email.strip().lower(),)
            ).fetchone()
        if not user or user["password_hash"] != _hash(body.password):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        return {
            "ok": True,
            "user": {
                "id":        user["id"],
                "email":     user["email"],
                "name":      user["name"],
                "role":      user["role"],
                "client_id": user["client_id"] if "client_id" in user.keys() else "lumera_demo",
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed.")


class UserCreate(BaseModel):
    email: str
    password: str
    name: str | None = None
    role: str | None = "staff"
    client_id: str | None = "lumera_demo"

@app.get("/auth/users")
def get_users():
    """List all admin users."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, email, name, role, client_id, active, created_at FROM admin_users ORDER BY created_at"
            ).fetchall()
        return {"users": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Get users error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch users.")

@app.post("/auth/users")
def create_user(body: UserCreate):
    """Add a new admin user."""
    valid_roles = {"admin", "staff", "client"}
    if body.role not in valid_roles:
        raise HTTPException(status_code=400, detail="role must be admin, staff, or client")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO admin_users (email, password_hash, name, role, client_id, active, created_at)
                   VALUES (?, ?, ?, ?, ?, 1, ?)""",
                (body.email.strip().lower(), _hash(body.password),
                 body.name or body.email.split("@")[0], body.role,
                 body.client_id or "lumera_demo",
                 datetime.now(central).isoformat())
            )
            conn.commit()
        return {"ok": True}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email already exists.")
    except Exception as e:
        logger.error(f"Create user error: {e}")
        raise HTTPException(status_code=500, detail="Could not create user.")

class UserUpdate(BaseModel):
    name: str | None = None
    password: str | None = None
    role: str | None = None
    client_id: str | None = None
    active: int | None = None

@app.patch("/auth/users/{user_id}")
def update_user(user_id: int, body: UserUpdate):
    """Update name, password, role, client_id, or active status."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if body.name is not None:
                conn.execute("UPDATE admin_users SET name=? WHERE id=?", (body.name, user_id))
            if body.password is not None:
                conn.execute("UPDATE admin_users SET password_hash=? WHERE id=?", (_hash(body.password), user_id))
            if body.role is not None:
                conn.execute("UPDATE admin_users SET role=? WHERE id=?", (body.role, user_id))
            if body.client_id is not None:
                conn.execute("UPDATE admin_users SET client_id=? WHERE id=?", (body.client_id, user_id))
            if body.active is not None:
                conn.execute("UPDATE admin_users SET active=? WHERE id=?", (body.active, user_id))
            conn.commit()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Update user error: {e}")
        raise HTTPException(status_code=500, detail="Could not update user.")

@app.delete("/auth/users/{user_id}")
def delete_user(user_id: int):
    """Remove an admin user."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM admin_users WHERE id=?", (user_id,))
            conn.commit()
        return {"ok": True}
    except Exception as e:
        logger.error(f"Delete user error: {e}")
        raise HTTPException(status_code=500, detail="Could not delete user.")


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
    cancel_keywords = ["cancel", "remove my call", "delete my call", "cancel my booking", "cancel my strategy"]
    if any(kw in user_message.lower() for kw in cancel_keywords) or booking.get("cancelling"):
        booking["cancelling"] = True
        if not booking.get("cancellation_name"):
            extracted = extract_booking_info_with_ai(user_message, booking)
            name = extracted.get("name")
            if name:
                booking["cancellation_name"] = name
                try:
                    cal_service = get_calendar_service()
                    cancelled = cancel_strategy_call_event(cal_service, name)
                    if cancelled:
                        log_event("booking_cancelled", session_id)
                        try:
                            with sqlite3.connect(DB_PATH) as conn:
                                conn.execute(
                                    "UPDATE bookings SET status='cancelled' WHERE name LIKE ? AND client_id=?",
                                    (f"%{name}%", CLIENT_ID)
                                )
                                conn.commit()
                        except Exception:
                            pass
                        reply = (
                            f"Done! I've cancelled the strategy call for {name}. Feel free to rebook anytime!"
                        )
                    else:
                        reply = f"I couldn't find a strategy call for {name}. Could you double-check the name?"
                except Exception as e:
                    logger.error(f"Cancel error: {e}")
                    reply = "I had trouble accessing the calendar. Please try again."
                booking["cancelling"] = False
                booking["cancellation_name"] = None
            else:
                reply = "Sure! What's the full name the strategy call was booked under?"

    # --- Booking flow ---
    if reply is None:
        extracted = extract_booking_info_with_ai(user_message, booking)

        if not booking["name"] and extracted.get("name"):
            booking["name"] = extracted["name"]
            logger.info(f"Extracted name: {booking['name']}")

        if not booking["business"] and extracted.get("business"):
            booking["business"] = extracted["business"]
            logger.info(f"Extracted business: {booking['business']}")

        if booking["name"]:
            log_lead(session_id, booking["name"], booking.get("business", ""))

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

        confirm_words = ["yes", "yeah", "sure", "ok", "okay", "that works", "sounds good", "perfect", "great", "confirmed", "yep", "yup"]
        if booking["time_suggestion"] and not booking["time_confirmed"]:
            if any(w in user_message.lower() for w in confirm_words):
                booking["time"] = booking["time_suggestion"]
                booking["time_confirmed"] = True

        if booking["name"] and booking["time"] and booking["time_confirmed"] and reply is None:
            try:
                cal_service = get_calendar_service()
                create_strategy_call_event(cal_service, booking["name"], booking["business"], booking["time"])
                log_event("booking_created", session_id)
                log_booking(session_id, booking["name"], booking["business"], booking["time"])
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                            "UPDATE leads SET status='qualified' WHERE session_id=? AND client_id=?",
                            (session_id, CLIENT_ID)
                        )
                        conn.commit()
                except Exception:
                    pass
                time_str = booking["time"].strftime("%A, %B %d at %I:%M %p")
                reply = (
                    f"You're all set, {booking['name']}! 🎉 "
                    f"Your free 30-minute strategy call is booked for {time_str} CT. "
                    f"We'll walk through your business and show exactly how Lumera can work for you. See you then!"
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
