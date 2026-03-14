from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta

SCOPES = ["https://www.googleapis.com/auth/calendar"]

SERVICE_ACCOUNT_FILE = "service_account.json"

CALENDAR_ID = "36d850b0c67fa12a1f5508645e3d11b2655cb344294a9644ed8644aee9aad637@group.calendar.google.com"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

service = build("calendar", "v3", credentials=credentials)

start_time = datetime.utcnow() + timedelta(minutes=5)
end_time = start_time + timedelta(hours=1)

event = {
    "summary": "Lumera AI Test Booking",
    "start": {
        "dateTime": start_time.isoformat() + "Z",
        "timeZone": "America/Chicago",
    },
    "end": {
        "dateTime": end_time.isoformat() + "Z",
        "timeZone": "America/Chicago",
    },
}

created_event = service.events().insert(
    calendarId=CALENDAR_ID,
    body=event
).execute()

print("Event created:", created_event.get("htmlLink"))