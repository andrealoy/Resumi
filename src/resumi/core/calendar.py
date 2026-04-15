from __future__ import annotations

import os
import re
from datetime import datetime

import dateparser
import pandas as pd


CALENDAR_FILE = "local_calendar.csv"

_MONTHS_FR = (
    "janvier",
    "février",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
    "decembre",
)
_TIME_ONLY_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def _now() -> datetime:
    """Current time helper kept separate for deterministic tests."""
    return datetime.now()


def _normalize_date_text(text: str) -> str:
    """Normalize French time expressions like '14h' into '14:00'."""
    normalized = " ".join(text.strip().split())
    return re.sub(
        r"\b(\d{1,2})\s*h(?:\s*(\d{1,2}))?\b",
        lambda m: f"{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}",
        normalized,
        flags=re.IGNORECASE,
    )


def _has_explicit_date(text: str) -> bool:
    """Return True if the text clearly contains a day/date reference."""
    low = text.casefold()
    if any(month in low for month in _MONTHS_FR):
        return True
    if any(word in low for word in ("demain", "aujourd", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")):
        return True
    return bool(re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", low))


def _parse_event_datetime(date_time: str) -> datetime | None:
    """Parse user date/time robustly for French calendar inputs."""
    normalized = _normalize_date_text(date_time)
    now = _now()

    if _TIME_ONLY_RE.fullmatch(normalized) and not _has_explicit_date(normalized):
        hour, minute = map(int, normalized.split(":"))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    dt = dateparser.parse(
        normalized,
        languages=["fr"],
        settings={
            "PREFER_DATES_FROM": "future",
            "DATE_ORDER": "DMY",
            "RELATIVE_BASE": now,
        },
    )

    if dt is None:
        return None

    # If the user explicitly gave today's day/month but the requested hour is
    # already in the past, dateparser can jump to next year because of the
    # 'future' preference. Keep the current year in that specific case.
    if dt.year == now.year + 1 and dt.month == now.month and dt.day == now.day:
        dt = dt.replace(year=now.year)

    return dt


def calendar_tool(event_details: str, date_time: str) -> str:
    """Enregistre un événement dans le calendrier.
    'date_time' peut être un format ISO ou une date naturelle en français."""
    try:
        dt = _parse_event_datetime(date_time)

        if dt is None:
            return (
                "ERREUR : Je n'ai pas réussi à identifier la date "
                "à laquelle tu fais référence. Peux-tu être plus précis ? :)"
            )

        if dt.hour == 0 and dt.minute == 0:
            dt = dt.replace(hour=9, minute=0)
            duree = " (09:00 - 10:00)"
        else:
            duree = ""

        new_event = {
            "Date": dt.strftime("%Y-%m-%d"),
            "Heure": dt.strftime("%H:%M"),
            "Événement": event_details,
        }

        # Sauvegarde locale (CSV)
        df = (
            pd.read_csv(CALENDAR_FILE)
            if os.path.exists(CALENDAR_FILE)
            else pd.DataFrame()
        )
        df = pd.concat([df, pd.DataFrame([new_event])], ignore_index=True)
        df = df.sort_values(by=["Date", "Heure"])
        df.to_csv(CALENDAR_FILE, index=False)

        return (
            f"SUCCÈS : '{event_details}' ajouté pour le "
            f"{dt.strftime('%d/%m/%Y à %H:%M')}{duree}."
        )
    except Exception as e:
        return f"ERREUR : {str(e)}"
