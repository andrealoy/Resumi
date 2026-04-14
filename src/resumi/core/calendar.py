from __future__ import annotations

import os

import dateparser
import pandas as pd


CALENDAR_FILE = "local_calendar.csv"


def calendar_tool(event_details: str, date_time: str) -> str:
    """Enregistre un événement dans le calendrier.
    'date_time' peut être un format ISO ou une date naturelle en français."""
    try:
        dt = dateparser.parse(
            date_time,
            languages=["fr"],
            settings={"PREFER_DATES_FROM": "future"},
        )

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
