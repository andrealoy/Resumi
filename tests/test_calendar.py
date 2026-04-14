import os
import csv

import pytest

from resumi.core.calendar import calendar_tool, CALENDAR_FILE


def test_calendar_tool_creates_csv_and_writes_row(tmp_path, monkeypatch):
    """Test that calendar_tool writes an event to the CSV with correct format.

    Uses a temporary working directory to avoid polluting the repo.
    """
    # Use tmp_path as cwd so the CALENDAR_FILE is created there
    monkeypatch.chdir(tmp_path)

    event = "Réunion test"
    # Use a natural French date string that dateparser can handle
    dt_str = "15 avril 2026 à 14h30"

    result = calendar_tool(event, dt_str)

    # Expect success message mentioning the event or SUCCÈS
    assert "SUCCÈS" in result or "SUCCES" in result

    # File should exist
    assert os.path.exists(CALENDAR_FILE)

    # Read CSV and verify last row matches the event and date/time
    with open(CALENDAR_FILE, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 1

    # Find any row matching our event
    matched = [r for r in reader if r.get("Événement", r.get("Evenement", "")) == event]
    assert matched, f"No CSV row found with event '{event}'; rows: {reader}"

    # Check date/time format in row
    row = matched[-1]
    # Date is YYYY-MM-DD and Heure is HH:MM
    assert row["Date"] == "2026-04-15"
    assert row["Heure"] == "14:30"


if __name__ == "__main__":
    pytest.main([__file__])
