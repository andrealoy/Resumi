"""Google Calendar integration – placeholder."""


class CalendarHandler:
    """Manage Google Calendar events. TODO: implement with Google API."""

    def __init__(self, *, calendar_id: str = "primary") -> None:
        self._calendar_id = calendar_id

    async def create_event(self, title: str, date: str) -> str:
        raise NotImplementedError("Google Calendar integration not yet implemented")
