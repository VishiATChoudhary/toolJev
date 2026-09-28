"""Three toy MCP servers for the demo. Run one with: python upstreams.py <name>"""

import sys

from fastmcp import FastMCP

TICKETS = [
    {"id": 101, "body": "Checkout returns HTTP 500 for every customer since the deploy, we are losing sales"},
    {"id": 102, "body": "Small typo in the footer of the pricing page"},
    {"id": 103, "body": "EU region dashboards are completely down, nobody can log in"},
    {"id": 104, "body": "Would love a dark mode option"},
    {"id": 105, "body": "I was charged twice for my March invoice"},
    {"id": 106, "body": "API latency doubled since this morning, some requests time out"},
]


def tickets() -> FastMCP:
    s = FastMCP("tickets")
    closed: dict[int, str] = {}

    @s.tool
    def list_tickets(limit: int = 50) -> list[dict]:
        """List open customer support tickets."""
        return [t for t in TICKETS if t["id"] not in closed][:limit]

    @s.tool
    def assign_ticket(ticket_id: int, team: str) -> dict:
        """Assign a support ticket to a team."""
        return {"ticket_id": ticket_id, "team": team}

    @s.tool
    def page_oncall(ticket_id: int, summary: str) -> dict:
        """Page the on-call engineer about an incident."""
        return {"paged": True, "ticket_id": ticket_id}

    return s


def weather() -> FastMCP:
    s = FastMCP("weather")

    @s.tool
    def forecast(city: str, days: int = 3) -> dict:
        """Weather forecast for a city."""
        return {"city": city, "days": [{"day": i, "high_c": 18 + i} for i in range(days)]}

    @s.tool
    def air_quality(city: str) -> dict:
        """Current air quality index for a city."""
        return {"city": city, "aqi": 42}

    return s


def calendar() -> FastMCP:
    s = FastMCP("calendar")

    @s.tool
    def list_events(date: str) -> list[dict]:
        """List calendar events on a date (YYYY-MM-DD)."""
        return [{"title": "Standup", "time": "09:30"}]

    @s.tool
    def create_event(title: str, date: str, time: str) -> dict:
        """Create a calendar event."""
        return {"created": title, "date": date, "time": time}

    return s


if __name__ == "__main__":
    {"tickets": tickets, "weather": weather, "calendar": calendar}[sys.argv[1]]().run(show_banner=False)
