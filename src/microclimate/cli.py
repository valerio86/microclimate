"""Command line interface."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import align, store
from .analysis import bias as bias_analysis
from .config import (
    AmbientCredentials,
    ConfigError,
    Location,
    PurpleAirCredentials,
    data_dir,
)
from .sources.ambient import AmbientClient, to_frame
from .sources.openmeteo import OpenMeteoClient
from .sources.purpleair import PurpleAirClient

app = typer.Typer(
    add_completion=False,
    help="Localized weather forecasting from your own sensors.",
)
console = Console()

DEFAULT_START = date(2024, 1, 1)


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


@app.command()
def devices() -> None:
    """List Ambient Weather devices on your account (to find your MAC)."""
    try:
        credentials = AmbientCredentials.from_env(require_mac=False)
    except ConfigError as error:
        _fail(str(error))

    with AmbientClient(credentials) as client:
        found = client.devices()

    if not found:
        console.print("[yellow]No devices found on this account.[/yellow]")
        return

    table = Table(title="Ambient Weather devices")
    table.add_column("MAC")
    table.add_column("Name")
    table.add_column("Location")
    for device in found:
        table.add_row(device.mac, device.name, device.location or "—")
    console.print(table)
    console.print("\nSet [cyan]AMBIENT_MAC[/cyan] in .env to the MAC you want.")


@app.command("backfill-station")
def backfill_station(
    start: datetime = typer.Option(
        DEFAULT_START.isoformat(), formats=["%Y-%m-%d"], help="Earliest date to fetch."
    ),
    resume: bool = typer.Option(
        True, help="Continue from the oldest record already stored."
    ),
) -> None:
    """Backfill station history from Ambient Weather (walks backward in time)."""
    try:
        credentials = AmbientCredentials.from_env()
    except ConfigError as error:
        _fail(str(error))

    start_utc = start.replace(tzinfo=timezone.utc)

    with store.connect() as conn:
        end_utc = datetime.now(timezone.utc)
        if resume:
            earliest = store.earliest_station_ts(conn)
            if earliest is not None:
                end_utc = earliest
                console.print(f"Resuming backward from [cyan]{earliest}[/cyan]")

        if end_utc <= start_utc:
            console.print("[green]Already backfilled to the requested start.[/green]")
            return

        estimated_days = max((end_utc - start_utc).days, 1)
        console.print(
            f"Fetching ~{estimated_days} days at ~1 request/sec "
            f"(≈{estimated_days // 60} min). Safe to interrupt and resume."
        )

        total = 0
        with AmbientClient(credentials) as client:
            with console.status("Fetching…") as status:
                for page in client.iter_history(start=start_utc, end=end_utc):
                    frame = to_frame(page)
                    total += store.upsert(conn, "obs_station", frame)
                    oldest = frame["ts"].min()
                    status.update(f"{total:,} records — back to {oldest:%Y-%m-%d}")

        console.print(f"[green]Stored {total:,} records.[/green]")
        _print_coverage(conn)


@app.command("update-station")
def update_station() -> None:
    """Fetch station readings since the newest one stored."""
    try:
        credentials = AmbientCredentials.from_env()
    except ConfigError as error:
        _fail(str(error))

    with store.connect() as conn:
        _, latest, _ = store.coverage(conn, "obs_station")
        start = latest or (datetime.now(timezone.utc) - timedelta(days=1))

        total = 0
        with AmbientClient(credentials) as client:
            for page in client.iter_history(start=start, end=datetime.now(timezone.utc)):
                total += store.upsert(conn, "obs_station", to_frame(page))

        console.print(f"[green]Added/refreshed {total:,} records.[/green]")


@app.command("backfill-forecast")
def backfill_forecast(
    start: datetime = typer.Option(
        DEFAULT_START.isoformat(), formats=["%Y-%m-%d"], help="Earliest date to fetch."
    ),
    end: datetime = typer.Option(
        None, formats=["%Y-%m-%d"], help="Latest date (default: yesterday)."
    ),
    leads: str = typer.Option(
        "0,1,2,3,5,7", help="Comma-separated forecast lead times in days."
    ),
) -> None:
    """Backfill Open-Meteo forecasts, including what was predicted N days ahead."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    end_date = (end or datetime.now() - timedelta(days=1)).date()
    lead_list = [int(value) for value in leads.split(",") if value.strip()]

    console.print(
        f"Fetching forecasts {start.date()} → {end_date} at leads {lead_list} "
        f"for {location.latitude}, {location.longitude}"
    )

    with store.connect() as conn, OpenMeteoClient(location) as client:
        total = 0
        with console.status("Fetching…") as status:
            for frame in client.fetch_range(start.date(), end_date, leads=lead_list):
                total += store.upsert(conn, "forecasts", frame)
                status.update(
                    f"{total:,} rows — through {frame['valid_time'].max():%Y-%m-%d}"
                )

        console.print(f"[green]Stored {total:,} forecast rows.[/green]")
        _print_coverage(conn)


@app.command("air-update")
def air_update(
    history_days: int = typer.Option(
        0, help="Also pull this many days of history (needs a history-enabled key)."
    ),
) -> None:
    """Record the current PurpleAir reading (and optionally recent history)."""
    try:
        credentials = PurpleAirCredentials.from_env()
    except ConfigError as error:
        _fail(str(error))

    with store.connect() as conn, PurpleAirClient(credentials) as client:
        total = store.upsert(conn, "obs_air", client.current())
        if history_days:
            end = datetime.now(timezone.utc)
            frame = client.history(end - timedelta(days=history_days), end)
            total += store.upsert(conn, "obs_air", frame)
        console.print(f"[green]Stored {total:,} air quality rows.[/green]")


@app.command()
def status() -> None:
    """Show what data is currently stored."""
    with store.connect() as conn:
        _print_coverage(conn)


@app.command()
def bias(
    variable: str = typer.Option("temp_f", help="Variable to analyze."),
    by: str = typer.Option(
        "local_hour", help="Break down by local_hour, local_month, or none."
    ),
    lead: int = typer.Option(1, help="Forecast lead time in days to report."),
) -> None:
    """Report how wrong the public forecast is at your location."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    with store.connect() as conn:
        station = conn.execute("SELECT * FROM obs_station ORDER BY ts").df()
        forecasts = conn.execute("SELECT * FROM forecasts ORDER BY valid_time").df()

    if station.empty or forecasts.empty:
        _fail("Need both station and forecast data — run the backfill commands first.")

    paired = align.pair(align.hourly_station(station), forecasts)
    if paired.empty:
        _fail("No overlapping hours between station and forecast data.")

    paired = align.add_time_features(paired, location.timezone)

    console.print(f"\n[bold]Overall forecast error[/bold] (forecast − actual)")
    _print_frame(bias_analysis.overall(paired))

    if by != "none":
        console.print(f"\n[bold]{variable} error by {by}, lead {lead}d[/bold]")
        breakdown = bias_analysis.by_group(paired, by, variable)
        _print_frame(breakdown[breakdown["lead_days"] == lead])


@app.command()
def export(
    directory: Path = typer.Option(None, help="Destination (default: DATA_DIR/export).")
) -> None:
    """Export tables to Parquet."""
    target = directory or (data_dir() / "export")
    with store.connect() as conn:
        for table in ("obs_station", "obs_air", "forecasts"):
            store.export_parquet(conn, table, target / f"{table}.parquet")
    console.print(f"[green]Exported to {target}[/green]")


def _print_coverage(conn) -> None:
    table = Table(title="Stored data")
    table.add_column("Table")
    table.add_column("From")
    table.add_column("To")
    table.add_column("Rows", justify="right")

    for name, time_column in (
        ("obs_station", "ts"),
        ("obs_air", "ts"),
        ("forecasts", "valid_time"),
    ):
        first, last, count = store.coverage(conn, name, time_column)
        table.add_row(
            name,
            f"{first:%Y-%m-%d}" if first else "—",
            f"{last:%Y-%m-%d}" if last else "—",
            f"{count:,}",
        )
    console.print(table)


def _print_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        console.print("[yellow]No rows.[/yellow]")
        return

    table = Table()
    for column in frame.columns:
        table.add_column(str(column), justify="right")
    for _, row in frame.iterrows():
        table.add_row(
            *[
                f"{value:.2f}" if isinstance(value, float) else f"{value}"
                for value in row
            ]
        )
    console.print(table)


if __name__ == "__main__":
    app()
