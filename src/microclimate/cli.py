"""Command line interface."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import align, features, store
from .analysis import backtest as backtest_analysis
from .analysis import bias as bias_analysis
from .analysis import frost as frost_analysis
from .analysis import rain as rain_analysis
from .analysis import rain_model
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

# Scored against the station over Sep 2025 - Jun 2026, lead 1: ICON 2.33 F MAE,
# ECMWF 2.48, GEM 2.42, GFS 2.82. Open-Meteo's `best_match` resolves to GFS at
# this location, so naming models explicitly is worth more than any correction
# fitted so far. All three are stored; ICON is the default for analysis.
DEFAULT_MODELS = "icon_seamless,ecmwf_ifs025,gfs_seamless"
DEFAULT_SOURCE = "open-meteo:icon_seamless"


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
    models: str = typer.Option(
        DEFAULT_MODELS,
        help="Comma-separated Open-Meteo models. Each is stored separately, so "
        "their disagreement can be used as an uncertainty signal.",
    ),
) -> None:
    """Backfill Open-Meteo forecasts, including what was predicted N days ahead."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    end_date = (end or datetime.now() - timedelta(days=1)).date()
    lead_list = [int(value) for value in leads.split(",") if value.strip()]
    model_list = [value.strip() for value in models.split(",") if value.strip()]

    console.print(
        f"Fetching forecasts {start.date()} → {end_date} at leads {lead_list} "
        f"from {len(model_list)} model(s) for {location.latitude}, {location.longitude}"
    )

    with store.connect() as conn:
        total = 0
        for model in model_list:
            with OpenMeteoClient(location, model=model) as client:
                with console.status(f"Fetching {model}…") as status:
                    for frame in client.fetch_range(
                        start.date(), end_date, leads=lead_list
                    ):
                        total += store.upsert(conn, "forecasts", frame)
                        status.update(
                            f"{model}: {total:,} rows — "
                            f"through {frame['valid_time'].max():%Y-%m-%d}"
                        )
            console.print(f"  [green]✓[/green] {model}")

        console.print(f"[green]Stored {total:,} forecast rows.[/green]")
        _print_sources(conn)


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
    source: str = typer.Option(DEFAULT_SOURCE, help="Forecast source to analyze."),
) -> None:
    """Report how wrong the public forecast is at your location."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    paired = _load_paired(location, source)

    console.print(f"\n[bold]Overall forecast error[/bold] (forecast − actual)")
    _print_frame(bias_analysis.overall(paired))

    if by != "none":
        console.print(f"\n[bold]{variable} error by {by}, lead {lead}d[/bold]")
        breakdown = bias_analysis.by_group(paired, by, variable)
        _print_frame(breakdown[breakdown["lead_days"] == lead])


@app.command()
def backtest(
    variable: str = typer.Option("temp_f", help="Variable to correct."),
    train_end: str = typer.Option(
        None, help="ISO date ending the training period, e.g. 2025-12-31."
    ),
    train_fraction: float = typer.Option(
        0.7, help="Fraction of the time span used for training, if no --train-end."
    ),
    lead: int = typer.Option(None, help="Show only this lead time."),
    source: str = typer.Option(DEFAULT_SOURCE, help="Forecast source to score."),
) -> None:
    """Score corrections on held-out data they were never fitted to."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    paired = _load_paired(location, source)
    # Attach observation history as of each forecast's issue time, then verify
    # the constraint held on this real data rather than only in tests.
    paired = features.add_lagged_observations(paired, variable=variable)
    features.assert_no_lookahead(paired, variable=variable)

    results, info = backtest_analysis.evaluate(
        paired, variable=variable, train_fraction=train_fraction, train_end=train_end
    )

    console.print(
        f"\n[bold]{info['variable']}[/bold] — fitted on "
        f"{info['train_start'][:10]} → {info['train_end'][:10]} "
        f"({info['train_rows']:,} rows), scored on "
        f"{info['test_start'][:10]} → {info['test_end'][:10]} "
        f"({info['test_rows']:,} rows)"
    )
    console.print("[dim]Skill is versus the raw forecast: >0 helped, <0 made it worse.[/dim]\n")

    shown = results if lead is None else results[results["lead_days"].isin([lead, "all"])]
    _print_frame(shown[["method", "lead_days", "n", "bias", "mae", "rmse", "skill"]])


@app.command()
def crossval(
    variable: str = typer.Option("temp_f", help="Variable to correct."),
    folds: int = typer.Option(5, help="Number of walk-forward folds."),
    holdout: float = typer.Option(
        0.2, help="Fraction of the record sealed off and never used for selection."
    ),
    source: str = typer.Option(DEFAULT_SOURCE, help="Forecast source to score."),
) -> None:
    """Compare methods across walk-forward folds, with a sealed final holdout."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    paired = _load_paired(location, source)
    paired = features.add_lagged_observations(paired, variable=variable)
    features.assert_no_lookahead(paired, variable=variable)

    development, sealed = backtest_analysis.seal_holdout(paired, holdout)
    console.print(
        f"\n[bold]{variable}[/bold] — {folds} walk-forward folds over "
        f"{str(development['valid_time'].min())[:10]} → "
        f"{str(development['valid_time'].max())[:10]}"
    )
    console.print(
        f"[dim]Sealed holdout {str(sealed['valid_time'].min())[:10]} → "
        f"{str(sealed['valid_time'].max())[:10]} ({len(sealed):,} rows) — "
        f"not used here, and not to be used until a model is frozen.[/dim]\n"
    )

    results, summary = backtest_analysis.cross_validate(
        development, variable=variable, n_folds=folds
    )
    if results.empty:
        _fail("Not enough data to build folds.")

    console.print("[bold]Skill per fold[/bold] (versus the raw forecast)")
    pivot = results.pivot(index="method", columns="fold", values="skill").reset_index()
    pivot.columns = ["method"] + [f"fold {c}" for c in pivot.columns[1:]]
    _print_frame(pivot)

    console.print("\n[bold]Across folds[/bold]")
    _print_frame(summary.sort_values("skill_mean", ascending=False))
    console.print(
        "[dim]skill_std is the number to watch: a high mean with a wide spread "
        "is not a reliable win.[/dim]"
    )


@app.command("rain-forecast")
def rain_forecast(
    days: int = typer.Option(5, help="How many days ahead to report."),
) -> None:
    """Rain chance for the days ahead, from the frozen three-feature model."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    model_list = [value.strip() for value in DEFAULT_MODELS.split(",") if value.strip()]
    sources = [f"open-meteo:{model}" for model in model_list]

    # Train on everything stored. Model selection is finished, so there is no
    # holdout to protect here — the sealed period has already been spent.
    with store.connect() as conn:
        station = conn.execute("SELECT * FROM obs_station ORDER BY ts").df()
        hourly = align.hourly_station(station)
        paired = {}
        for source in sources:
            forecasts = conn.execute(
                "SELECT * FROM forecasts WHERE source = ? ORDER BY valid_time", [source]
            ).df()
            if forecasts.empty:
                _fail(f"No stored forecasts for {source} — run backfill-forecast.")
            paired[source] = align.add_time_features(
                align.pair(hourly, forecasts), location.timezone
            )

    training = rain_model.daily_features(
        paired, location.timezone, primary=sources[0], lead_days=1
    )
    if training.empty:
        _fail("No usable training days.")

    live = {}
    for model in model_list:
        with OpenMeteoClient(location, model=model) as client:
            live[f"open-meteo:{model}"] = client.fetch_upcoming(days + 1)

    upcoming = rain_model.upcoming_features(live, location.timezone)
    if upcoming.empty:
        _fail("No complete days in the live forecast.")

    # Never show 0% or 100%: the model has ~450 training days and no business
    # claiming certainty about weather, however confident the fit happens to be.
    upcoming["rain_chance"] = rain_model.fit_predict_frozen(training, upcoming).clip(
        0.01, 0.99
    )

    console.print(
        f"\n[bold]Rain chance[/bold] — trained on {len(training):,} days, "
        f"{len(model_list)} models ({', '.join(model_list)})"
    )
    table = Table()
    table.add_column("Day")
    table.add_column("Models wet", justify="right")
    table.add_column("Per-model inches")
    table.add_column("Spread", justify="right")
    table.add_column("Rain chance", justify="right")
    for _, row in upcoming.head(days).iterrows():
        chance = row["rain_chance"]
        colour = "red" if chance >= 0.6 else "yellow" if chance >= 0.3 else "green"
        table.add_row(
            f"{row['day']:%a %d %b}",
            f"{int(row['models_wet'])}/{len(model_list)}",
            row["per_model_in"],
            f"{row['model_spread_in']:.2f}",
            f"[{colour}]{chance:.0%}[/{colour}]",
        )
    console.print(table)
    console.print(
        "[dim]Calibrated at the extremes; under-confident in the middle — a "
        "stated 45% has historically meant nearer 65%. Liquid rain only: the "
        "gauge cannot see snow, so this is not meaningful below freezing.[/dim]"
    )


@app.command("rain-health")
def rain_health(
    lead: int = typer.Option(1, help="Forecast lead time in days."),
    source: str = typer.Option(DEFAULT_SOURCE, help="Forecast source to compare against."),
    all_months: bool = typer.Option(False, help="Show every month, not just problems."),
) -> None:
    """Check the rain gauge for periods when it was blocked.

    A clogged funnel looks exactly like a dry spell, so it is invisible unless
    checked for. Only warm hours count, so snow cannot be mistaken for a fault.
    """
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    paired = _load_paired(location, source)
    at_lead = paired[paired["lead_days"] == lead]
    health = rain_analysis.gauge_health(at_lead, location.timezone)
    if health.empty:
        _fail("Not enough warm-hour data to assess the gauge.")

    problems = health[health["verdict"].isin(["BLOCKED", "suspect"])]
    console.print(
        f"\n[bold]Rain gauge health[/bold] — {len(health)} months assessed, "
        f"{len(problems)} flagged"
    )
    console.print(
        "[dim]Warm hours only (≥38 °F), so frozen precipitation cannot explain "
        "a zero. BLOCKED means the forecast was wet repeatedly and the gauge "
        "never once tipped.[/dim]\n"
    )

    shown = health if all_months else (problems if not problems.empty else health)
    _print_frame(
        shown[
            [
                "month", "hours", "forecast_in", "observed_in",
                "forecast_wet_hours", "observed_wet_hours", "catch_ratio", "verdict",
            ]
        ]
    )

    known = ", ".join(f"{a}→{b}" for a, b in rain_analysis.GAUGE_OUTAGES)
    console.print(f"\n[dim]Excluded from rain analysis: {known}[/dim]")
    unlisted = problems[~rain_analysis.in_outage(problems["month"] + "-15")]
    if not unlisted.empty:
        console.print(
            f"[yellow]{len(unlisted)} flagged month(s) are not in GAUGE_OUTAGES — "
            f"add them to rain.py if the gauge was genuinely blocked.[/yellow]"
        )


@app.command("rain-chance")
def rain_chance(
    lead: int = typer.Option(1, help="Forecast lead time in days."),
    source: str = typer.Option(DEFAULT_SOURCE, help="Forecast source to score."),
    train_end: str = typer.Option(None, help="ISO date ending the training period."),
) -> None:
    """Score daily rain probability, excluding hours the gauge cannot see."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    paired = _load_paired(location, source)
    at_lead = paired[paired["lead_days"] == lead]
    if at_lead.empty:
        _fail(f"No paired data at lead {lead}.")

    daily = rain_analysis.daily_targets(at_lead, location.timezone)
    if daily.empty:
        _fail(
            "No complete days survived the snow filter. "
            "Refetch forecasts so snowfall is available."
        )

    train, test = backtest_analysis.chronological_split(
        daily.rename(columns={"day": "valid_time"}), train_end=train_end
    )
    train = train.rename(columns={"valid_time": "day"})
    test = test.rename(columns={"valid_time": "day"})

    outcome = test["wet"].to_numpy(dtype=float)
    climatology = rain_analysis.climatological_probability(train, test)
    calibrated = rain_analysis.forecast_amount_probability(train, test)

    console.print(
        f"\n[bold]Daily rain chance at lead {lead}d[/bold] — "
        f"{len(train):,} training days, {len(test):,} test days, "
        f"base rate {train['wet'].mean():.0%}"
    )
    console.print(
        "[dim]Snow-affected days are excluded: an unheated gauge records them as "
        "dry, so their labels are wrong rather than merely noisy.[/dim]\n"
    )

    rows = [
        {
            "method": "climatology",
            "brier": rain_analysis.brier_score(climatology, outcome),
            "skill": rain_analysis.brier_skill(climatology, outcome),
        },
        {
            "method": "calibrated forecast",
            "brier": rain_analysis.brier_score(calibrated, outcome),
            "skill": rain_analysis.brier_skill(calibrated, outcome),
        },
    ]
    _print_frame(pd.DataFrame(rows))

    console.print("\n[bold]Reliability of the calibrated forecast[/bold]")
    console.print("[dim]observed_rate should track mean_predicted.[/dim]")
    _print_frame(rain_analysis.reliability(calibrated, outcome, bins=5))


@app.command("frost-skill")
def frost_skill(
    lead: int = typer.Option(1, help="Forecast lead time in days to assess."),
    threshold: float = typer.Option(32.0, help="Temperature defining a frost night."),
    train_end: str = typer.Option(None, help="ISO date ending the training period."),
    source: str = typer.Option(DEFAULT_SOURCE, help="Forecast source to score."),
) -> None:
    """Score the yes/no frost call on held-out nights."""
    try:
        location = Location.from_env()
    except ConfigError as error:
        _fail(str(error))

    paired = _load_paired(location, source)
    at_lead = paired[paired["lead_days"] == lead].dropna(subset=["temp_f_error"])
    if at_lead.empty:
        _fail(f"No paired data at lead {lead}.")

    train, test = backtest_analysis.chronological_split(at_lead, train_end=train_end)
    corrector = backtest_analysis.RegimeCorrector().fit(train, "temp_f_error")

    nights = frost_analysis.nightly_minima(
        test, location.timezone, corrections=corrector.predict(test)
    )
    if nights.empty:
        _fail("No complete nights in the held-out period.")

    summary = frost_analysis.summarize(nights, threshold)
    console.print(
        f"\n[bold]Frost call at lead {lead}d[/bold] — {summary['nights']:,} held-out nights, "
        f"{summary['frost_nights']:,} with frost ({summary['base_rate']:.0%})"
    )
    console.print(
        f"[dim]Overnight minimum error: raw {summary['raw_bias_f']:+.2f} °F bias / "
        f"{summary['raw_mae_f']:.2f} MAE, corrected {summary['corrected_bias_f']:+.2f} / "
        f"{summary['corrected_mae_f']:.2f}[/dim]\n"
    )

    curve = frost_analysis.skill_curve(nights, threshold=threshold)
    _print_frame(
        curve[
            ["warn_at", "forecast", "hits", "misses", "false_alarms", "pod", "far", "csi", "pss"]
        ]
    )
    console.print(
        "[dim]pod = frosts caught; far = warnings that were wrong; "
        "pss = hit rate − false-alarm rate (0 = no skill).[/dim]"
    )


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


def _load_paired(location: Location, source: str | None = None) -> pd.DataFrame:
    """Observations and forecasts joined into one row per (hour, lead).

    Exactly one forecast source must be selected: several models are stored
    side by side, and joining them all at once would silently multiply every
    observation hour by the number of models.
    """
    source = source or DEFAULT_SOURCE
    with store.connect() as conn:
        station = conn.execute("SELECT * FROM obs_station ORDER BY ts").df()
        forecasts = conn.execute(
            "SELECT * FROM forecasts WHERE source = ? ORDER BY valid_time", [source]
        ).df()

    if station.empty:
        _fail("No station data — run backfill-station first.")
    if forecasts.empty:
        available = _available_sources()
        _fail(
            f"No forecasts stored for source '{source}'.\n"
            f"Available: {', '.join(available) if available else '(none)'}\n"
            f"Run backfill-forecast, or pass --source with one of the above."
        )

    paired = align.pair(align.hourly_station(station), forecasts)
    if paired.empty:
        _fail("No overlapping hours between station and forecast data.")

    return align.add_time_features(paired, location.timezone)


def _available_sources() -> list[str]:
    with store.connect() as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source FROM forecasts ORDER BY source"
            ).fetchall()
        ]


def _print_sources(conn) -> None:
    table = Table(title="Forecast sources")
    table.add_column("Source")
    table.add_column("From (UTC)")
    table.add_column("To (UTC)")
    table.add_column("Rows", justify="right")

    rows = conn.execute(
        "SELECT source, min(valid_time), max(valid_time), count(*) "
        "FROM forecasts GROUP BY source ORDER BY source"
    ).fetchall()
    for source, first, last, count in rows:
        table.add_row(
            source,
            f"{first.astimezone(timezone.utc):%Y-%m-%d}" if first else "—",
            f"{last.astimezone(timezone.utc):%Y-%m-%d}" if last else "—",
            f"{count:,}",
        )
    console.print(table)


def _print_coverage(conn) -> None:
    table = Table(title="Stored data")
    table.add_column("Table")
    table.add_column("From (UTC)")
    table.add_column("To (UTC)")
    table.add_column("Rows", justify="right")

    for name, time_column in (
        ("obs_station", "ts"),
        ("obs_air", "ts"),
        ("forecasts", "valid_time"),
    ):
        first, last, count = store.coverage(conn, name, time_column)
        # DuckDB hands TIMESTAMPTZ back in the machine's local zone, so a row
        # stored at 2024-01-01T00:00Z would otherwise print as 2023-12-31 and
        # read like the backfill overshot its start date.
        table.add_row(
            name,
            f"{first.astimezone(timezone.utc):%Y-%m-%d}" if first else "—",
            f"{last.astimezone(timezone.utc):%Y-%m-%d}" if last else "—",
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
