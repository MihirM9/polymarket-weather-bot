# v4 Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 9 optimizations from the 151 Trading Strategies paper and microstructure analysis to improve edge capture, reduce risk, and add performance tracking.

**Architecture:** Four independent workstreams touching different files. Station-specific modeling replaces city-grid NWS forecasts with exact NOAA station lookups. METAR adds a third ensemble source for same-day markets. Decision engine gets seasonal σ, time-decay sizing, correlated exposure caps, and capped Kelly. Execution gets maker/taker fee optimization. Position tracker gets adverse selection detection. Trade logger gets Sharpe tracking.

**Tech Stack:** Python 3.10+, aiohttp, math (no new dependencies)

---

### Task 1: Station-Specific NOAA Modeling

Replace generic NWS grid-point forecasts with station-specific observation points that match Polymarket's resolution sources. The bot currently uses lat/lon grid points (e.g., 40.7789,-73.9692 for New York) which resolve to NWS grid forecasts. But Polymarket markets resolve based on specific NOAA stations (e.g., Central Park for NYC). Airport/park stations can differ 2-5°F from grid forecasts.

**Files:**
- Modify: `config.py` — add `NOAA_STATIONS` config
- Modify: `forecast_scanner.py` — add station observation fetching
- Modify: `ensemble_blender.py` — integrate station data as highest-weight source

**Step 1: Add station config to `config.py`**

Add after `nws_points` parsing (after line 87):

```python
# NOAA observation stations — these are the actual resolution sources
# Format: "City:STATION_ID" (4-letter ICAO codes for METAR-reporting stations)
NOAA_STATIONS_DEFAULT = (
    "New York:KNYC,Chicago:KORD,Los Angeles:KLAX,"
    "Miami:KMIA,Houston:KIAH,Dallas:KDFW"
)
```

Add to `Config.__post_init__`:
```python
raw_stations = os.getenv("NOAA_STATIONS", NOAA_STATIONS_DEFAULT)
self.noaa_stations: Dict[str, str] = {}
for pair in raw_stations.split(","):
    pair = pair.strip()
    if ":" in pair:
        city, station = pair.split(":", 1)
        self.noaa_stations[city.strip()] = station.strip()
```

**Step 2: Add station observation fetcher to `forecast_scanner.py`**

Add new async method to `ForecastScanner` class after `scan_all`:

```python
async def fetch_station_observation(
    self,
    session: aiohttp.ClientSession,
    city: str,
) -> Optional[float]:
    """
    Fetch latest observation from the specific NOAA station.
    Returns current temperature in °F or None.
    Used for same-day markets where real-time obs > forecast.
    """
    station_id = cfg.noaa_stations.get(city)
    if not station_id:
        return None

    url = f"{NWS_BASE}/stations/{station_id}/observations/latest"
    data = await fetch_with_retry(
        session, url, headers=NWS_HEADERS,
        label=f"NWS-obs-{city}"
    )
    if not data:
        return None

    try:
        props = data.get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        if temp_c is None:
            return None
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        logger.info(f"Station obs {station_id} ({city}): {temp_f:.1f}°F")
        return temp_f
    except (KeyError, TypeError) as e:
        logger.debug(f"Station obs parse error for {city}: {e}")
        return None
```

**Step 3: Integrate station obs into ensemble blending in `main.py`**

In `run_scan_cycle`, after OWM supplemental fetch (after line 111), add station observations as a third source. In the blending loop (lines 114-125), if the forecast is for today and we have a station observation, add it as a `ForecastPoint` with weight=1.5 (highest trust — it's actual observed data) and sigma=0.5 (very tight — it's a real measurement, only uncertainty is whether the day's high has been reached yet).

Add to `main.py` after line 111:
```python
# Fetch station observations for same-day forecasts
from datetime import date as date_type
station_obs: Dict[str, float] = {}
async with aiohttp.ClientSession() as session:
    for city in cfg.cities:
        obs = await scanner.fetch_station_observation(session, city)
        if obs is not None:
            station_obs[city] = obs
```

Modify the blending loop (lines 114-125) to include station obs:
```python
for fc in forecasts:
    key = f"{fc.city}_{fc.forecast_date.isoformat()}"
    owm_point = owm_forecasts.get(key)
    supplemental = [owm_point] if owm_point else []

    # Add station observation for same-day markets (highest weight)
    if fc.forecast_date == date_type.today() and fc.city in station_obs:
        from ensemble_blender import ForecastPoint
        station_point = ForecastPoint(
            source="station_obs",
            high_f=station_obs[fc.city],
            sigma=0.5,  # very tight — real measurement
            weight=1.5,  # highest trust
        )
        supplemental.append(station_point)

    ensemble = blender.blend(fc.high_f, fc.sigma, supplemental)
    fc.high_f = ensemble.blended_high
    fc.sigma = ensemble.ensemble_sigma
    fc.confidence = compute_confidence(fc.sigma, fc.is_stable, fc.regime_multiplier)
```

**Step 4: Add NOAA_STATIONS to `.env.example`**

```
# NOAA observation stations (ICAO codes) — resolution source stations
# These should match the exact stations Polymarket uses for resolution
NOAA_STATIONS=New York:KNYC,Chicago:KORD,Los Angeles:KLAX,Miami:KMIA,Houston:KIAH,Dallas:KDFW
```

**Step 5: Commit**
```bash
git add config.py forecast_scanner.py main.py .env.example
git commit -m "feat: add station-specific NOAA modeling for resolution source accuracy"
```

---

### Task 2: Maker/Taker Fee Optimization

The bot currently places limit orders at `price_limit = min(price_yes + 0.02, p_true - 0.02)` — effectively crossing the spread as a taker. On Polymarket, makers pay 0% fees while takers pay ~1-2%. By placing passive limit orders just inside the best bid/ask instead of crossing, we save the fee on every trade.

**Files:**
- Modify: `decision_engine.py:257,290` — change price_limit calculation
- Modify: `execution.py:197-237` — use orderbook data for smart pricing
- Modify: `config.py` — add `MAKER_SPREAD_OFFSET` config

**Step 1: Add maker config to `config.py`**

Add after `fee_rate` line (line 36):
```python
maker_spread_offset: float = float(os.getenv("MAKER_SPREAD_OFFSET", "0.005"))
maker_fee_rate: float = float(os.getenv("MAKER_FEE_RATE", "0.0"))
```

**Step 2: Add orderbook-aware pricing to `execution.py`**

Add method to `OrderExecutor` class after `_check_orderbook_depth`:

```python
def _get_maker_price(self, signal: TradeSignal) -> Optional[float]:
    """
    Compute a passive maker price from the orderbook.
    Instead of crossing the spread (taker), place just inside the best level.
    Returns adjusted price or None to use default.

    For BUY: place at best_bid + offset (sit at top of bid queue)
    For SELL: place at best_ask - offset (sit at top of ask queue)
    """
    if not self.client or self.dry_run:
        return None

    try:
        book = self.client.get_order_book(signal.token_id)
        if not book:
            return None

        offset = cfg.maker_spread_offset

        if signal.side == "BUY":
            bids = book.get("bids", [])
            if bids:
                best_bid = float(bids[0].get("price", 0))
                maker_price = best_bid + offset
                # Don't exceed our max willingness to pay
                return min(maker_price, signal.price_limit)
        else:
            asks = book.get("asks", [])
            if asks:
                best_ask = float(asks[0].get("price", 0))
                maker_price = best_ask - offset
                # Don't go below our min acceptable price
                return max(maker_price, signal.price_limit)

    except Exception as e:
        logger.debug(f"Maker price lookup failed: {e}")

    return None
```

**Step 3: Use maker pricing in `execute_signal`**

In `execution.py`, modify `execute_signal` (around line 275) to try maker pricing first:

```python
# Try maker pricing first (v4: fee optimization)
maker_price = self._get_maker_price(signal)
price = maker_price if maker_price is not None else signal.price_limit
```

**Step 4: Update EV calculations to use maker fee when placing passive orders**

In `decision_engine.py`, update the fee parameter passed to EV functions. Since we're now placing as maker (0% fee), the EV improves. Change lines 237 and 269:

Replace the EV calculation blocks to use maker fee rate:
```python
# Use maker fee rate since we place passive limit orders (v4)
maker_fee = cfg.maker_fee_rate
ev_y = _ev_yes(p_true, price_yes, fee=maker_fee)
```

And similarly for the No side:
```python
ev_n = _ev_no(p_true, price_yes, fee=maker_fee)
```

**Step 5: Add to `.env.example`**

```
# Maker/taker optimization (v4)
MAKER_SPREAD_OFFSET=0.005
MAKER_FEE_RATE=0.0
```

**Step 6: Commit**
```bash
git add decision_engine.py execution.py config.py .env.example
git commit -m "feat: maker/taker fee optimization — passive limit orders save 1-2% per trade"
```

---

### Task 3: Seasonal Sigma Adjustment

Temperature forecast uncertainty varies significantly by season. Spring (frontal passages, rapid warming) and fall (transition weather) have 50-100% more variance than mid-summer (stable high pressure) or mid-winter (stable arctic patterns). The current flat sigma-by-horizon table doesn't capture this.

Based on §14.3 of the paper (CDD/HDD variance analysis): seasonal standard deviation of daily highs is ~4-6°F in spring/fall vs ~2-3°F in summer.

**Files:**
- Modify: `forecast_scanner.py` — add seasonal multiplier to `_sigma_for_horizon`

**Step 1: Add seasonal sigma multiplier table**

Add after `SIGMA_BY_HORIZON` (after line 43):

```python
# Seasonal sigma multiplier — accounts for systematic forecast variance
# by time of year. Spring/fall transitions have much wider forecast error
# than mid-summer/winter stable patterns. (Paper §14.3: CDD/HDD variance)
SEASONAL_SIGMA_MULT: Dict[int, float] = {
    1: 1.1,   # Jan: winter — moderately stable, occasional arctic blasts
    2: 1.15,  # Feb: late winter — warming signals start, moderate variability
    3: 1.3,   # Mar: early spring — highly variable, frontal passages
    4: 1.35,  # Apr: peak spring — maximum forecast uncertainty
    5: 1.2,   # May: late spring — settling but still variable
    6: 1.0,   # Jun: early summer — increasingly stable
    7: 0.9,   # Jul: mid-summer — most stable, high pressure dominant
    8: 0.9,   # Aug: mid-summer — same
    9: 1.05,  # Sep: early fall — beginning transition
    10: 1.2,  # Oct: fall — increasing variability
    11: 1.25, # Nov: late fall — volatile transition
    12: 1.1,  # Dec: winter — moderately stable
}
```

**Step 2: Apply seasonal multiplier in `_sigma_for_horizon`**

Modify `_sigma_for_horizon` to accept an optional date and apply the seasonal multiplier:

```python
def _sigma_for_horizon(hours_out: float, forecast_date: Optional[date] = None) -> float:
    """Return forecast uncertainty σ (°F) given hours until resolution.
    Applies seasonal adjustment if forecast_date is provided."""
    for threshold, sigma in SIGMA_BY_HORIZON:
        if hours_out <= threshold:
            base = sigma
            break
    else:
        base = 6.0  # beyond 7 days

    # Apply seasonal multiplier (v4: Paper §14.3)
    if forecast_date is not None:
        month = forecast_date.month
        base *= SEASONAL_SIGMA_MULT.get(month, 1.0)

    return base
```

**Step 3: Update callers to pass forecast_date**

In `forecast_scanner.py` line 247, change:
```python
sigma_base = _sigma_for_horizon(hours_out, forecast_date)
```

In `ensemble_blender.py` line 193, change:
```python
sigma = _sigma_for_horizon(hours_out, target_date) * 1.15
```

**Step 4: Commit**
```bash
git add forecast_scanner.py ensemble_blender.py
git commit -m "feat: seasonal sigma adjustment — spring/fall get wider uncertainty bands"
```

---

### Task 4: Time-Decay Capital Allocation

A market resolving tomorrow ties up capital for ~24h. A market resolving in 5 days ties it up 5x longer. The bot should demand proportionally higher EV for longer-duration trades to optimize capital velocity. This is the "time value of capital" concept from the paper's carry trade analysis (§8.2).

**Files:**
- Modify: `decision_engine.py` — adjust EV threshold by days-to-resolution

**Step 1: Add time-decay method to `DecisionEngine`**

Add after `_dynamic_edge_threshold` (after line 178):

```python
@staticmethod
def _time_decay_ev_threshold(base_threshold: float, days_to_resolution: int) -> float:
    """
    Scale minimum EV threshold by time capital is locked.

    A 1-day trade at 3% EV is equivalent to a 5-day trade at 15% EV
    in terms of annualized return on capital.

    We use sqrt scaling (not linear) to avoid being too harsh on
    2-3 day trades which are the sweet spot for weather forecasting accuracy.

    Formula: threshold * sqrt(days) where days >= 1

    Examples at base=0.03:
      1 day  → 0.030 (unchanged)
      2 days → 0.042
      3 days → 0.052
      5 days → 0.067
      7 days → 0.079
    """
    days = max(1, days_to_resolution)
    return base_threshold * math.sqrt(days)
```

**Step 2: Apply time-decay in `evaluate` method**

Add `import math` at top of decision_engine.py if not already present.

In the `evaluate` method, after computing `dynamic_edge` (line 222), add:

```python
# Time-decay: demand higher EV for longer-duration trades (v4)
days_to_res = max(1, (mkt.market_date - date.today()).days)
time_adj_ev_threshold = self._time_decay_ev_threshold(
    cfg.min_ev_threshold, days_to_res
)
```

Then change the EV threshold checks on lines 241 and 273 from:
```python
if ev_y > cfg.min_ev_threshold and edge_y > dynamic_edge:
```
to:
```python
if ev_y > time_adj_ev_threshold and edge_y > dynamic_edge:
```

And similarly for the No side (line 273).

**Step 3: Add days_to_resolution to TradeSignal rationale**

In the rationale strings (lines 258-264 and 291-297), add `days_out={days_to_res}` to the rationale for observability.

**Step 4: Commit**
```bash
git add decision_engine.py
git commit -m "feat: time-decay capital allocation — higher EV required for longer-duration trades"
```

---

### Task 5: METAR as Third Ensemble Source

METAR (Meteorological Aerodrome Report) data from airports updates every 30-60 minutes — much faster than NWS grid forecasts (every 6-12 hours). For same-day markets, METAR-derived temperature trends give a significant speed edge.

**Files:**
- Create: `metar_fetcher.py` — METAR parsing and temperature extraction
- Modify: `ensemble_blender.py` — integrate METAR as third ensemble source
- Modify: `main.py` — add METAR fetch to scan cycle

**Step 1: Create `metar_fetcher.py`**

```python
"""
metar_fetcher.py — METAR Aviation Weather Data Fetcher
=======================================================
Fetches real-time METAR observations from aviationweather.gov.
Updates every 30-60 min — much faster than NWS grid forecasts.

For same-day markets, METAR current temperature + trend extrapolation
provides a speed edge over traders waiting for NWS model updates.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Dict, Optional

import aiohttp

from api_utils import fetch_with_retry
from config import cfg

logger = logging.getLogger(__name__)

METAR_BASE = "https://aviationweather.gov/api/data/metar"


@dataclass
class MetarObservation:
    """Parsed METAR observation for a station."""
    station: str
    observed_at: datetime
    temp_f: float
    dewpoint_f: float
    wind_speed_kt: int
    raw_metar: str


def _parse_metar_temp(raw: str) -> Optional[float]:
    """
    Extract temperature from raw METAR string.
    Format: T followed by temp/dewpoint like 'T0272/0189'
    where 0272 = 27.2°C, or 'M' prefix for negative.
    Falls back to the simple temp field like '27/19'.
    """
    # Try precise T-group first: T0272/0189
    m = re.search(r'\bT(\d{4})/(\d{4})\b', raw)
    if m:
        temp_tenths = int(m.group(1))
        temp_c = temp_tenths / 10.0
        if temp_tenths >= 5000:  # negative: 1-bit sign encoding
            temp_c = -(temp_tenths - 5000) / 10.0
        return temp_c * 9.0 / 5.0 + 32.0

    # Fallback: simple temp/dewpoint field like "27/19" or "M03/M07"
    m = re.search(r'\b(M?\d{2})/(M?\d{2})\b', raw)
    if m:
        temp_str = m.group(1)
        temp_c = float(temp_str.replace('M', '-'))
        return temp_c * 9.0 / 5.0 + 32.0

    return None


class MetarFetcher:
    """Fetches and parses METAR observations for configured stations."""

    def __init__(self):
        self._cache: Dict[str, MetarObservation] = {}  # station → latest obs

    async def fetch_observation(
        self,
        session: aiohttp.ClientSession,
        city: str,
    ) -> Optional[MetarObservation]:
        """Fetch latest METAR for a city's configured station."""
        station = cfg.noaa_stations.get(city)
        if not station:
            return None

        url = f"{METAR_BASE}?ids={station}&format=raw&taf=false&hours=1"
        data = await fetch_with_retry(
            session, url, label=f"METAR-{city}", timeout_sec=10.0
        )

        # aviationweather.gov returns plain text, not JSON
        # We need to handle this differently
        if data is None:
            # Try raw text fetch
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return None
                    raw_text = await resp.text()
            except Exception as e:
                logger.debug(f"METAR fetch failed for {city}: {e}")
                return None
        else:
            raw_text = str(data)

        if not raw_text or not raw_text.strip():
            return None

        # Parse the first METAR line
        raw_metar = raw_text.strip().split('\n')[0]
        temp_f = _parse_metar_temp(raw_metar)
        if temp_f is None:
            logger.debug(f"Could not parse METAR temp for {city}: {raw_metar[:80]}")
            return None

        obs = MetarObservation(
            station=station,
            observed_at=datetime.now(timezone.utc),
            temp_f=temp_f,
            dewpoint_f=0.0,  # not critical for our use
            wind_speed_kt=0,
            raw_metar=raw_metar,
        )

        self._cache[city] = obs
        logger.info(f"METAR {station} ({city}): {temp_f:.1f}°F")
        return obs

    async def fetch_all(self) -> Dict[str, MetarObservation]:
        """Fetch METAR for all configured cities."""
        results: Dict[str, MetarObservation] = {}
        async with aiohttp.ClientSession() as session:
            for city in cfg.cities:
                obs = await self.fetch_observation(session, city)
                if obs:
                    results[city] = obs
        return results
```

**Step 2: Integrate METAR into `main.py` scan cycle**

Add import at top of `main.py`:
```python
from metar_fetcher import MetarFetcher
```

Add `metar = MetarFetcher()` in the initialization block (after line 183).

In `run_scan_cycle`, after OWM fetch, add METAR fetch for same-day:
```python
# Fetch METAR observations for same-day markets (v4: speed edge)
metar_obs = await metar.fetch_all()
```

Modify the blending loop to include METAR as a supplemental source:
```python
# Add METAR observation for same-day markets
if fc.forecast_date == date_type.today() and fc.city in metar_obs:
    from ensemble_blender import ForecastPoint
    obs = metar_obs[fc.city]
    metar_point = ForecastPoint(
        source="metar",
        high_f=obs.temp_f,
        sigma=0.5,  # very tight — direct measurement
        weight=1.5,  # highest trust — real observed data
    )
    supplemental.append(metar_point)
```

Note: This overlaps with the station observation in Task 1. In practice, METAR subsumes the station obs fetch — use METAR as the primary real-time source and remove the duplicate `fetch_station_observation` from Task 1. The station config from Task 1 is still needed for the ICAO codes.

**Step 3: Commit**
```bash
git add metar_fetcher.py main.py ensemble_blender.py
git commit -m "feat: add METAR aviation weather as third ensemble source for same-day edge"
```

---

### Task 6: Correlated Exposure Caps

The current bot treats each city independently for exposure caps. But NYC and Chicago temps are correlated (both Northeast/Midwest, same weather systems). If both show edge, the bot might take max positions in both — doubling effective exposure to the same weather pattern. From the paper §3.18 (covariance-based portfolio optimization), we should account for inter-city correlation.

**Files:**
- Modify: `decision_engine.py` — add correlation-aware exposure tracking

**Step 1: Add city correlation groups**

Add after the imports in `decision_engine.py`:

```python
# City correlation groups — cities in the same group share weather systems
# and their forecast errors are correlated. Exposure within a group is
# capped at 1.5x single-city cap (not 2x) to prevent overconcentration.
# Ref: Paper §3.18 (covariance-based portfolio construction)
CORRELATION_GROUPS = {
    "northeast": ["New York", "Chicago"],  # same frontal systems
    "gulf": ["Houston", "Miami"],          # Gulf moisture patterns
    "south_central": ["Dallas", "Houston"],  # overlapping heat patterns
    "west": ["Los Angeles"],               # independent (Pacific coast)
}

# Max exposure multiplier for correlated group
# (1.5x means two correlated cities get 1.5x budget of one, not 2x)
CORRELATED_GROUP_CAP_MULT = 1.5
```

**Step 2: Add group exposure tracking to `DecisionEngine.__init__`**

Add to `__init__` (line 117):
```python
self._group_exposure: Dict[str, float] = {}  # group_name → current exposure
```

Reset it in `_reset_daily_if_needed`:
```python
self._group_exposure = {}
```

**Step 3: Add correlation check to `_size_position`**

Add method to `DecisionEngine`:
```python
def _get_city_groups(self, city: str) -> List[str]:
    """Return all correlation groups this city belongs to."""
    return [g for g, cities in CORRELATION_GROUPS.items() if city in cities]

def _check_group_exposure(self, city: str, proposed_size: float) -> float:
    """
    Cap position size if the city's correlation group is near its limit.
    Returns the adjusted (possibly reduced) size.
    """
    groups = self._get_city_groups(city)
    if not groups:
        return proposed_size

    single_cap = cfg.per_market_max_pct * cfg.bankroll
    group_cap = single_cap * CORRELATED_GROUP_CAP_MULT

    for group in groups:
        current = self._group_exposure.get(group, 0.0)
        remaining = group_cap - current
        proposed_size = min(proposed_size, max(0.0, remaining))

    return proposed_size
```

**Step 4: Wire into `_size_position`**

In `_size_position`, after the `remaining_budget` cap (line 328) and before the dust floor check (line 331), add:

```python
# Correlated exposure cap (v4: Paper §3.18)
raw_size = self._check_group_exposure(city, raw_size)
```

Note: `_size_position` currently doesn't take `city` as a parameter. Add it:

Change signature from:
```python
def _size_position(self, kelly_frac: float, tracker=None) -> float:
```
to:
```python
def _size_position(self, kelly_frac: float, tracker=None, city: str = "") -> float:
```

And update the two call sites in `evaluate` (lines 242 and 274):
```python
size = self._size_position(kelly_y, tracker, city=mkt.city)
```

After sizing succeeds (before return), update group exposure:
```python
# Track group exposure
for group in self._get_city_groups(city):
    self._group_exposure[group] = self._group_exposure.get(group, 0.0) + raw_size
```

**Step 5: Commit**
```bash
git add decision_engine.py
git commit -m "feat: correlated exposure caps — prevent overconcentration in same weather systems"
```

---

### Task 7: Adverse Selection Detection

From the paper §3.19 (market-making): if our limit orders fill instantly, it likely means informed traders are hitting us because they know the price is wrong (adverse selection). If orders never fill, we're too conservative. Track fill speed as a signal.

**Files:**
- Modify: `position_tracker.py` — add fill speed tracking and adverse selection metric
- Modify: `execution.py` — log fill speed in trade log

**Step 1: Add fill speed tracking to `PositionTracker`**

Add to `PositionTracker.__init__` (after line 102):

```python
# Adverse selection tracking (Paper §3.19)
self._fill_speeds: List[float] = []  # seconds from submission to fill
self._instant_fill_count: int = 0    # fills < 10 seconds (suspicious)
self._total_fills: int = 0
```

Reset these in `_reset_daily_if_needed`:
```python
self._fill_speeds = []
self._instant_fill_count = 0
self._total_fills = 0
```

**Step 2: Track fill speed in `poll_fills`**

In `poll_fills`, when a fill is detected (around line 210, after `order.status = OrderStatus.FILLED`), add:

```python
fill_time = order.age_seconds
self._fill_speeds.append(fill_time)
self._total_fills += 1
if fill_time < 10:
    self._instant_fill_count += 1
    logger.warning(
        f"Instant fill detected ({fill_time:.1f}s) for {order.outcome_label} — "
        f"possible adverse selection"
    )
```

**Step 3: Add adverse selection metric property**

Add to `PositionTracker`:
```python
@property
def adverse_selection_rate(self) -> float:
    """Fraction of fills that happened suspiciously fast (<10s).
    High rate (>50%) suggests informed traders are picking us off."""
    if self._total_fills == 0:
        return 0.0
    return self._instant_fill_count / self._total_fills

@property
def avg_fill_speed(self) -> float:
    """Average time to fill in seconds."""
    if not self._fill_speeds:
        return 0.0
    return sum(self._fill_speeds) / len(self._fill_speeds)
```

**Step 4: Add to exposure summary**

Modify `get_exposure_summary` to include fill metrics:
```python
def get_exposure_summary(self) -> str:
    adv_sel = f", adv_sel_rate={self.adverse_selection_rate:.0%}" if self._total_fills > 0 else ""
    fill_spd = f", avg_fill={self.avg_fill_speed:.0f}s" if self._fill_speeds else ""
    return (
        f"Exposure: realized=${self.realized_exposure:.2f}, "
        f"pending=${self.pending_exposure:.2f}, "
        f"total=${self.total_exposure:.2f}, "
        f"active_orders={self.active_order_count}, "
        f"filled={self.filled_order_count}"
        f"{fill_spd}{adv_sel}"
    )
```

**Step 5: Commit**
```bash
git add position_tracker.py
git commit -m "feat: adverse selection detection — track fill speed to detect informed counter-trading"
```

---

### Task 8: Sharpe Ratio Tracking

The paper's backtesting appendix emphasizes Sharpe ratio as the key performance metric. Currently the bot tracks PnL but not risk-adjusted returns. Adding rolling Sharpe lets you compare the bot's performance against simply holding USDC.

**Files:**
- Modify: `execution.py` — add Sharpe tracking to `TradeLogger`
- Modify: `main.py` — compute and log daily Sharpe

**Step 1: Add performance tracker to `execution.py`**

Add new class after `TradeLogger`:

```python
class PerformanceTracker:
    """
    Tracks risk-adjusted performance metrics.
    Ref: Paper Appendix A — out-of-sample Sharpe ratio calculation.
    """

    def __init__(self):
        self._daily_returns: List[float] = []  # daily PnL as % of bankroll
        self._trade_returns: List[float] = []  # per-trade returns

    def record_daily_pnl(self, pnl: float, bankroll: float):
        """Record end-of-day PnL as a return percentage."""
        if bankroll > 0:
            self._daily_returns.append(pnl / bankroll)

    def record_trade_return(self, pnl: float, size: float):
        """Record individual trade return."""
        if size > 0:
            self._trade_returns.append(pnl / size)

    @property
    def sharpe_ratio(self) -> float:
        """
        Annualized Sharpe ratio from daily returns.
        Sharpe = (mean_daily_return / std_daily_return) * sqrt(365)
        """
        if len(self._daily_returns) < 2:
            return 0.0
        mean_r = sum(self._daily_returns) / len(self._daily_returns)
        variance = sum((r - mean_r) ** 2 for r in self._daily_returns) / (len(self._daily_returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.001
        return (mean_r / std_r) * math.sqrt(365)

    @property
    def win_rate(self) -> float:
        """Fraction of positive trades."""
        if not self._trade_returns:
            return 0.0
        return sum(1 for r in self._trade_returns if r > 0) / len(self._trade_returns)

    @property
    def avg_return(self) -> float:
        """Average per-trade return."""
        if not self._trade_returns:
            return 0.0
        return sum(self._trade_returns) / len(self._trade_returns)

    def get_summary(self) -> str:
        return (
            f"Sharpe={self.sharpe_ratio:.2f}, "
            f"WinRate={self.win_rate:.0%}, "
            f"AvgReturn={self.avg_return:.1%}, "
            f"Days={len(self._daily_returns)}, "
            f"Trades={len(self._trade_returns)}"
        )
```

**Step 2: Add `import math` to `execution.py`**

**Step 3: Integrate into `main.py`**

Add import:
```python
from execution import PerformanceTracker
```

Initialize in `main()`:
```python
perf_tracker = PerformanceTracker()
```

Pass to `run_scan_cycle` and add signature parameter. At end of each day (or in the summary cycle), call:
```python
perf_tracker.record_daily_pnl(engine.daily_pnl, cfg.bankroll)
```

Include in Telegram summary:
```python
f"Performance: {perf_tracker.get_summary()}\n"
```

**Step 4: Commit**
```bash
git add execution.py main.py
git commit -m "feat: add Sharpe ratio and performance tracking"
```

---

### Task 9: Capped Adaptive Kelly

The current Kelly implementation can go up to 1.5x base fraction at high confidence. The paper's references on fractional Kelly (Kelly, 1956; Thorp, 2006) recommend hard-capping the fraction to prevent extreme sizing even when the model is very confident. Add an absolute Kelly cap.

**Files:**
- Modify: `decision_engine.py` — add Kelly cap
- Modify: `config.py` — add `MAX_KELLY_MULT` config

**Step 1: Add config**

In `config.py`, add after `kelly_fraction`:
```python
max_kelly_mult: float = float(os.getenv("MAX_KELLY_MULT", "1.25"))
```

**Step 2: Cap the adaptive Kelly**

In `decision_engine.py`, modify `_adaptive_kelly_fraction`:

```python
@staticmethod
def _adaptive_kelly_fraction(base_fraction: float, confidence: float) -> float:
    """
    Scale Kelly fraction by forecast confidence, with absolute cap.

    v4: Added MAX_KELLY_MULT cap to prevent overconfident sizing.
    Even at confidence=1.0, we cap at 1.25x base (was 1.50x).
    Ref: Paper — Kelly (1956), Thorp (2006) on fractional Kelly.
    """
    scaling = 0.25 + 1.25 * confidence  # range: [0.25, 1.50]
    scaling = min(scaling, cfg.max_kelly_mult)  # hard cap
    return base_fraction * scaling
```

**Step 3: Add to `.env.example`**

```
# Maximum Kelly multiplier (v4: hard cap on position sizing confidence)
MAX_KELLY_MULT=1.25
```

**Step 4: Commit**
```bash
git add decision_engine.py config.py .env.example
git commit -m "feat: capped adaptive Kelly — hard cap prevents overconfident sizing"
```

---

## Dependency Graph

```
Task 1 (stations) ──┐
                     ├──► Task 5 (METAR) uses station config from Task 1
Task 3 (seasonal σ) │    (merge station obs + METAR into one flow)
                     │
Task 2 (maker/taker) ──── independent
Task 4 (time-decay)  ──── independent
Task 6 (corr. caps)  ──── independent
Task 7 (adv. select) ──── independent
Task 8 (Sharpe)       ──── independent
Task 9 (Kelly cap)    ──── independent
```

**Parallelization plan:**
- **Group A** (forecast pipeline): Tasks 1 + 3 + 5 → all touch forecast_scanner/ensemble_blender
- **Group B** (decision engine): Tasks 4 + 6 + 9 → all touch decision_engine.py
- **Group C** (execution): Task 2 → touches execution.py + decision_engine.py (fee part only)
- **Group D** (tracking): Tasks 7 + 8 → touch position_tracker.py + execution.py

Execute Groups A-D in parallel, then integrate and resolve any merge conflicts.
