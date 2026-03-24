# Polymarket Weather Trading Bot

Autonomous trading bot for Polymarket US city daily high-temperature bucket markets.
Exploits systematic mispricing between high-accuracy NOAA/NWS forecasts (~90% at 5-day,
1-2°F short-term) and retail-heavy market pricing.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ forecast_scanner │────▶│ polymarket_parser │────▶│ decision_engine │────▶│   execution      │
│ (NWS/NOAA API)  │     │ (Gamma API)       │     │ (EV/Kelly/Risk) │     │ (CLOB/Telegram)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
```

**Main loop**: Every 2 minutes → scan forecasts → discover markets → evaluate EV → execute.

## Modules

| Module | File | Role |
|--------|------|------|
| Config | `config.py` | Loads `.env`, typed access to all params |
| Module 1 | `forecast_scanner.py` | NWS API → daily high forecasts → Gaussian bucket probs |
| Module 2 | `polymarket_parser.py` | Gamma API → active temp markets → match to cities/dates |
| Module 3 | `decision_engine.py` | EV calc, Kelly sizing, risk caps, signal ranking |
| Module 4 | `execution.py` | Order placement (py-clob-client), CSV logging, Telegram |
| Main | `main.py` | Async scan loop, graceful shutdown |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your:
#   - Polygon private key (PRIVATE_KEY)
#   - Telegram bot token + chat ID (optional but recommended)
#   - Bankroll and risk parameters
#   - MODE=dry-run (default) or MODE=live
```

### 3. Run in dry-run mode (RECOMMENDED FIRST)
```bash
mkdir -p logs
python main.py
```

Dry-run will:
- Fetch real NWS forecasts and Polymarket prices
- Compute real EV/Kelly signals
- Simulate fills (no real orders)
- Log everything to `logs/trades.csv` and `logs/scans.csv`
- Send Telegram alerts (if configured)

### 4. Review logs
```bash
# Trade decisions
cat logs/trades.csv

# Scan summaries
cat logs/scans.csv

# Full bot log
tail -f logs/bot.log
```

### 5. Go live (only after validating dry-run)
```bash
# In .env:
MODE=live

# Ensure PRIVATE_KEY has USDC on Polygon
python main.py
```

## Edge Thesis

Retail participants in temperature ladder markets use crude heuristics (anchor to
current weather, round-number bias, recency bias). Meanwhile, NOAA/NWS forecasts
achieve ~90% accuracy at 5 days and 1-2°F precision in the final 24-72 hours.

The bot converts forecast distributions into bucket probabilities using a Gaussian
model, compares against market prices, and trades when expected value exceeds
configurable thresholds — capturing the systematic gap between forecast quality
and retail pricing.

## Risk Controls

| Parameter | Default | Research Reference |
|-----------|---------|-------------------|
| Kelly fraction | 0.15 (15% of full Kelly) | §5.3: use 0.1–0.25 fractional |
| Per-market cap | 3% of bankroll | §5.3: 2-5% per market |
| Max position | $10 | §5.4: $5-30 per position |
| Daily loss cap | $50 | §5.4: -3-5% shutdown |
| Min EV threshold | 3% | §5.2: positive EV only |
| Min edge | 8% | §5.2: meaningful edge |
| Daily exposure cap | 30% of bankroll | §5.4: multi-market scaling |
| Instability skip | >4°F swing | Adaptive: skip volatile |

## Deployment (VPS)

```bash
# On Ubuntu VPS:
sudo apt update && sudo apt install python3-pip python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with systemd or screen/tmux:
screen -S polybot
python main.py

# Or create a systemd service:
# /etc/systemd/system/polymarket-bot.service
# [Unit]
# Description=Polymarket Weather Bot
# After=network.target
# [Service]
# User=ubuntu
# WorkingDirectory=/home/ubuntu/polymarket-weather-bot
# ExecStart=/home/ubuntu/polymarket-weather-bot/venv/bin/python main.py
# Restart=always
# [Install]
# WantedBy=multi-user.target
```

## Why This Setup Captures Edge

1. **Data advantage**: NWS forecasts are free, high-quality, and programmatically
   accessible — most retail traders don't systematically consume them.

2. **Speed**: 2-minute scan cycle catches mispricing before manual traders or
   slower bots can react, especially in the critical 24-72h window.

3. **Math**: Gaussian bucket probability model + Kelly sizing + EV thresholds
   ensure only +EV trades are taken at appropriate sizes.

4. **Scale**: 6 cities × 7 days × 10+ buckets = hundreds of opportunities per
   scan cycle, with low correlation between markets.

5. **Risk**: Conservative fractional Kelly + daily caps prevent catastrophic
   drawdowns even with model error.

## License

Private use only. Not financial advice. Use at your own risk.
