# Polymarket Weather Bot — Complete Deployment Guide

From zero to running bot. Every step, in order.

---

## Phase 1: Get a Server (VPS)

You need a machine that runs 24/7. Your laptop won't work — it sleeps, disconnects, etc.

### Option A: DigitalOcean (Recommended for simplicity)

1. Go to **digitalocean.com**, sign up (GitHub student pack gives $200 credit).
2. Click **Create → Droplet**.
3. Choose:
   - **Image**: Ubuntu 24.04 LTS
   - **Plan**: Basic → Regular → $6/mo (1 GB RAM, 1 vCPU) — more than enough
   - **Region**: New York (lowest latency to Polymarket/Polygon nodes)
   - **Authentication**: SSH key (recommended) or password
4. Click **Create Droplet**. You'll get an IP address like `164.90.xxx.xxx`.
5. SSH in:
   ```bash
   ssh root@164.90.xxx.xxx
   ```

### Option B: AWS EC2

1. Go to **aws.amazon.com** → EC2 → Launch Instance.
2. Choose **Ubuntu 24.04**, **t3.micro** (free tier eligible).
3. Create/select a key pair (.pem file).
4. Launch, then SSH:
   ```bash
   chmod 400 your-key.pem
   ssh -i your-key.pem ubuntu@<public-ip>
   ```

### Option C: Run locally first (for testing only)

Skip the VPS for now if you just want to validate dry-run on your own machine.
You'll need Python 3.10+ installed.

---

## Phase 2: Server Setup

Once SSHed into your VPS (or on your local machine):

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv git -y

# Create a non-root user (on VPS)
adduser botuser
usermod -aG sudo botuser
su - botuser

# Create project directory
mkdir -p ~/polymarket-weather-bot
cd ~/polymarket-weather-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

---

## Phase 3: Get a Polygon Wallet & Private Key

The bot needs a Polygon (MATIC) wallet with USDC to place trades on Polymarket's CLOB.

### Step 1: Create wallet

**Option A — MetaMask (easiest)**:
1. Install MetaMask browser extension (metamask.io).
2. Create new wallet → save your seed phrase securely (write it down, never screenshot).
3. Click network dropdown → Add Network → add Polygon:
   - Network Name: `Polygon Mainnet`
   - RPC URL: `https://polygon-rpc.com`
   - Chain ID: `137`
   - Currency Symbol: `MATIC`
   - Explorer: `https://polygonscan.com`

**Option B — Generate via command line** (more advanced):
```bash
pip install eth-account
python3 -c "
from eth_account import Account
acct = Account.create()
print(f'Address:     {acct.address}')
print(f'Private Key: {acct.key.hex()}')
print('SAVE BOTH. The private key controls all funds.')
"
```

### Step 2: Export your private key

From MetaMask:
1. Click the three dots → Account Details → Show Private Key.
2. Enter your password.
3. Copy the key (starts with `0x...`). **This is what goes in your `.env` file.**

> **SECURITY**: Your private key controls all funds in the wallet. Never share it.
> Never commit it to git. Never paste it in Discord/Telegram.

### Step 3: Fund the wallet

You need USDC on Polygon (this is what Polymarket uses for trading).

**Method A — Bridge from Ethereum/other chain**:
1. Buy USDC on Coinbase/Kraken.
2. Withdraw USDC to your wallet address **on Polygon network** (most exchanges now support direct Polygon withdrawals — look for "Polygon" or "MATIC" network option).

**Method B — Buy on Polygon directly**:
1. Use an onramp like **Transak** or **MoonPay** (accessible through MetaMask's "Buy" button).
2. Buy USDC on Polygon.

**Method C — Swap**:
1. Send MATIC to your wallet.
2. Go to **quickswap.exchange** or **uniswap** on Polygon.
3. Swap MATIC → USDC.

You also need a tiny amount of MATIC for gas fees (~0.1 MATIC, costs <$0.10).

**Recommended starting amount**: $100–500 USDC for dry-run validation, then scale up.

### Step 4: Enable Polymarket CLOB trading

Before the bot can place orders, your wallet needs to approve the Polymarket contracts:

1. Go to **polymarket.com**, connect your wallet (MetaMask).
2. Deposit USDC into Polymarket (this approves the contract interactions).
3. You can withdraw it back — the point is to initialize the on-chain approvals.

Alternatively, `py-clob-client` handles approvals programmatically, but doing it
once via the UI is simpler.

---

## Phase 4: Set Up Telegram Alerts

This is optional but strongly recommended — you'll get real-time trade notifications on your phone.

### Step 1: Create a Telegram bot

1. Open Telegram, search for **@BotFather**.
2. Send `/newbot`.
3. Follow prompts: give it a name (e.g., "Polymarket Weather Bot") and username (e.g., `polyweather_bot`).
4. BotFather gives you an **API token** like `7123456789:AAH...`. Save this.

### Step 2: Get your chat ID

1. Send any message to your new bot in Telegram (just say "hi").
2. Open this URL in your browser (replace YOUR_TOKEN):
   ```
   https://api.telegram.org/botYOUR_TOKEN/getUpdates
   ```
3. Find `"chat":{"id":123456789}` in the JSON. That number is your **chat ID**.

Alternatively, search for **@userinfobot** on Telegram and it'll tell you your ID.

---

## Phase 5: Upload & Configure the Bot

### Step 1: Upload bot files to your server

**Option A — SCP from local machine**:
```bash
scp -r ./polymarket-weather-bot/* botuser@164.90.xxx.xxx:~/polymarket-weather-bot/
```

**Option B — Clone or copy-paste**:
Upload each file via your preferred method. The files are:
```
polymarket-weather-bot/
├── config.py
├── forecast_scanner.py
├── polymarket_parser.py
├── decision_engine.py
├── execution.py
├── main.py
├── requirements.txt
├── .env.example
└── logs/
```

### Step 2: Install dependencies

```bash
cd ~/polymarket-weather-bot
source venv/bin/activate
pip install -r requirements.txt
```

If `py-clob-client` fails to install (it can be finicky), that's fine for dry-run mode —
the bot gracefully falls back. For live mode you'll need it:
```bash
pip install py-clob-client
# If that fails, try:
pip install py-clob-client --no-deps
pip install eth-account web3 requests
```

### Step 3: Configure .env

```bash
cp .env.example .env
nano .env
```

Fill in:
```env
# Your Polygon private key (from Phase 3)
PRIVATE_KEY=0xabc123...your_actual_key...

# Telegram (from Phase 4)
TELEGRAM_TOKEN=7123456789:AAHxyz...
TELEGRAM_CHAT_ID=123456789

# Start conservative
BANKROLL=200
MAX_POSITION_USD=5
DAILY_LOSS_CAP=20
MODE=dry-run
```

Leave everything else at defaults for now.

---

## Phase 6: Run in Dry-Run Mode

This is the critical validation step. The bot fetches real data and computes real signals
but does NOT place any orders.

```bash
cd ~/polymarket-weather-bot
source venv/bin/activate
mkdir -p logs

# Run it
python main.py
```

You should see output like:
```
============================================================
  Polymarket Weather Trading Bot
  Mode: 🧪 DRY-RUN
  Bankroll: $200
  Cities: New York, Chicago, Los Angeles, Miami, Houston, Dallas
  Scan interval: 120s
============================================================
...
Step 1/4: Fetching NWS forecasts...
Forecast scan complete: 42 city-date pairs across 6 cities
Step 2/4: Discovering temperature markets on Polymarket...
Parsed 8 active temperature markets
...
```

### What to check in dry-run

1. **Forecasts are loading**: Check `logs/bot.log` for NWS data.
2. **Markets are discovered**: The bot finds active temperature markets.
3. **Signals make sense**: Open `logs/trades.csv`:
   ```bash
   cat logs/trades.csv
   # or for a nicer view:
   column -t -s, logs/trades.csv | head -20
   ```
   Verify that:
   - `p_true` values are reasonable (e.g., 0.85+ for the "right" bucket)
   - `ev` is positive
   - `edge` is meaningful (0.08+)
   - `size_usd` respects your caps

4. **Telegram alerts arrive**: You should get messages on your phone for each simulated trade.

5. **No errors**: Check `logs/bot.log` for exceptions.

### Run for at least 24-48 hours in dry-run before going live.

Look at whether the signals would have been profitable by checking resolved markets
against the bot's predictions.

---

## Phase 7: Go Live

Only after dry-run validation:

### Step 1: Edit .env

```bash
nano .env
# Change:
MODE=live
```

### Step 2: Run with persistence (so it survives SSH disconnection)

**Option A — screen** (simplest):
```bash
screen -S polybot
cd ~/polymarket-weather-bot
source venv/bin/activate
python main.py

# Detach: press Ctrl+A, then D
# Reattach later: screen -r polybot
```

**Option B — tmux**:
```bash
tmux new -s polybot
cd ~/polymarket-weather-bot
source venv/bin/activate
python main.py

# Detach: press Ctrl+B, then D
# Reattach: tmux attach -t polybot
```

**Option C — systemd** (auto-restarts on crash):

Create the service file:
```bash
sudo nano /etc/systemd/system/polymarket-bot.service
```

Paste:
```ini
[Unit]
Description=Polymarket Weather Trading Bot
After=network.target

[Service]
User=botuser
WorkingDirectory=/home/botuser/polymarket-weather-bot
ExecStart=/home/botuser/polymarket-weather-bot/venv/bin/python main.py
Restart=always
RestartSec=30
Environment=PATH=/home/botuser/polymarket-weather-bot/venv/bin:/usr/bin

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable polymarket-bot
sudo systemctl start polymarket-bot

# Check status
sudo systemctl status polymarket-bot

# View logs
sudo journalctl -u polymarket-bot -f
```

---

## Phase 8: Monitor & Maintain

### Daily checks

1. **Telegram**: You'll get trade alerts and periodic PnL summaries.
2. **Logs**:
   ```bash
   # Recent trades
   tail -20 logs/trades.csv

   # Bot health
   tail -50 logs/bot.log

   # Scan stats
   tail -10 logs/scans.csv
   ```
3. **Wallet balance**: Check on polygonscan.com or via MetaMask.

### If something goes wrong

The bot has multiple safety layers:
- **Daily loss cap**: Auto-shuts down if losses exceed $DAILY_LOSS_CAP.
- **Instability detection**: Skips markets where forecasts are swinging wildly.
- **Position caps**: Never exceeds per-market or per-position limits.
- **Graceful shutdown**: Ctrl+C (or `kill` the process) stops cleanly.

### Scaling up

After 1-2 weeks of profitable live trading:
1. Increase `BANKROLL` and `MAX_POSITION_USD` gradually.
2. Consider adding more cities in `CITIES` and `NWS_POINTS`.
3. Decrease `SCAN_INTERVAL_SEC` to 60 for faster reaction.
4. Reduce `MIN_EDGE` slightly (e.g., 0.06) to capture more trades.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No active temperature markets found" | Normal if Polymarket doesn't currently list temp markets. The bot will keep scanning. |
| NWS API errors (500/503) | Transient — the bot retries next cycle. NWS occasionally has outages. |
| `py-clob-client` import error | Install with `pip install py-clob-client`. Only needed for live mode. |
| Telegram not sending | Verify TELEGRAM_TOKEN and TELEGRAM_CHAT_ID. Send a test message to your bot first. |
| "Daily loss cap breached" | Working as intended — bot paused to protect your bankroll. Resets at midnight. |
| Order rejected by Polymarket | Check wallet has enough USDC. Verify approvals (Phase 3, Step 4). |
| Bot crashes on startup | Check `.env` format (no quotes around values, no trailing spaces). |

---

## Cost Summary

| Item | Cost |
|------|------|
| VPS (DigitalOcean) | $6/month |
| NWS/NOAA API | Free |
| Polymarket/Gamma API | Free |
| Telegram Bot API | Free |
| Polygon gas fees | ~$0.01-0.05 per trade |
| Starting USDC bankroll | $100-500 recommended |
| **Total to start** | **~$106-506** |
