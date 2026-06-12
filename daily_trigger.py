"""
=============================================================
  COCOA AGENT — Daily Trigger (Memory Store version)
=============================================================
  Creates a Managed Agents session with a persistent memory
  store attached, runs the full pipeline. State files live in
  /mnt/memory/ and survive between sessions automatically.

  Setup (run once):
    python daily_trigger.py --setup

  Daily run:
    python daily_trigger.py

  Required env vars:
    ANTHROPIC_API_KEY
    AGENT_ID
    ENVIRONMENT_ID

  Optional:
    MEMORY_STORE_ID    — created by --setup, saved to .memory_store_id
    GEE_SERVICE_ACCOUNT_JSON
    COCOA_ENV_FILE
=============================================================
"""

import os
import sys
import json
import time
import anthropic

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

AGENT_ID       = os.environ.get("AGENT_ID", "")
ENVIRONMENT_ID = os.environ.get("ENVIRONMENT_ID", "")
MEMORY_STORE_ID_FILE = ".memory_store_id"

DAILY_INSTRUCTION = """
Run this bash script exactly. Print stdout/stderr. Do not add steps.

```bash
set -u

cd /workspace

if [ -d .git ]; then
  git fetch origin main
  git reset --hard origin/main
else
  git clone https://github.com/Alex-richardson1/cocoa_agent.git .
fi

echo "Restoring runtime credential files..."

if [ -f /tmp/.env ]; then
  cp /tmp/.env /workspace/.env
  echo ".env copied to /workspace/.env"
else
  echo "WARNING: /tmp/.env not found"
fi

if [ -f /tmp/gee-service-account.json ]; then
  cp /tmp/gee-service-account.json /workspace/gee-service-account.json
  echo "GEE service account copied to /workspace/gee-service-account.json"
else
  echo "WARNING: /tmp/gee-service-account.json not found"
fi

if [ -f /workspace/.env ]; then
  echo "Exporting environment variables from /workspace/.env"
  set -a
  . /workspace/.env
  set +a
else
  echo "WARNING: /workspace/.env not found; cocoa_agent.py may not have API keys"
fi

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import yfinance, pandas, numpy, requests, bs4, dotenv
print("core imports OK")
PY

MEMORY_DIR="/mnt/memory/cocoa-surveillance-memory"
STATE_DIR="$MEMORY_DIR/state"

echo "Checking memory mount..."
if [ ! -d "$MEMORY_DIR" ]; then
  echo "ERROR: memory store mount not found at $MEMORY_DIR"
  echo "Available /mnt contents:"
  ls -lah /mnt || true
  exit 1
fi

echo "Checking memory state directory..."
if [ ! -d "$STATE_DIR" ]; then
  echo "ERROR: memory state directory not found at $STATE_DIR"
  echo "Memory store contents:"
  ls -lah "$MEMORY_DIR" || true
  exit 1
fi

echo "Memory state files:"
ls -lah "$STATE_DIR" || true

echo "Restoring state files into /workspace..."
cp "$STATE_DIR"/*.json /workspace/ || true
cp "$STATE_DIR"/*.md /workspace/ || true

echo "Restored workspace state files:"
ls -lah /workspace/*.json /workspace/*.md 2>/dev/null || true

echo "COT history restore check:"
python3 - <<'PY'
import json
import os

path = "/workspace/cot_cocoa_history.json"

if not os.path.exists(path):
    print("ERROR: cot_cocoa_history.json was not restored")
else:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"cot_cocoa_history type: {type(data).__name__}")
    print(f"cot_cocoa_history length: {len(data) if hasattr(data, '__len__') else 'N/A'}")

    if isinstance(data, list) and data:
        first = data[0]
        last = data[-1]
        print(f"first report_date: {first.get('report_date')}")
        print(f"last report_date: {last.get('report_date')}")
        print(f"last managed_money_net: {last.get('managed_money_net')}")
    elif isinstance(data, dict):
        print(f"top-level keys: {list(data.keys())[:20]}")
PY

PIPELINE_STATUS=0
python3 cocoa_pipeline.py || PIPELINE_STATUS=$?

# Run the analyst/alerting layer only if price exists.
if [ "$PIPELINE_STATUS" -eq 0 ]; then
  python3 cocoa_agent.py || true
fi

for f in \
  cocoa_shadow_ledger.json \
  cocoa_prediction_ledger.json \
  cocoa_opportunity_log.json \
  cocoa_monitor_log.json \
  cot_cocoa_history.json \
  ice_warehouse_history.json \
  cocoa_crop_health.json \
  cocoa_crop_diff.json \
  cocoa_feedback_summary.json \
  climatology_cache.json \
  cocoa_postmortems.json \
  cocoa_weekly_history.json \
  cocoa_daily_snapshot.json \
  cocoa_pipeline_health.json \
  cocoa_daily_report.md \
  cocoa_daily_rec.json
do
  [ -f "/workspace/$f" ] && cp "/workspace/$f" /mnt/memory/cocoa-surveillance-memory/state/
done

```python
if [ -f cocoa_pipeline_health.json ]; then
  cat cocoa_pipeline_health.json
else
  echo "WARNING: cocoa_pipeline_health.json was not created"
fi

echo "Saved memory state files:"
ls -lah "$STATE_DIR" || true

exit "$PIPELINE_STATUS"
exit "$PIPELINE_STATUS"

"""

def get_memory_store_id() -> str:
    """Load or prompt for the memory store ID."""
    # Check env var first
    store_id = os.environ.get("MEMORY_STORE_ID")
    if store_id:
        return store_id

    # Check local file
    try:
        with open(MEMORY_STORE_ID_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def setup():
    """One-time setup: create the memory store."""
    client = anthropic.Anthropic()

    print("Creating memory store for cocoa agent...")
    store = client.beta.memory_stores.create(
        name="cocoa-surveillance-memory",
        description=(
            "Persistent state for the cocoa trading surveillance agent. "
            "Contains prediction ledgers, shadow predictions, COT history, "
            "crop health cache, climatology normals, post-mortems, and weekly reports."
        ),
    )

    store_id = store.id
    print(f"Memory store created: {store_id}")

    # Save locally
    with open(MEMORY_STORE_ID_FILE, "w") as f:
        f.write(store_id)
    print(f"Saved to {MEMORY_STORE_ID_FILE}")

    # Seed with initial state files if they exist locally
    state_files = [
        "cocoa_shadow_ledger.json",
        "cocoa_prediction_ledger.json",
        "cocoa_opportunity_log.json",
        "cocoa_monitor_log.json",
        "cot_cocoa_history.json",
        "ice_warehouse_history.json",
        "cocoa_crop_health.json",
        "cocoa_crop_diff.json",
        "cocoa_feedback_summary.json",
        "climatology_cache.json",
        "cocoa_postmortems.json",
        "cocoa_weekly_history.json",
        "grinding_data_cache.json",
        "cocoa_daily_snapshot.json",
        "cocoa_pipeline_health.json",
        "cocoa_daily_report.md",
        "cocoa_daily_rec.json",
    ]

    seeded = 0
    for filename in state_files:
        if os.path.exists(filename):
            print(f"  Seeding {filename}...")
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()

            client.beta.memory_stores.memories.create(
                memory_store_id=store_id,
                path=f"/state/{filename}",
                content=content,
            )
            seeded += 1

    print(f"\nSetup complete. {seeded} state files seeded.")
    print(f"Add MEMORY_STORE_ID={store_id} to your GitHub secrets.")
    return store_id


def run_daily():
    """Create a session with memory attached and run the pipeline."""
    client = anthropic.Anthropic()
    memory_store_id = get_memory_store_id()

    print("=" * 55)
    print("  COCOA AGENT — Daily Trigger")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 55)

    if not memory_store_id:
        print("ERROR: No memory store ID. Run: python daily_trigger.py --setup")
        sys.exit(1)

    # ── Create session with memory store attached ─
    print(f"\nCreating session (memory: {memory_store_id[:20]}...)...")

    session = client.beta.sessions.create(
        agent=AGENT_ID,
        environment_id=ENVIRONMENT_ID,
        title=f"Daily run {time.strftime('%Y-%m-%d')}",
        resources=[{
            "type": "memory_store",
            "memory_store_id": memory_store_id,
            "access": "read_write",
            "instructions": (
                "Persistent cocoa surveillance state. State files are stored "
                "under /state/. Restore them before running the pipeline and "
                "write updated state files back after the run."
            ),
        }],
    )
    print(f"Session ID: {session.id}")

        # ── Build instruction with credentials inline ─
    gee_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "")
    env_content = os.environ.get("COCOA_ENV_FILE", "")

    setup_commands = ""
    if gee_json:
        setup_commands += f"\nCreate /tmp/gee-service-account.json with this content:\n{gee_json}\n"
    if env_content:
        setup_commands += f"\nCreate /tmp/.env with this content:\n{env_content}\n"

    full_instruction = setup_commands + "\n" + DAILY_INSTRUCTION

    # ── Send instruction and stream response ──────
    print("\nRunning pipeline...")
    print("-" * 55)

    with client.beta.sessions.events.stream(session.id) as stream:
        client.beta.sessions.events.send(
            session.id,
            events=[{
                "type": "user.message",
                "content": [{"type": "text", "text": full_instruction}],
            }],
        )
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if hasattr(block, "text"):
                        print(block.text, end="")
            elif event.type == "session.status_idle":
                print("\n\nAgent finished.")
                break

    print("-" * 55)
    print("\n✅ Daily run complete. State saved to memory store.")


if __name__ == "__main__":
    if "--setup" in sys.argv:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)
        setup()
    else:
        if not AGENT_ID or not ENVIRONMENT_ID:
            print("ERROR: Set AGENT_ID and ENVIRONMENT_ID environment variables.")
            sys.exit(1)
        run_daily()
