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
Run the cocoa market surveillance pipeline. Follow these steps EXACTLY.
Do NOT recreate any Python scripts — they come from git.
Do NOT send any Telegram messages yourself — the Python scripts handle alerts.
Do NOT improvise or add extra steps. Just execute the commands below.

1. CLONE REPO (fresh container each session):
   cd /workspace && git clone https://github.com/Alex-richardson1/cocoa_agent.git . 2>/dev/null || git pull origin main

2. INSTALL DEPENDENCIES (quiet):
   pip install -q pandas yfinance requests feedparser python-dotenv anthropic ta earthengine-api 2>/dev/null

3. RESTORE STATE from memory:
   mkdir -p /mnt/memory/cocoa-surveillance-memory/state
   cp /mnt/memory/cocoa-surveillance-memory/state/*.json /workspace/ 2>/dev/null || echo "No prior state"

4. CREATE CREDENTIALS:
   The .env and gee-service-account.json content is provided above this instruction — create those files in /workspace/ if not already present.

5. RUN DATA GATHERER:
   cd /workspace && python3 cocoa_data_gatherer.py

6. RUN CROP MONITOR (skip if GEE fails):
   python3 cocoa_crop_monitor.py --incremental 2>/dev/null || echo "Crop monitor skipped"

7. RUN AGENT (no --force-alert):
   python3 cocoa_agent.py

8. SAVE STATE back to memory:
   cp /workspace/cocoa_shadow_ledger.json /workspace/cocoa_prediction_ledger.json /workspace/cocoa_opportunity_log.json /workspace/cot_cocoa_history.json /workspace/cocoa_crop_health.json /workspace/cocoa_feedback_summary.json /workspace/climatology_cache.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null
   cp /workspace/cocoa_postmortems.json /workspace/cocoa_weekly_history.json /workspace/cocoa_weekly_report.md /workspace/cocoa_daily_report.md /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null

9. PRINT SUMMARY (console only — no Telegram):
   Print the opportunity score, alert level, and one-line summary. Nothing else.
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
        "cot_cocoa_history.json",
        "cocoa_crop_health.json",
        "climatology_cache.json",
        "grinding_data_cache.json",
    ]

    seeded = 0
    for filename in state_files:
        if os.path.exists(filename):
            print(f"  Seeding {filename}...")
            with open(filename, "r") as f:
                content = f.read()
            client.beta.memory_stores.memories.create(
                memory_store_id=store_id,
                content=content,
                metadata={"filename": filename, "type": "state"},
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
        }],
    )
    print(f"Session ID: {session.id}")

        # ── Build instruction with credentials inline ─
    gee_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "")
    env_content = os.environ.get("COCOA_ENV_FILE", "")

    setup_commands = ""
    if gee_json:
        # Escape single quotes for bash heredoc safety
        gee_escaped = gee_json.replace("'", "'\\''")
        setup_commands += f"\nAlso create /workspace/gee-service-account.json with this content:\n{gee_json}\n"
    if env_content:
        setup_commands += f"\nAlso create /workspace/.env with this content:\n{env_content}\n"

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
    if not AGENT_ID or not ENVIRONMENT_ID:
        print("ERROR: Set AGENT_ID and ENVIRONMENT_ID environment variables.")
        sys.exit(1)

    if "--setup" in sys.argv:
        setup()
    else:
        run_daily()
