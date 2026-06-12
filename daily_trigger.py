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
Do NOT recreate any Python scripts. Do NOT send Telegram messages. Do NOT improvise.

1. CLONE REPO:
   cd /workspace && git clone https://github.com/Alex-richardson1/cocoa_agent.git . 2>/dev/null || git pull origin main 2>/dev/null

2. MOVE CREDENTIALS:
   cp /tmp/.env /workspace/.env 2>/dev/null; cp /tmp/gee-service-account.json /workspace/gee-service-account.json 2>/dev/null

3. INSTALL DEPENDENCIES:
   pip install -q pandas numpy yfinance requests feedparser python-dotenv httpx earthengine-api 2>/dev/null || true
   python3 -c "import importlib, pathlib, site; exec(\"try:\\n import sgmllib\\nexcept ImportError:\\n sp=site.getsitepackages()[0]\\n pathlib.Path(sp+'/sgmllib.py').write_text('from html.parser import HTMLParser as SGMLParser\\\\nclass TestSGMLParser: pass\\\\n')\\n print('sgmllib stub created')\")" 2>/dev/null || true

4. RESTORE STATE:
   mkdir -p /mnt/memory/cocoa-surveillance-memory/state; cp /mnt/memory/cocoa-surveillance-memory/state/*.json /workspace/ 2>/dev/null; cp /mnt/memory/cocoa-surveillance-memory/state/*.md /workspace/ 2>/dev/null; echo "State restored"

5. RUN PIPELINE:
   cd /workspace && python3 cocoa_pipeline.py

6. READ THE PIPELINE OUTPUT and the health report. Print the health summary.

7. SAVE STATE:
   cp /workspace/cocoa_shadow_ledger.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_prediction_ledger.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_opportunity_log.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cot_cocoa_history.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_crop_health.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_feedback_summary.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/climatology_cache.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_postmortems.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_weekly_history.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_daily_snapshot.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; cp /workspace/cocoa_pipeline_health.json /mnt/memory/cocoa-surveillance-memory/state/ 2>/dev/null; echo "State saved"

8. PRINT SUMMARY: Print the pipeline health, price, COT signal, and any failures. Nothing else.
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
    if not AGENT_ID or not ENVIRONMENT_ID:
        print("ERROR: Set AGENT_ID and ENVIRONMENT_ID environment variables.")
        sys.exit(1)

    if "--setup" in sys.argv:
        setup()
    else:
        run_daily()
