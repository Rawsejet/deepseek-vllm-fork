#!/bin/bash
# Stop DeepSeek-V4-Flash server
# Usage: ./stop-deepseek.sh
#
# Kills only this server (process group + its port) — does not sweep other
# vLLM instances, so Qwen servers on other ports are untouched.

LOG_DIR=~/deepseek-service
PID_FILE=$LOG_DIR/deepseek-v4-server.pid
PORT_FILE=$LOG_DIR/deepseek-v4-server.port
BACKEND_FILE=$LOG_DIR/deepseek-v4-server.backend
DEFAULT_PORT=8095

# Read port from .port file (saved by start-deepseek.sh), fall back to default
if [ -f "$PORT_FILE" ]; then
    PORT=$(cat "$PORT_FILE")
else
    PORT=$DEFAULT_PORT
fi

echo "Stopping DeepSeek-V4-Flash (port $PORT)..."

# Kill by process group if we have the PID (catches vLLM's renamed
# EngineCore/Worker_TP* children, which share the launcher's process group)
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "$PGID" ] && [ "$PGID" != "0" ]; then
        echo "  Killing process group $PGID..."
        kill -- -$PGID 2>/dev/null
    else
        echo "  Main process not running (stale PID file)."
    fi
else
    echo "  No PID file — sweeping port $PORT..."
fi

# Sweep by port binding
fuser -k ${PORT}/tcp 2>/dev/null

sleep 3

# Force-kill any stragglers bound to that port
fuser -9k ${PORT}/tcp 2>/dev/null

sleep 1

rm -f "$PID_FILE" "$PORT_FILE" "$BACKEND_FILE"

# Verify
if ss -tlnp | grep -q ":${PORT} " 2>/dev/null; then
    echo "  ⚠ Warning: port $PORT still in use!"
else
    echo "  ✓ Port $PORT is free."
fi
echo "✓ DeepSeek-V4-Flash server stopped."

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU %s: %s MB used, %s MB free\n", $1, $3, $4}'
echo ""
echo "Done."
