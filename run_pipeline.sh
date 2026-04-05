#!/usr/bin/env bash
# Run the multi-agent stock analysis pipeline (Ollama + LangGraph).
# Usage: ./run_pipeline.sh
# Schedule for Monday morning via cron or launchd.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin/python3"
LOG="$SCRIPT_DIR/data/pipeline.log"

mkdir -p "$SCRIPT_DIR/data"

echo "========================================" | tee -a "$LOG"
echo "Pipeline run: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Ensure Ollama is running (macOS app auto-starts the server)
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama..." | tee -a "$LOG"
    open -a Ollama 2>/dev/null || true
    sleep 8
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "ERROR: Ollama not reachable at localhost:11434" | tee -a "$LOG"
        exit 1
    fi
fi
echo "Ollama OK" | tee -a "$LOG"

# 18-minute hard timeout (15 min target + 3 min buffer)
if command -v timeout &>/dev/null; then
    timeout 1080 "$VENV" -m pipeline.run_pipeline 2>&1 | tee -a "$LOG"
elif command -v gtimeout &>/dev/null; then
    gtimeout 1080 "$VENV" -m pipeline.run_pipeline 2>&1 | tee -a "$LOG"
else
    "$VENV" -m pipeline.run_pipeline 2>&1 | tee -a "$LOG"
fi

echo "Exit code: $?" | tee -a "$LOG"
echo "" | tee -a "$LOG"
