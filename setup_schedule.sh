#!/usr/bin/env bash
# Set up the scheduler to auto-start on weekday mornings via macOS launchd.
#
# Usage:
#   ./setup_schedule.sh              # install (runs Mon-Fri 8:25 AM)
#   ./setup_schedule.sh uninstall    # remove
#
# The scheduler itself handles market hours and intervals.
# launchd just ensures it starts each weekday morning.

set -euo pipefail

LABEL="com.stockagents.scheduler"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python3"
LOG="$PROJECT_DIR/data/launchd.log"

if [ "${1:-}" = "uninstall" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Uninstalled $LABEL"
    exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/data"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${PROJECT_DIR}/scheduler.py</string>
        <string>--interval</string>
        <string>60</string>
        <string>--start</string>
        <string>08:30</string>
        <string>--end</string>
        <string>15:00</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>

    <!-- Run Mon-Fri at 8:25 AM local time -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>25</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>25</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>25</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>25</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>25</integer></dict>
    </array>

    <key>StandardOutPath</key>
    <string>${LOG}</string>
    <key>StandardErrorPath</key>
    <string>${LOG}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
PLIST_EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "Installed and loaded: $LABEL"
echo "  Plist:    $PLIST"
echo "  Schedule: Mon-Fri at 8:25 AM → runs pipeline every 60 min until 3:00 PM"
echo "  Log:      $LOG"
echo ""
echo "To change interval: edit the plist or run scheduler.py directly:"
echo "  python scheduler.py --interval 30 --start 08:30 --end 15:00"
echo ""
echo "To uninstall: ./setup_schedule.sh uninstall"
