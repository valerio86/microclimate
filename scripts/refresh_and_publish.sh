#!/bin/bash
# Daily job: refresh data/dashboard.html, then publish it to GitHub Pages.
# Installed via com.microclimate.refresh.plist.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/microclimate refresh --quiet
scripts/publish_pages.sh
