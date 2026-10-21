#!/usr/bin/env bash
# Build the 2-page notes PDF: charts from saved results, then report/note.html → report/main.pdf (headless Chrome).
#   bash scripts/build_note_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/make_note_figures.py
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE="$(mktemp -d)"   # throwaway profile, never the user's own
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --user-data-dir="$PROFILE" \
  --print-to-pdf=report/main.pdf "file://$PWD/report/note.html" 2>/dev/null
echo "wrote report/main.pdf"
