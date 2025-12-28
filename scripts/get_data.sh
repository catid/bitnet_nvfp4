#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/data"
OUT_FILE="$OUT_DIR/tinyshakespeare.txt"

mkdir -p "$OUT_DIR"

if [[ -f "$OUT_FILE" ]]; then
  echo "Dataset already exists: $OUT_FILE"
  exit 0
fi

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$OUT_FILE"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT_FILE" "$URL"
else
  echo "Error: need curl or wget to download dataset." >&2
  exit 1
fi

echo "Downloaded: $OUT_FILE"
