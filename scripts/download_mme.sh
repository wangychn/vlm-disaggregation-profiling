#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT_DIR/data/mme}"
SPLIT="${2:-test}"
MAX_ITEMS="${3:-50}"
CACHE_DIR="${4:-}"
CATEGORY="${5:-}"

cd "$ROOT_DIR"

CMD=(
  python3 -m engine.datasets
  --root "$DEST"
  --split "$SPLIT"
  --max-items "$MAX_ITEMS"
)

if [[ -n "$CACHE_DIR" ]]; then
  CMD+=(--cache-dir "$CACHE_DIR")
fi

if [[ -n "$CATEGORY" ]]; then
  CMD+=(--category "$CATEGORY")
fi

echo "Preparing MME split=$SPLIT max_items=$MAX_ITEMS into $DEST"
"${CMD[@]}"
