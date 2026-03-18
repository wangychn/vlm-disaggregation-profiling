#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT_DIR/data/coco}"
SPLIT="${2:-val2017}"

mkdir -p "$DEST"

cd "$ROOT_DIR"
echo "Downloading COCO split=$SPLIT into $DEST"
python3 -m engine.datasets --root "$DEST" --split "$SPLIT" --download
