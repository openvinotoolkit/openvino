#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
ONEDNN_DIR="$THIS_DIR/onednn_local"
BUILD_DIR="$ONEDNN_DIR/build"

if [[ ! -d "$ONEDNN_DIR" ]]; then
  echo "Cloning oneDNN into $ONEDNN_DIR ..."
  git clone https://github.com/oneapi-src/oneDNN.git "$ONEDNN_DIR"
else
  echo "Using existing $ONEDNN_DIR"
fi

cmake -S "$ONEDNN_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DDNNL_BUILD_TESTS=ON -DDNNL_BUILD_EXAMPLES=OFF
cmake --build "$BUILD_DIR" --target benchdnn -j

BIN="$BUILD_DIR/tests/benchdnn/benchdnn"
echo "benchdnn built: $BIN"
echo "hint: run comparison via: $THIS_DIR/gemmv_vs_benchdnn.sh"

