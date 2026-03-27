#!/usr/bin/env bash
# Build script for Quoridor C++ engine
# Downloads single-header dependencies if needed, then compiles.
# Requires: cpp-httplib and nlohmann-json (installed via brew or as local headers)

set -e
cd "$(dirname "$0")"

# --- Locate headers ---------------------------------------------------------
# Prefer local copies (./httplib.h, ./json.hpp); fall back to brew paths.

BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
HTTPLIB_SYSTEM="${BREW_PREFIX}/include/httplib.h"
JSON_SYSTEM="${BREW_PREFIX}/include/nlohmann/json.hpp"
JSON_SYSTEM_FLAT="${BREW_PREFIX}/include/json.hpp"   # some installs flatten it

NEED_DOWNLOAD=0

if [ ! -f httplib.h ] && [ ! -f "$HTTPLIB_SYSTEM" ]; then
    echo "cpp-httplib not found. Trying: brew install cpp-httplib"
    brew install cpp-httplib || { echo "ERROR: install cpp-httplib manually"; exit 1; }
fi

if [ ! -f json.hpp ] && [ ! -f "$JSON_SYSTEM" ] && [ ! -f "$JSON_SYSTEM_FLAT" ]; then
    echo "nlohmann-json not found. Trying: brew install nlohmann-json"
    brew install nlohmann-json || { echo "ERROR: install nlohmann-json manually"; exit 1; }
fi

# --- Extra include paths for the compiler ------------------------------------
EXTRA_INC="-I${BREW_PREFIX}/include"

# --- Compiler flags ----------------------------------------------------------
CXX="${CXX:-c++}"
CXXFLAGS="-std=c++17 -O3 -march=native -DNDEBUG ${EXTRA_INC}"
LDFLAGS="-lpthread"

echo "Compiling engine.cpp  (CXX=${CXX})..."
"$CXX" $CXXFLAGS engine.cpp -o engine $LDFLAGS

echo ""
echo "Build successful: $(pwd)/engine"
echo ""
echo "Usage:"
echo "  ./engine                    # uses default base dir (/Users/abhijay/barricade-engine)"
echo "  ./engine /path/to/base/dir  # explicit base dir (for game_log.jsonl, dom_debug.json)"
echo ""
echo "Server listens on http://127.0.0.1:5123"
