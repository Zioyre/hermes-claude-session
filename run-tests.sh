#!/bin/sh
# Run the plugin's tests on Hermes's runtime interpreter, with test-only deps from .testdeps.
HPY=${HERMES_PYTHON:-$(command -v python3)}
ROOT=$(cd "$(dirname "$0")" && pwd)
DEPS=${TESTDEPS:-$ROOT/.testdeps}
HERMES_SRC=${HERMES_SRC:-$HOME/.hermes/hermes-agent}
cd "$ROOT" && PYTHONPATH="$DEPS:$HERMES_SRC${PYTHONPATH:+:$PYTHONPATH}" exec "$HPY" -m pytest -q -p no:cacheprovider "$@"
