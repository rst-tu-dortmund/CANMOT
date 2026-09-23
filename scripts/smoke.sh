#!/usr/bin/env bash
set -euo pipefail
exec python -m pytest tests/test_smoke.py -q
