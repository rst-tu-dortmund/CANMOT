#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1 || { echo "usage: $0 ENVIRONMENT_NAME" >&2; exit 2; }
exec canmot preflight --environment "$1"
