#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 2 || { echo "usage: $0 ENVIRONMENT_NAME RECIPE" >&2; exit 2; }
exec canmot optimize "$2" --environment "$1"
