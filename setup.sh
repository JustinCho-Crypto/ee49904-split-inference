#!/usr/bin/env bash
# Create the pinned environment and fetch the course kit.
#   PYTHON  : a Python >= 3.10 (default: python3.13, then python3)
#   SPLIT_VENV : where the virtualenv goes (default ~/split-inference-venv-313)
#   KIT_DIR : where the course repo is cloned (default ./kit)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$(command -v python3.13 || command -v python3)}"
VENV="${SPLIT_VENV:-$HOME/split-inference-venv-313}"
KIT_DIR="${KIT_DIR:-$HERE/kit}"
KIT_REPO="https://github.com/sunghyunc7/ee49904-term-project-2026f.git"
KIT_COMMIT="e69d540"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10 (torch 2.13 does not support 3.9)"'
[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$HERE/requirements-lock.txt"
"$VENV/bin/python" -c "import sys, torch, torchvision; print('python', sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__)"

if [ ! -d "$KIT_DIR/.git" ]; then git clone -q "$KIT_REPO" "$KIT_DIR"; fi
git -C "$KIT_DIR" checkout -q "$KIT_COMMIT"
echo "kit at $(git -C "$KIT_DIR" rev-parse --short HEAD) -> $KIT_DIR/split-inference/starter"
