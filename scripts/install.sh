#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

libero=third_party/LIBERO
revision=8f1084e3132a39270c3a13ebe37270a43ece2a01

if [[ ! -d "$libero/.git" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$libero"
fi
if [[ -n "$(git -C "$libero" status --porcelain)" ]]; then
    echo "LIBERO has local changes; commit or stash them before installing." >&2
    exit 1
fi
git -C "$libero" checkout --detach "$revision"

if grep -aq '^version https://git-lfs.github.com/spec/v1$' checkpoints/corp_libero.pt; then
    git lfs install --local
    git lfs pull --include="checkpoints/corp_libero.pt" --exclude=""
fi
(cd checkpoints && sha256sum --check SHA256SUMS)

if [[ ! -x .venv/bin/python ]]; then
    python3.12 -m venv .venv
fi
python=.venv/bin/python
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
"$python" -m pip install --upgrade pip
"$python" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install -c constraints.txt ".[libero]"
"$python" -m pip install -c constraints.txt -e "$libero"
"$python" -m pip check

echo "Installed. Run: source .venv/bin/activate"
