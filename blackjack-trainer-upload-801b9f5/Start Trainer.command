#!/bin/sh
cd "$(dirname "$0")" || exit 1

fail() {
    echo "$1" >&2
    if [ -t 0 ]; then
        printf 'Press Return to close. '
        read -r reply
    fi
    exit 1
}

python=.venv/bin/python
if [ ! -x "$python" ]; then
    # Finder's Terminal session may not include Homebrew on its PATH.
    export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
    for candidate in python3.13 python3.14 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            break
        fi
    done
    echo "Setting up Match Trainer for the first time..."
    "$candidate" -m venv .venv ||
        fail "Setup failed. Install Python 3.13 from python.org and open this file again."
fi

if ! "$python" -c 'import numpy' 2>/dev/null; then
    echo "Installing what Match Trainer needs. This requires internet access."
    "$python" -m pip install --disable-pip-version-check -r requirements.txt ||
        fail "Installation failed. Check the error above and open this file again to retry."
fi

"$python" -m trainer || fail "Match Trainer stopped with an error. See the message above."
