#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'Python environment not found. First follow the installation steps in README.md.'
  read -r '?Press Enter to close.'
  exit 1
fi
.venv/bin/python jump.py --auto
print '\nAutomatic mode stopped. Local reports, if any, are under logs/.'
read -r '?Press Enter to close.'
