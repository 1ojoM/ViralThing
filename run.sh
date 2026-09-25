#!/bin/sh
# Установка (один раз) и запуск ViralThing на http://localhost:8000
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt || exit 1
fi
[ -f .env ] || cp .env.example .env
echo "Откройте http://localhost:8000"
exec .venv/bin/uvicorn app:app --port 8000
