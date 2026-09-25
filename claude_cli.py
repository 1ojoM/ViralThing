"""Бэкенд «по подписке»: запросы идут через Claude Code CLI (`claude -p`),
вошедший в ваш аккаунт Claude Pro/Max. API-ключ не нужен.

Только для личного использования: правила Anthropic не разрешают
пропускать через свою подписку запросы других пользователей.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

TIMEOUT = 600  # сек на один запрос


def _binary() -> str:
    path = (
        os.environ.get("VIRALTHING_CLAUDE_BIN")
        or shutil.which("claude")
        or str(Path.home() / ".local/bin/claude")
    )
    if not Path(path).exists():
        raise RuntimeError(
            "Не найден Claude Code CLI. Установите: curl -fsSL https://claude.ai/install.sh | bash"
        )
    return path


def _clean_env() -> dict:
    # Без этого CLI взял бы API-ключ из .env вместо подписки
    return {
        k: v for k, v in os.environ.items()
        if not k.startswith("ANTHROPIC_")
        and not k.startswith("CLAUDE_CODE_")
        and k not in ("CLAUDECODE", "CLAUDE_EFFORT")
    }


def call(system: str, content: list, schema: type[BaseModel], effort: str) -> BaseModel:
    cmd = [
        _binary(), "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json", "--verbose",
        "--json-schema", json.dumps(schema.model_json_schema(), ensure_ascii=False),
        "--system-prompt", system,
        "--tools", "",                      # файлы и команды модели не нужны
        "--permission-mode", "dontAsk",     # никогда не ждать подтверждений
        "--no-session-persistence",
        "--effort", effort,
    ]
    if model := os.environ.get("VIRALTHING_CLI_MODEL"):
        cmd += ["--model", model]

    message = {"type": "user", "message": {"role": "user", "content": content}}
    with tempfile.TemporaryDirectory(prefix="viralthing_cli_") as cwd:
        try:
            proc = subprocess.run(
                cmd, input=json.dumps(message, ensure_ascii=False) + "\n",
                capture_output=True, text=True, timeout=TIMEOUT,
                cwd=cwd, env=_clean_env(),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code не ответил за 10 минут — попробуйте ещё раз")

    result = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event

    if result is None:
        detail = (proc.stderr or proc.stdout).strip()[-500:]
        raise RuntimeError(f"Claude Code завершился с ошибкой: {detail or proc.returncode}")
    if result.get("is_error") or result.get("structured_output") is None:
        text = str(result.get("result") or result.get("subtype") or "")
        if "login" in text.lower() or "log in" in text.lower() or "auth" in text.lower():
            raise RuntimeError("Claude Code не авторизован: запустите claude в терминале и войдите в аккаунт")
        raise RuntimeError(f"Claude Code: {text[:500] or 'не вернул результат'}")

    return schema.model_validate(result["structured_output"])
