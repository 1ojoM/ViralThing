"""Viral Remix — веб-сервер прототипа.

Запуск: .venv/bin/uvicorn app:app --port 8000
"""

import os
import shutil
import tempfile
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_env()

import llm  # noqa: E402  (после загрузки .env — модуль читает REMIX_MODEL)
import video  # noqa: E402

MAX_UPLOAD = 200 * 1024 * 1024

app = FastAPI(title="Viral Remix")


def _api_error(e: Exception) -> HTTPException:
    if isinstance(e, anthropic.AuthenticationError):
        return HTTPException(401, "Неверный ключ Claude API. Проверьте ANTHROPIC_API_KEY в файле .env")
    if isinstance(e, anthropic.RateLimitError):
        return HTTPException(429, "Превышен лимит запросов к Claude API — подождите минуту")
    if isinstance(e, anthropic.APIStatusError):
        return HTTPException(502, f"Ошибка Claude API ({e.status_code}): {e.message}")
    if isinstance(e, anthropic.APIConnectionError):
        return HTTPException(502, "Нет соединения с Claude API")
    if isinstance(e, (RuntimeError, ValueError)):
        return HTTPException(422, str(e))
    if "api_key" in str(e) or "authentication" in str(e).lower():
        return HTTPException(401, "Не задан ключ Claude API. Добавьте ANTHROPIC_API_KEY в файл .env")
    return HTTPException(500, f"Внутренняя ошибка: {e}")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), speech: bool = Form(True)):
    tmpdir = tempfile.mkdtemp(prefix="remix_")
    try:
        path = os.path.join(tmpdir, "input" + Path(file.filename or "v.mp4").suffix)
        size = 0
        with open(path, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Файл больше 200 МБ")
                out.write(chunk)

        try:
            duration, scenes = video.split_scenes(path)
        except Exception as e:
            raise HTTPException(422, f"Не удалось прочитать видео: {e}")
        transcript = video.transcribe(path) if speech else ""

        try:
            formula = llm.analyze(duration, scenes, transcript)
        except Exception as e:
            raise _api_error(e)

        return {
            "duration": round(duration, 2),
            "transcript": transcript,
            "scenes": [
                {
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "thumb": video.b64(s.frames[len(s.frames) // 2][1]),
                }
                for s in scenes
            ],
            "formula": formula.model_dump(),
        }
    finally:
        # Видео не храним — только результат анализа
        shutil.rmtree(tmpdir, ignore_errors=True)


class PromptRequest(BaseModel):
    formula: dict
    topic: str
    wishes: str = ""


@app.post("/api/prompts")
def prompts(req: PromptRequest):
    if not req.topic.strip():
        raise HTTPException(422, "Введите тему нового видео")
    try:
        pack = llm.make_prompts(req.formula, req.topic.strip(), req.wishes.strip())
    except Exception as e:
        raise _api_error(e)
    return pack.model_dump()
