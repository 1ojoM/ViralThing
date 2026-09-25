"""Запросы к Claude: разбор формулы видео и генерация промптов для Kling."""

import json
import os
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

import claude_cli
from video import Scene, b64

BACKEND = os.environ.get("VIRALTHING_BACKEND", "api")  # api | subscription
MODEL = os.environ.get("VIRALTHING_MODEL", "claude-opus-5")
EFFORT = os.environ.get("VIRALTHING_EFFORT", "high")

_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ---------- Схемы ответов ----------

class SceneBreakdown(BaseModel):
    index: int
    start: float
    end: float
    role: str = Field(description="Роль сцены в структуре: хук, завязка, нарастание, развязка, loop…")
    shot: str = Field(description="Крупность и ракурс: крупный план, общий, POV, сверху…")
    camera: str = Field(description="Движение камеры: статика, наезд, облёт, следование…")
    action: str = Field(description="Что происходит в кадре, одно предложение")


class Formula(BaseModel):
    summary: str = Field(description="О чём видео, 1–2 предложения")
    hook: str = Field(description="Что именно происходит в первые 1–2 секунды")
    hook_why: str = Field(description="Почему хук останавливает скролл")
    structure: str = Field(description="Драматургия: как развивается ролик от начала до конца")
    pacing: str = Field(description="Темп: длина сцен, ритм склеек, ускорения")
    visual_style: str = Field(description="Визуальный стиль: реализм/анимация, свет, палитра, камера")
    audio: str = Field(description="Роль звука: озвучка, музыка, звуки, текст на экране")
    viral_factors: list[str] = Field(description="3–5 причин, почему видео набирает просмотры")
    scenes: list[SceneBreakdown]


class Clip(BaseModel):
    index: int
    source_scenes: list[int] = Field(description="Номера сцен оригинала, которые повторяет клип")
    role: str = Field(description="Роль клипа в структуре, по-русски")
    duration: Literal[5, 10] = Field(description="Длительность генерации в Kling")
    trim_to: float = Field(description="Сколько секунд оставить при монтаже, по ритму оригинала")
    prompt: str = Field(description="Промпт для Kling на английском")
    prompt_ru: str = Field(description="Перевод промпта на русский для пользователя")


class PromptPack(BaseModel):
    concept: str = Field(description="Идея нового ролика, 1–2 предложения по-русски")
    character: str = Field(description="Якорное описание героя на английском (повторяется дословно в промптах) или пустая строка")
    style: str = Field(description="Короткий стиль на английском, одинаковый для всех клипов")
    clips: list[Clip]
    negative_prompt: str = Field(description="Негативный промпт для Kling на английском, через запятую")
    tips: list[str] = Field(description="2–4 совета по монтажу по-русски")


# ---------- Промпты ----------

ANALYZE_SYSTEM = """Ты — аналитик вирусных коротких видео (TikTok, Reels, Shorts), особенно созданных ИИ.
Тебе дают раскадровку ролика: сцены с таймкодами, ключевые кадры и, если есть, расшифровку речи.
Твоя задача — извлечь переносимую «формулу» ролика: структуру, хук, темп, работу камеры и стиль.
Описывай механику, а не сюжет: формулу потом применят к совершенно другой теме.
Описывай только то, что видно на кадрах; если по кадрам что-то неясно (например, движение камеры) — делай разумный вывод по разнице между кадрами одной сцены.
Все текстовые поля — на русском, кратко и конкретно."""

KLING_SYSTEM = """Ты пишешь промпты для генерации видео в Kling (text-to-video, базовые модели).
Задача: по «формуле» вирусного ролика и новой теме пользователя собрать новый ролик из коротких клипов, который повторяет структуру, хук, темп и работу камеры оригинала, но с другим сюжетом, героями и объектами. Ничего не копируй дословно из оригинала: ни персонажей, ни брендов, ни узнаваемых сцен.

Правила промпта для Kling — самое важное. Kling лучше всего следует коротким простым промптам, поэтому:
1. Один клип = один план = одно главное действие. Никаких «then», «after that», смен сцены внутри клипа.
2. Формула: [герой] + [действие] + [место] + [камера] + [свет/стиль]. 1–2 предложения, обычно 20–30 слов, строго не больше 35. Лучше выкинуть второстепенную деталь, чем превысить лимит.
3. Только конкретные видимые вещи: существительные и глаголы, которые можно снять. Без метафор, эмоций-абстракций и оценок.
4. Запрещены слова-пустышки: viral, stunning, masterpiece, 8k, 4k, ultra detailed, hyperrealistic, award-winning, trending, best quality.
5. Не больше одного движения камеры на клип, простыми терминами: static shot, slow push in, pull back, pan left/right, tracking shot, orbit around, handheld, low angle, top-down view, close-up, wide shot.
6. Не больше двух персонажей в кадре. Никакого текста, надписей, логотипов и цифр в кадре.
7. Нельзя писать отрицания («no people», «without text») в основном промпте — всё нежелательное только в negative_prompt.
8. Консистентность: если есть герой, опиши его одной короткой якорной фразой (5–12 слов, внешность + одежда/цвет) в поле character и вставляй её ДОСЛОВНО в каждый промпт, где он есть. Поле style (3–8 слов) дословно заканчивает каждый промпт.
9. Соотношение сторон 9:16 и длительность задаются настройками Kling — не пиши их в промпт.
10. duration: 5 для всех клипов, кроме длинных непрерывных планов > 6 сек (тогда 10). trim_to — длительность соответствующей сцены оригинала (сколько оставить при монтаже), но не больше duration.
11. Если сцен в оригинале больше 8 — объедини похожие соседние сцены, чтобы клипов было не больше 8. Первый клип — всегда хук, он должен быть визуально самым сильным и понятным без контекста.
12. negative_prompt — 6–12 коротких пунктов через запятую (например: text, watermark, blurry, distorted face, extra fingers, morphing, flicker).

Если пользователь дал пожелания по герою или стилю — используй их. Тексты для пользователя (concept, role, prompt_ru, tips) — по-русски."""


# ---------- Вызовы ----------

def _call(system: str, content: list, schema: type[BaseModel]):
    if BACKEND == "subscription":
        return claude_cli.call(system, content, schema, EFFORT)
    response = client().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=system,
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        output_format=schema,
        # Если классификатор безопасности отклонит запрос, API сам повторит его
        # на рекомендованной резервной модели.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Модель отказалась обрабатывать это видео")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise RuntimeError("Ответ модели оборвался — попробуйте ещё раз")
    return response.parsed_output


def analyze(duration: float, scenes: list[Scene], transcript: str) -> Formula:
    avg = duration / max(len(scenes), 1)
    content: list = [{
        "type": "text",
        "text": f"Длительность ролика: {duration:.1f} с. Сцен: {len(scenes)}. "
                f"Средняя длина сцены: {avg:.1f} с. Ниже — сцены по порядку.",
    }]
    for i, scene in enumerate(scenes, 1):
        content.append({
            "type": "text",
            "text": f"Сцена {i}: {scene.start:.1f}–{scene.end:.1f} с ({scene.duration:.1f} с)",
        })
        for t, jpeg in scene.frames:
            content.append({"type": "text", "text": f"кадр @ {t:.1f} с"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(jpeg)},
            })
    content.append({
        "type": "text",
        "text": ("Расшифровка речи:\n" + transcript) if transcript
        else "Речи в ролике нет или она не распознавалась.",
    })
    content.append({
        "type": "text",
        "text": "Разбери формулу этого ролика. В scenes — по одной записи на каждую сцену выше, с теми же таймкодами.",
    })
    return _call(ANALYZE_SYSTEM, content, Formula)


def make_prompts(formula: dict, topic: str, wishes: str) -> PromptPack:
    text = (
        "Формула оригинального ролика:\n"
        + json.dumps(formula, ensure_ascii=False, indent=1)
        + f"\n\nТема нового ролика: {topic}"
    )
    if wishes:
        text += f"\nПожелания по герою/стилю: {wishes}"
    return _call(KLING_SYSTEM, [{"type": "text", "text": text}], PromptPack)
