"""Разбор видео: склейки, ключевые кадры, распознавание речи.

Всё на PyAV (бандлит свой FFmpeg), системный ffmpeg не нужен.
"""

import base64
import io
import os
from dataclasses import dataclass, field

import av
import numpy as np

MAX_DURATION = 180.0      # анализируем не больше 3 минут
SAMPLE_EVERY = 0.2        # как часто сохранять кадр-кандидат (сек)
CUT_THRESHOLD = 30.0      # порог разницы кадров для склейки (0–255)
MIN_SCENE_LEN = 0.5       # сцены короче склеиваем с соседней
MAX_SCENES = 12
MAX_FRAMES = 24           # всего кадров, отправляемых в модель
THUMB_MAX_SIDE = 640


@dataclass
class Scene:
    start: float
    end: float
    frames: list = field(default_factory=list)  # [(t, jpeg_bytes)]

    @property
    def duration(self) -> float:
        return self.end - self.start


def _to_jpeg(frame: av.VideoFrame) -> bytes:
    img = frame.to_image()
    img.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _signature(frame: av.VideoFrame) -> np.ndarray:
    # Маленькая копия кадра — сравниваем цвет и яркость, шум почти не влияет
    small = frame.reformat(width=48, height=48, format="rgb24")
    return small.to_ndarray().astype(np.int16)


def scan(path: str) -> tuple[float, list[float], list[tuple[float, bytes]]]:
    """Один проход по видео: длительность, моменты склеек, кадры-кандидаты."""
    cuts: list[float] = []
    samples: list[tuple[float, bytes]] = []
    prev_sig = None
    last_cut = 0.0
    next_sample = 0.0
    t = 0.0

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("В файле нет видеодорожки")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if t > MAX_DURATION:
                break
            sig = _signature(frame)
            if prev_sig is not None:
                diff = float(np.abs(sig - prev_sig).mean())
                if diff > CUT_THRESHOLD and t - last_cut >= MIN_SCENE_LEN:
                    cuts.append(t)
                    last_cut = t
                    samples.append((t, _to_jpeg(frame)))  # первый кадр новой сцены
                    next_sample = t + SAMPLE_EVERY
            prev_sig = sig
            if t >= next_sample:
                samples.append((t, _to_jpeg(frame)))
                next_sample = t + SAMPLE_EVERY

    return t, cuts, samples


def _merge_to_limit(bounds: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Если сцен слишком много — сливаем самую короткую с соседом."""
    bounds = list(bounds)
    while len(bounds) > MAX_SCENES:
        i = min(range(len(bounds)), key=lambda k: bounds[k][1] - bounds[k][0])
        j = i - 1 if i > 0 else i + 1
        a, b = sorted((i, j))
        bounds[a:b + 1] = [(bounds[a][0], bounds[b][1])]
    return bounds


def _pick_times(start: float, end: float) -> list[float]:
    d = end - start
    if d < 2.5:
        return [start + d * 0.5]
    if d < 6:
        return [start + d * 0.2, start + d * 0.8]
    return [start + d * 0.1, start + d * 0.5, start + d * 0.9]


def split_scenes(path: str) -> tuple[float, list[Scene]]:
    duration, cuts, samples = scan(path)
    if not samples:
        raise ValueError("Не удалось прочитать кадры видео")

    edges = [0.0] + cuts + [duration]
    bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    bounds = [b for b in bounds if b[1] - b[0] > 0.05] or [(0.0, duration)]
    bounds = _merge_to_limit(bounds)

    scenes = []
    for start, end in bounds:
        scene = Scene(start, end)
        inside = [s for s in samples if start <= s[0] < end] or samples
        for want in _pick_times(start, end):
            best = min(inside, key=lambda s: abs(s[0] - want))
            if best not in scene.frames:
                scene.frames.append(best)
        scenes.append(scene)

    # Общий лимит кадров: у длинных сцен оставляем только средний кадр
    while sum(len(s.frames) for s in scenes) > MAX_FRAMES:
        longest = max(scenes, key=lambda s: len(s.frames))
        if len(longest.frames) == 1:
            break
        longest.frames = [longest.frames[len(longest.frames) // 2]]

    return duration, scenes


_whisper = None


def transcribe(path: str) -> str:
    """Речь с таймкодами. Пустая строка, если речи/звука нет."""
    global _whisper
    try:
        if _whisper is None:
            from faster_whisper import WhisperModel

            name = os.environ.get("WHISPER_MODEL", "small")
            _whisper = WhisperModel(name, device="cpu", compute_type="int8")
        segments, _ = _whisper.transcribe(path, vad_filter=True)
        lines = [
            f"[{seg.start:.1f}–{seg.end:.1f}] {seg.text.strip()}"
            for seg in segments
            if seg.start <= MAX_DURATION
        ]
        return "\n".join(lines)
    except Exception as e:  # нет аудиодорожки, не скачалась модель и т.п.
        print(f"[transcribe] пропущено: {e}")
        return ""


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")
