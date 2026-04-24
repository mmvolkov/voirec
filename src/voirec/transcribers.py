"""Transcriber implementations for different models."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
import tempfile
import os


def _get_channel_count(audio_path: str) -> int:
    """Get number of audio channels via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels",
         "-of", "default=noprint_wrappers=1:nokey=1",
         audio_path],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


@contextmanager
def _as_channel_wav(audio_path: str, channel_idx: int):
    """Extract a single channel as 16kHz mono WAV."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-i", audio_path, "-ar", "16000",
             "-af", f"pan=mono|c0=c{channel_idx}", "-y", tmp.name],
            check=True, capture_output=True,
        )
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def transcribe_channels(transcriber: "BaseTranscriber", audio_path: str, language: str | None = None) -> str:
    """Transcribe audio, splitting by channels if multi-channel."""
    n = _get_channel_count(audio_path)
    if n <= 1:
        return transcriber.transcribe(audio_path, language=language)

    parts = []
    for i in range(n):
        with _as_channel_wav(audio_path, i) as wav:
            text = transcriber.transcribe(wav, language=language)
        parts.append(f"[Канал {i + 1}]: {text}")
    return "\n".join(parts)


@contextmanager
def _as_wav(audio_path: str):
    """Convert audio to 16kHz mono WAV if needed, yield path, cleanup temp file."""
    if audio_path.lower().endswith(".wav"):
        yield audio_path
        return

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", tmp.name],
            check=True, capture_output=True,
        )
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


class BaseTranscriber(ABC):
    """Base class for all transcribers."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text."""
        pass


class WhisperTranscriber(BaseTranscriber):
    """OpenAI Whisper transcriber via onnx-asr."""

    def __init__(self, model_name: str = "onnx-community/whisper-large-v3-turbo"):
        """
        Args:
            model_name: Whisper model from onnx-community HuggingFace, e.g.:
                        onnx-community/whisper-tiny, whisper-base,
                        onnx-community/whisper-large-v3-turbo
        """
        import onnx_asr

        self.model = onnx_asr.load_model(model_name)

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        with _as_wav(audio_path) as wav:
            kwargs = {"language": language} if language else {}
            results = self.model.recognize(wav, **kwargs)
        if isinstance(results, list):
            return " ".join(r if isinstance(r, str) else r.text for r in results if r).strip()
        return (results if isinstance(results, str) else results.text or "").strip()



class ParakeetTranscriber(BaseTranscriber):
    """NVIDIA NeMo Parakeet v3 multilingual transcriber via onnx-asr."""

    def __init__(self, model_name: str = "nemo-parakeet-tdt-0.6b-v3"):
        """
        Args:
            model_name: Parakeet model variant:
                        nemo-parakeet-tdt-0.6b-v3 (multilingual),
                        nemo-parakeet-tdt-0.6b-v2, nemo-parakeet-ctc-0.6b,
                        nemo-parakeet-rnnt-0.6b
        """
        import onnx_asr

        self.model = onnx_asr.load_model(model_name)

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        with _as_wav(audio_path) as wav:
            kwargs = {"language": language} if language else {}
            results = self.model.recognize(wav, **kwargs)
        if isinstance(results, list):
            return " ".join(r if isinstance(r, str) else r.text for r in results if r).strip()
        return (results if isinstance(results, str) else results.text or "").strip()


class GigaAmTranscriber(BaseTranscriber):
    """GigaAM v3 transcriber via onnx-asr (Russian ASR)."""

    def __init__(self, model_name: str = "gigaam-v3-e2e-rnnt"):
        """
        Args:
            model_name: Model variant: gigaam-v3-ctc, gigaam-v3-rnnt,
                        gigaam-v3-e2e-ctc, gigaam-v3-e2e-rnnt
                        e2e варианты включают пунктуацию и нормализацию текста
        """
        import onnx_asr

        self.model = onnx_asr.load_model(model_name)

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        with _as_wav(audio_path) as wav:
            results = self.model.recognize(wav)
        if isinstance(results, list):
            return " ".join(r if isinstance(r, str) else r.text for r in results if r).strip()
        return (results if isinstance(results, str) else results.text or "").strip()


# --- Diarization ---

@dataclass
class DiarizedSegment:
    start: float
    end: float
    speaker: str
    text: str


@contextmanager
def _extract_segment(audio_path: str, start: float, end: float):
    """Вырезать временной сегмент как 16kHz mono WAV через ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-ss", str(start), "-to", str(end),
             "-i", audio_path, "-ar", "16000", "-ac", "1", "-y", tmp.name],
            check=True, capture_output=True,
        )
        yield tmp.name
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def _diarize_by_channels(transcriber: "BaseTranscriber", audio_path: str, language: str | None = None) -> list[DiarizedSegment]:
    """Диаризация для многоканального аудио: каждый канал = отдельный спикер."""
    import onnx_asr

    vad = onnx_asr.load_vad("silero")
    adapter = transcriber.model.with_vad(vad)  # type: ignore[attr-defined]
    n = _get_channel_count(audio_path)
    kwargs = {"language": language} if language else {}
    all_segments: list[DiarizedSegment] = []
    for i in range(n):
        with _as_channel_wav(audio_path, i) as wav:
            for seg in adapter.recognize(wav, **kwargs):
                if seg.text.strip():
                    all_segments.append(DiarizedSegment(seg.start, seg.end, f"СПИКЕР_{i + 1}", seg.text.strip()))
    return sorted(all_segments, key=lambda s: s.start)


def diarize_and_transcribe(
    transcriber: "BaseTranscriber",
    audio_path: str,
    num_speakers: int | None = None,
    max_speakers: int | None = None,
    embed_model: str = "ecapa",  # kept for API compat, unused
    language: str | None = None,
) -> list[DiarizedSegment]:
    """Диаризация + транскрибация.

    Для многоканального аудио (стерео телефонные записи) использует канальное разделение.
    Для моно — resemblyzer (GE2E) + sklearn кластеризация.
    """
    n_channels = _get_channel_count(audio_path)
    if n_channels > 1 and hasattr(transcriber, "model"):
        return _diarize_by_channels(transcriber, audio_path, language=language)

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as e:
        try:
            import resemblyzer  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Установите зависимости диаризации: uv pip install -e '.[diarize]'"
            ) from e
        raise

    SR = 16000
    encoder = VoiceEncoder("cpu")
    wav = preprocess_wav(audio_path)  # float32 numpy array @ 16kHz mono
    total_dur = len(wav) / SR

    # Скользящее окно: 1.5s окно, 0.75s шаг
    win_len = int(1.5 * SR)
    hop_len = int(0.75 * SR)
    windows, starts = [], []
    for i in range(0, max(1, len(wav) - win_len + 1), hop_len):
        chunk = wav[i:i + win_len]
        if len(chunk) >= int(0.2 * SR):
            windows.append(chunk)
            starts.append(i / SR)

    if not windows:
        text = transcriber.transcribe(audio_path, language=language)
        return [DiarizedSegment(0.0, total_dur, "СПИКЕР_1", text.strip())] if text.strip() else []

    embeds = np.array([encoder.embed_utterance(w) for w in windows])

    if num_speakers is not None:
        clusterer = AgglomerativeClustering(
            n_clusters=num_speakers, metric='cosine', linkage='average'
        )
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.4,
            metric='cosine', linkage='average'
        )
    labels = clusterer.fit_predict(embeds)

    if max_speakers is not None and len(set(labels)) > max_speakers:
        clusterer = AgglomerativeClustering(
            n_clusters=max_speakers, metric='cosine', linkage='average'
        )
        labels = clusterer.fit_predict(embeds)

    # Склеиваем последовательные окна одного спикера
    merged: list[tuple[float, float, int]] = []
    seg_start = starts[0]
    seg_label = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != seg_label:
            merged.append((seg_start, starts[i], seg_label))
            seg_start = starts[i]
            seg_label = int(labels[i])
    merged.append((seg_start, total_dur, seg_label))

    result = []
    for start, end, label in merged:
        if end - start < 0.3:
            continue
        text = _transcribe_segment(transcriber, audio_path, start, end, language=language)
        if text.strip():
            result.append(DiarizedSegment(start, end, f"СПИКЕР_{label + 1}", text.strip()))
    return result


_MAX_SEG_DUR = 30.0  # секунды; GigaAM/Parakeet падают на длинных сегментах


def _transcribe_segment(transcriber: "BaseTranscriber", audio_path: str, start: float, end: float, language: str | None = None) -> str:
    """Транскрибировать сегмент, нарезая на куски если он слишком длинный."""
    dur = end - start
    if dur <= _MAX_SEG_DUR:
        with _extract_segment(audio_path, start, end) as seg_wav:
            return transcriber.transcribe(seg_wav, language=language)

    parts = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + _MAX_SEG_DUR, end)
        with _extract_segment(audio_path, chunk_start, chunk_end) as seg_wav:
            text = transcriber.transcribe(seg_wav, language=language)
            if text.strip():
                parts.append(text.strip())
        chunk_start = chunk_end
    return " ".join(parts)


def format_dialogue(segments: list[DiarizedSegment]) -> str:
    """Отформатировать сегменты в диалог с тайм-кодами."""
    lines = []
    for seg in segments:
        h = int(seg.start // 3600)
        m = int((seg.start % 3600) // 60)
        s = int(seg.start % 60)
        ts = f"{h:02d}:{m:02d}:{s:02d}"
        lines.append(f"[{ts}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)
