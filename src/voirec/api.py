"""FastAPI HTTP API for voirec transcription service."""

import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .transcribers import (
    GigaAmTranscriber,
    ParakeetTranscriber,
    WhisperTranscriber,
    diarize_and_transcribe,
    format_dialogue,
    transcribe_channels,
)

VERSION = "0.0.1"

TRANSCRIBER_DEFAULTS = {
    "whisper": "onnx-community/whisper-large-v3-turbo",
    "gigaam": "gigaam-v3-e2e-rnnt",
    "parakeet": "nemo-parakeet-tdt-0.6b-v3",
}

TRANSCRIBER_CLASSES = {
    "whisper": WhisperTranscriber,
    "gigaam": GigaAmTranscriber,
    "parakeet": ParakeetTranscriber,
}

app = FastAPI(title="voirec API", version=VERSION)


def _load_api_keys() -> set[str] | None:
    """Load API keys from env. Returns None if auth is disabled."""
    keys_file = os.environ.get("VOIREC_API_KEYS_FILE", "").strip()
    if keys_file:
        try:
            text = Path(keys_file).read_text()
            keys = {line.strip() for line in text.splitlines() if line.strip()}
            return keys or None
        except OSError:
            pass

    raw = os.environ.get("VOIREC_API_KEYS", "").strip()
    if raw:
        keys = {k.strip() for k in raw.split(",") if k.strip()}
        return keys or None

    return None


async def verify_auth(request: Request) -> None:
    keys = _load_api_keys()
    if keys is None:
        return

    token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    if token is None:
        token = request.headers.get("X-API-Key")

    if not token or token not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}


@app.get("/models", dependencies=[Depends(verify_auth)])
async def models():
    return {
        "whisper": {
            "default": "onnx-community/whisper-large-v3-turbo",
            "description": "Universal ASR (99 languages), any HuggingFace ONNX Whisper model",
        },
        "gigaam": {
            "default": "gigaam-v3-e2e-rnnt",
            "available": ["gigaam-v3-ctc", "gigaam-v3-rnnt", "gigaam-v3-e2e-ctc", "gigaam-v3-e2e-rnnt"],
            "description": "Russian ASR via ONNX",
        },
        "parakeet": {
            "default": "nemo-parakeet-tdt-0.6b-v3",
            "available": [
                "nemo-parakeet-tdt-0.6b-v3",
                "nemo-parakeet-tdt-0.6b-v2",
                "nemo-parakeet-ctc-0.6b-v3",
                "nemo-parakeet-rnnt-1.1b-v3",
            ],
            "description": "NVIDIA NeMo multilingual ASR via ONNX",
        },
    }


@app.post("/transcribe", dependencies=[Depends(verify_auth)])
async def transcribe(
    file: UploadFile = File(...),
    transcriber: str = Form("whisper"),
    model: str | None = Form(None),
    language: str | None = Form(None),
    textonly: bool = Form(False),
    diarize: bool = Form(False),
    num_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    transcriber_name = transcriber.lower()
    if transcriber_name not in TRANSCRIBER_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown transcriber '{transcriber}'. Choose from: {', '.join(TRANSCRIBER_CLASSES)}",
        )

    model_name = model or TRANSCRIBER_DEFAULTS[transcriber_name]

    # Preserve original file extension for ffmpeg compatibility
    original_ext = Path(file.filename or "audio").suffix or ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            instance = TRANSCRIBER_CLASSES[transcriber_name](model_name=model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e

        try:
            if diarize:
                segments = diarize_and_transcribe(
                    instance,
                    tmp_path,
                    num_speakers=num_speakers,
                    max_speakers=max_speakers,
                    language=language,
                )
                text = format_dialogue(segments)
            else:
                text = transcribe_channels(instance, tmp_path, language=language)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if textonly:
        return PlainTextResponse(text)

    return JSONResponse({
        "transcriber": transcriber_name,
        "model": model_name,
        "language": language,
        "text": text,
        "diarized": diarize,
    })


def run():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="voirec HTTP API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("voirec.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run()
