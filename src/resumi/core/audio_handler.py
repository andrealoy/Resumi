"""Audio recording and Whisper transcription logic."""
from __future__ import annotations
 
import os
import queue
import tempfile
import threading
from datetime import datetime
 
import numpy as np
from scipy.io.wavfile import write as wav_write
 
# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_SECONDS  = 2
SILENCE_THRESH = 0.005
LANGUAGE       = "fr"      # None = auto-detect
MODEL_SIZE     = "small"   # tiny | base | small | medium | large
# ──────────────────────────────────────────
 
# ── Lazy Whisper loader ───────────────────
_whisper_model = None
_whisper_lock  = threading.Lock()
 
 
def get_whisper():
    """Load and cache the Whisper model (thread-safe)."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                import whisper
                _whisper_model = whisper.load_model(MODEL_SIZE)
    return _whisper_model
 
 
# ── Recording state ───────────────────────
_audio_queue:   queue.Queue          = queue.Queue()
_stop_event:    threading.Event      = threading.Event()
_record_thread: threading.Thread | None = None
_is_recording   = False
 
 
def is_recording() -> bool:
    return _is_recording
 
 
def start_recording() -> str:
    """Start microphone capture. Returns a status message."""
    global _is_recording, _record_thread
 
    if _is_recording:
        return "Déjà en cours…"
 
    _stop_event.clear()
 
    # Drain stale audio
    while not _audio_queue.empty():
        _audio_queue.get_nowait()
 
    _is_recording = True
    _record_thread = threading.Thread(target=_record_loop, daemon=True)
    _record_thread.start()
    return "Enregistrement en cours…"
 
 
def stop_recording() -> tuple[str, str]:
    """Stop capture and transcribe buffered audio.
 
    Returns:
        (status_message, transcript_text)
    """
    global _is_recording
 
    if not _is_recording:
        return "Pas d'enregistrement en cours.", ""
 
    _stop_event.set()
    _is_recording = False
 
    # Collect all buffered chunks
    chunks: list[np.ndarray] = []
    while not _audio_queue.empty():
        chunks.append(_audio_queue.get())
 
    if not chunks:
        return "Aucun audio capturé.", ""
 
    lines: list[str] = []
    for chunk in chunks:
        text = _transcribe_chunk(chunk)
        if text:
            ts = datetime.now().strftime("%H:%M:%S")
            lines.append(f"[{ts}] {text}")
 
    if not lines:
        return "Terminé.", "(silence détecté)"
 
    return "Transcription terminée.", "\n".join(lines)
 
 
def transcribe_file(path: str) -> tuple[str, str]:
    """Transcribe an audio or video file.
 
    Args:
        path: Absolute path to the file.
 
    Returns:
        (status_message, transcript_text)
    """
    if not path:
        return "Aucun fichier fourni.", ""
    try:
        opts = {"language": LANGUAGE} if LANGUAGE else {}
        result = get_whisper().transcribe(path, fp16=False, **opts)
        text = result["text"].strip()
        return "Transcription terminée.", text or "(aucun contenu détecté)"
    except Exception as exc:
        return f"Erreur : {exc}", ""
 
 
# ── Internal helpers ──────────────────────
 
def _record_loop() -> None:
    import sounddevice as sd
 
    samples = CHUNK_SECONDS * SAMPLE_RATE
 
    def callback(indata, frames, t, status):
        if not _stop_event.is_set():
            _audio_queue.put(indata.copy())
 
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=samples,
        callback=callback,
    ):
        _stop_event.wait()
 
 
def _transcribe_chunk(chunk: np.ndarray) -> str | None:
    chunk = chunk.flatten()
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < SILENCE_THRESH:
        return None
 
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
 
    wav_write(tmp_path, SAMPLE_RATE, (chunk * 32767).astype(np.int16))
    try:
        opts = {"language": LANGUAGE} if LANGUAGE else {}
        result = get_whisper().transcribe(tmp_path, fp16=False, **opts)
        return result["text"].strip() or None
    finally:
        os.remove(tmp_path)