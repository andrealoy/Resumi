"""Audio recording and OpenAI API transcription logic."""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import numpy as np
from scipy.io.wavfile import write as wav_write
from openai import OpenAI
from dotenv import load_dotenv

# Initialisation du client (vérifie que OPENAI_API_KEY est dans ton .env)
load_dotenv()
client = OpenAI()

# ──────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_SECONDS  = 0.5
LANGUAGE       = "fr"

# ── State ─────────────────────────────────
_audio_queue:   queue.Queue = queue.Queue()
_stop_event:    threading.Event = threading.Event()
_record_thread: threading.Thread | None = None
_is_recording   = False

def _api_transcribe(file_path: str) -> str:
    """Helper interne pour appeler l'API OpenAI."""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=LANGUAGE
        )
    return transcript.text.strip()

def is_recording() -> bool:
    return _is_recording

def start_recording() -> str:
    global _is_recording, _record_thread
    if _is_recording:
        return "Déjà en cours…"
    
    _stop_event.clear()
    while not _audio_queue.empty():
        _audio_queue.get_nowait()

    _is_recording = True
    _record_thread = threading.Thread(target=_record_loop, daemon=True)
    _record_thread.start()
    return "Enregistrement en cours…"

def stop_recording() -> tuple[str, str]:
    global _is_recording
    if not _is_recording:
        return "Pas d'enregistrement en cours.", ""

    _stop_event.set()
    _is_recording = False
    
    if _record_thread:
        _record_thread.join(timeout=2)

    chunks: list[np.ndarray] = []
    while not _audio_queue.empty():
        chunks.append(_audio_queue.get())

    if not chunks:
        return "Aucun audio capturé.", ""

    full_audio = np.concatenate(chunks).flatten()
    
    # Normalisation pour éviter la saturation
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio = full_audio / peak

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Sauvegarde temporaire du vocal micro
        wav_write(tmp_path, SAMPLE_RATE, (full_audio * 32767).astype(np.int16))
        
        # Appel API
        transcript = _api_transcribe(tmp_path)
        
        if not transcript:
            return "Terminé.", "(silence détecté)"
            
        return "Transcription terminée.", transcript

    except Exception as exc:
        return f"Erreur API : {exc}", ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def transcribe_file(path: str) -> tuple[str, str]:
    """Transcrit un fichier existant (audio ou vidéo) via l'API."""
    if not path or not os.path.exists(path):
        return "Fichier introuvable.", ""
    
    try:
        transcript = _api_transcribe(path)
        return "Transcription terminée.", transcript or "(aucun contenu détecté)"
    except Exception as exc:
        return f"Erreur API : {exc}", ""

# ── Internal loop ──────────────────────────

def _record_loop() -> None:
    import sounddevice as sd
    samples = int(CHUNK_SECONDS * SAMPLE_RATE)

    def callback(indata, frames, t, status):
        if not _stop_event.is_set():
            _audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=samples,
            callback=callback,
        ):
            _stop_event.wait()
    except Exception:
        global _is_recording
        _is_recording = False