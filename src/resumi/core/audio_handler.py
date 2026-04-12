"""Audio handler – placeholder for Whisper transcription pipeline."""


class AudioHandler:
    """Transcribe audio into text. TODO: integrate Whisper."""

    async def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError("Audio transcription not yet implemented")
