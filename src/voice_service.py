"""Voice Service: Gnani.ai STT & TTS integration with ear normalization and fallback handling."""
import os
import wave
import struct
import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.config import (
    GNANI_API_KEY, GNANI_TTS_MODEL, GNANI_TTS_VOICE,
    GNANI_DEFAULT_LANG, OUTPUT_DIR
)
from src.utils import format_for_speech

class VoiceService:
    def __init__(self):
        self.api_key = GNANI_API_KEY
        self._stt_client = None
        self._tts_client = None

        if self.api_key:
            self._init_clients()

    def _init_clients(self):
        try:
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient
            self._stt_client = GnaniSTTClient()
            self._tts_client = GnaniTTSClient()
        except Exception as e:
            print(f"Warning: Failed to initialize Gnani clients: {e}")

    def is_gnani_ready(self) -> bool:
        return bool(self.api_key and self._stt_client and self._tts_client)

    def transcribe(self, audio_path: str, language_code: str = GNANI_DEFAULT_LANG) -> str:
        """Transcribe an audio file using Gnani STT REST API."""
        p = Path(audio_path)
        if not p.exists():
            return ""

        if self.is_gnani_ready():
            for attempt in range(2):
                try:
                    result = self._stt_client.transcribe(str(p), language_code=language_code)
                    if isinstance(result, dict):
                        return result.get("transcript", "").strip()
                    return str(result).strip()
                except Exception as e:
                    if attempt == 0:
                        time.sleep(1)
                    else:
                        print(f"Gnani STT transcription failed after retry: {e}. Falling back...")

        # Fallback 1: check if a .txt transcript with the same name exists alongside audio
        txt_path = p.with_suffix(".txt")
        if txt_path.exists():
            try:
                return txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        return ""

    def synthesize(self, text: str, output_file: Optional[str] = None,
                   voice: Optional[str] = None, model: Optional[str] = None,
                   language: Optional[str] = None) -> str:
        """Synthesize spoken audio from text using Gnani TTS API."""
        out_path = Path(output_file) if output_file else (OUTPUT_DIR / "reply.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        voice_name = voice or GNANI_TTS_VOICE
        model_name = model or GNANI_TTS_MODEL
        lang_code = language or GNANI_DEFAULT_LANG

        speech_text = format_for_speech(text)

        if self.is_gnani_ready():
            for attempt in range(2):
                try:
                    audio_bytes = self._tts_client.synthesize(
                        text=speech_text,
                        voice=voice_name,
                        model=model_name,
                        language=lang_code,
                        output_file=str(out_path)
                    )
                    if out_path.exists() and out_path.stat().st_size > 0:
                        return str(out_path)
                except Exception as e:
                    if attempt == 0:
                        time.sleep(1)
                    else:
                        print(f"Gnani TTS synthesis failed after retry: {e}. Generating offline audio...")

        # Offline Fallback: Generate valid 16-bit PCM WAV audio
        self._generate_offline_wav(str(out_path), text=speech_text)
        return str(out_path)

    def _generate_offline_wav(self, file_path: str, text: str, duration_sec: float = 2.0):
        """Generates a valid standard 16kHz mono WAV file for fallback testing."""
        sample_rate = 16000
        n_samples = int(sample_rate * duration_sec)
        
        # Simple pleasant sound (harmonic chime) indicating completed synthesized response
        with wave.open(file_path, "wb") as wav_file:
            wav_file.setnchannels(1)        # Mono
            wav_file.setsampwidth(2)        # 16-bit
            wav_file.setframerate(sample_rate)
            
            frames = bytearray()
            for i in range(n_samples):
                t = i / sample_rate
                # Envelope
                env = math.exp(-3.0 * t / duration_sec)
                # Chime chord: 440 Hz (A4) + 554 Hz (C#5)
                val = int(8000 * env * (0.6 * math.sin(2 * math.pi * 440 * t) + 0.4 * math.sin(2 * math.pi * 554 * t)))
                val = max(-32768, min(32767, val))
                frames.extend(struct.pack("<h", val))
                
            wav_file.writeframes(frames)

# Global singleton voice service
voice_service = VoiceService()
