"""Aplicación local: micrófono → ASR propio; Whisper valida y PostgreSQL registra."""
from __future__ import annotations

import math
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext
import uuid
import sys
from difflib import SequenceMatcher

import numpy as np
import psycopg
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel

ASR_ROOT = Path(__file__).resolve().parents[1] / "oido-asr-core"
sys.path.insert(0, str(ASR_ROOT))
from recognizer import OwnRecognizer, Prediction

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATABASE_URL = os.environ["DATABASE_URL"]
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 5
LANGUAGE = os.getenv("LANGUAGE", "es")
INPUT_DEVICE = os.getenv("INPUT_DEVICE") or None
SAVE_AUDIO = os.getenv("SAVE_AUDIO", "true").lower() == "true"
AUDIO_DIR = ROOT / "recordings"


class Repository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def create_session(self) -> uuid.UUID:
        with psycopg.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audio_sessions (sample_rate, channels, language) VALUES (%s, 1, %s) RETURNING id",
                (SAMPLE_RATE, LANGUAGE),
            )
            return cur.fetchone()[0]

    def save_result(self, session_id: uuid.UUID, sequence: int, audio: np.ndarray,
                    own: Prediction, validator_text: str, validator_confidence: float,
                    validator_name: str) -> uuid.UUID:
        now = datetime.now(timezone.utc)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        rms_dbfs = -100.0 if rms < 1e-8 else 20 * math.log10(rms)
        audio_path = None
        if SAVE_AUDIO:
            AUDIO_DIR.mkdir(exist_ok=True)
            filename = f"{session_id}_{sequence:06d}.wav"
            path = AUDIO_DIR / filename
            sf.write(path, audio, SAMPLE_RATE)
            audio_path = str(path)
        with psycopg.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audio_segments
                (session_id, sequence_number, started_at, ended_at, rms_dbfs, audio_path)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (session_id, sequence, now, now, rms_dbfs, audio_path),
            )
            segment_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO transcriptions
                (segment_id, predicted_text, language, model_name, confidence)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (segment_id, own.text, LANGUAGE, "oido-asr-core", own.confidence),
            )
            transcription_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO validator_predictions
                (segment_id, validator_name, validator_version, predicted_text, confidence)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
                """,
                (segment_id, validator_name, None, validator_text, validator_confidence),
            )
            validator_id = cur.fetchone()[0]
            similarity = SequenceMatcher(None, own.text.lower().strip(), validator_text.lower().strip()).ratio()
            if similarity >= 0.85 and own.confidence >= 0.50:
                state = "CC"
            elif own.confidence >= 0.50 and similarity < 0.50:
                state = "CD"
            elif own.confidence < 0.50 and validator_text:
                state = "DC"
            else:
                state = "DD"
            cur.execute(
                """
                INSERT INTO comparison_rounds
                (transcription_id, validator_prediction_id, similarity, game_state, own_confidence, validator_confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (transcription_id, validator_id, similarity, state, own.confidence, validator_confidence),
            )
            return transcription_id, state

    def save_correction(self, transcription_id: uuid.UUID, text: str) -> None:
        with psycopg.connect(self.database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO corrections (transcription_id, corrected_text)
                VALUES (%s, %s)
                ON CONFLICT (transcription_id) DO UPDATE
                SET corrected_text = EXCLUDED.corrected_text, corrected_at = now()
                """,
                (transcription_id, text),
            )


class Transcriber(threading.Thread):
    def __init__(self, output: queue.Queue):
        super().__init__(daemon=True)
        self.output = output
        self.stop_event = threading.Event()
        self.repository = Repository(DATABASE_URL)
        self.session_id: uuid.UUID | None = None

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            self.output.put(("status", "Cargando modelo ASR propio..."))
            own_recognizer = OwnRecognizer()
            model_name = os.getenv("WHISPER_MODEL", "small")
            validator = WhisperModel(
                model_name,
                device=os.getenv("WHISPER_DEVICE", "cpu"),
                compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            )
            self.session_id = self.repository.create_session()
            audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)

            def callback(indata, frames, time_info, status):
                if status:
                    self.output.put(("status", f"Audio: {status}"))
                try:
                    audio_queue.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    pass

            self.output.put(("status", "Escuchando..."))
            buffer: list[np.ndarray] = []
            collected = 0
            sequence = 0
            target = SAMPLE_RATE * CHUNK_SECONDS
            with sd.InputStream(device=INPUT_DEVICE, samplerate=SAMPLE_RATE, channels=1, callback=callback):
                while not self.stop_event.is_set():
                    try:
                        block = audio_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    buffer.append(block)
                    collected += len(block)
                    if collected < target:
                        continue
                    audio = np.concatenate(buffer)
                    buffer, collected = [], 0
                    own = own_recognizer.predict(audio)
                    segments, _ = validator.transcribe(audio, language=LANGUAGE, vad_filter=True)
                    parts = list(segments)
                    validator_text = " ".join(part.text.strip() for part in parts).strip()
                    if not own.text and not validator_text:
                        continue
                    validator_confidence = float(np.mean([math.exp(part.avg_logprob) for part in parts])) if parts else 0.0
                    sequence += 1
                    transcription_id, game_state = self.repository.save_result(
                        self.session_id, sequence, audio, own, validator_text, validator_confidence, model_name
                    )
                    self.output.put(("text", own.text, validator_text, transcription_id, own.confidence, game_state))
        except Exception as error:
            self.output.put(("error", str(error)))
        finally:
            self.output.put(("stopped", "Detenido"))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Oído STT — Transcripción en vivo")
        root.geometry("900x600")
        self.events: queue.Queue = queue.Queue()
        self.worker: Transcriber | None = None
        self.latest_id: uuid.UUID | None = None

        self.status = tk.StringVar(value="Listo")
        tk.Label(root, textvariable=self.status, anchor="w").pack(fill="x", padx=12, pady=8)
        self.text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 14))
        self.text.pack(fill="both", expand=True, padx=12, pady=6)
        controls = tk.Frame(root)
        controls.pack(fill="x", padx=12, pady=6)
        self.start_button = tk.Button(controls, text="Iniciar micrófono", command=self.start)
        self.start_button.pack(side="left")
        tk.Button(controls, text="Detener", command=self.stop).pack(side="left", padx=6)
        self.correction = tk.Entry(controls, font=("Segoe UI", 11))
        self.correction.pack(side="left", fill="x", expand=True, padx=8)
        tk.Button(controls, text="Guardar corrección", command=self.save_correction).pack(side="right")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll()

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.worker = Transcriber(self.events)
        self.worker.start()
        self.start_button.configure(state="disabled")

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()

    def save_correction(self) -> None:
        text = self.correction.get().strip()
        if not self.latest_id or not text:
            return
        try:
            Repository(DATABASE_URL).save_correction(self.latest_id, text)
            self.status.set("Corrección guardada; ejecute oido-learning para procesarla.")
            self.correction.delete(0, tk.END)
        except Exception as error:
            messagebox.showerror("No se pudo guardar", str(error))

    def poll(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind, *values = event
            if kind == "text":
                own_text, validator_text, self.latest_id, confidence, game_state = values
                shown_text = own_text or "[El modelo propio aún no tiene entrenamiento]"
                self.text.insert(tk.END, f"PROPIO: {shown_text}\nVALIDADOR: {validator_text}\nEstado: {game_state}\n\n")
                self.text.see(tk.END)
                self.status.set(f"Modelo propio — confianza: {confidence:.0%}; comparación: {game_state}")
            elif kind == "error":
                self.status.set("Error de captura o base de datos")
                messagebox.showerror("Oído STT", values[0])
            else:
                self.status.set(values[0])
                if kind == "stopped":
                    self.start_button.configure(state="normal")
        self.root.after(100, self.poll)

    def close(self) -> None:
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    App(window)
    window.mainloop()
