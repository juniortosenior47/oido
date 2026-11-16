"""Entrena OidoCTC solamente con audio y correcciones humanas verificadas."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

from alphabet import VOCABULARY, encode
from features import log_mel
from model import OidoCTC

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


class CorrectedAudio(Dataset):
    def __init__(self, database_url: str):
        with psycopg.connect(database_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.audio_path, c.corrected_text
                FROM corrections c
                JOIN transcriptions t ON t.id = c.transcription_id
                JOIN audio_segments s ON s.id = t.segment_id
                WHERE s.audio_path IS NOT NULL AND c.corrected_text <> ''
                ORDER BY c.corrected_at
                """
            )
            self.rows = cur.fetchall()

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        path, text = self.rows[index]
        audio, rate = sf.read(path, dtype="float32")
        if rate != 16_000:
            raise ValueError(f"{path}: se esperaba 16000 Hz y llegó {rate}")
        return log_mel(torch.tensor(audio)), torch.tensor(encode(text), dtype=torch.long)


def collate(batch):
    features, labels = zip(*batch)
    return pad_sequence(features, batch_first=True), pad_sequence(labels, batch_first=True), \
        torch.tensor([len(x) for x in features]), torch.tensor([len(x) for x in labels])


def train(epochs: int, batch_size: int):
    database_url = os.environ["DATABASE_URL"]
    dataset = CorrectedAudio(database_url)
    if len(dataset) < 20:
        raise RuntimeError("Se requieren al menos 20 audios con corrección humana antes de entrenar.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = OidoCTC(len(VOCABULARY) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    objective = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for features, labels, input_lengths, target_lengths in loader:
            logits = model(features).transpose(0, 1)
            loss = objective(logits, labels, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"Época {epoch + 1}/{epochs}; pérdida CTC={total / len(loader):.4f}")
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "version": "oido-ctc-v1"}, model_dir / "latest.pt")
    print("Modelo propio guardado en models/latest.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)
