from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from alphabet import BLANK, VOCABULARY, decode_ctc
from features import log_mel
from model import OidoCTC


@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: float
    model_version: str


class OwnRecognizer:
    def __init__(self, checkpoint: str | Path | None = None):
        self.checkpoint = Path(checkpoint or Path(__file__).parent / "models" / "latest.pt")
        self.model: OidoCTC | None = None
        self.version = "untrained"
        if self.checkpoint.exists():
            data = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.model = OidoCTC(len(VOCABULARY) + 1)
            self.model.load_state_dict(data["state_dict"])
            self.model.eval()
            self.version = str(data.get("version", self.checkpoint.stem))

    def predict(self, audio: np.ndarray) -> Prediction:
        if self.model is None:
            return Prediction("", 0.0, self.version)
        waveform = torch.tensor(audio, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(log_mel(waveform).unsqueeze(0))[0]
            probabilities = logits.exp()
            confidence_values, tokens = probabilities.max(dim=-1)
        text = decode_ctc(tokens.tolist())
        non_blank = tokens != BLANK
        confidence = float(confidence_values[non_blank].mean()) if non_blank.any() else 0.0
        return Prediction(text, confidence, self.version)
