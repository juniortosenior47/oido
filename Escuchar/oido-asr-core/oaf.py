"""Formato OAF v1: una secuencia compacta de características acústicas."""
from __future__ import annotations
import numpy as np

FEATURE_VERSION = "oaf-v1"
FRAME = 240  # 15 ms a 16 kHz
N_BANDS = 16

def extract(audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    rows = []
    for start in range(0, max(1, len(audio) - FRAME + 1), FRAME):
        x = audio[start:start + FRAME]
        if len(x) < FRAME: x = np.pad(x, (0, FRAME - len(x)))
        rms = np.sqrt(np.mean(x * x))
        zcr = np.mean(x[:-1] * x[1:] < 0)
        mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-9
        hz = np.fft.rfftfreq(len(x), 1 / sample_rate)
        centroid = (hz * mag).sum() / mag.sum()
        peak = hz[np.argmax(mag)]
        rolloff = hz[np.searchsorted(np.cumsum(mag), mag.sum() * .85)]
        bands = [np.log1p(group.mean()) for group in np.array_split(mag, N_BANDS)]
        rows.append([20 * np.log10(rms + 1e-9), zcr, centroid, peak, rolloff, *bands])
    return np.asarray(rows, dtype=np.float32)

def resample(sequence: np.ndarray, frames: int = 40) -> np.ndarray:
    old = np.linspace(0, 1, len(sequence)); new = np.linspace(0, 1, frames)
    return np.stack([np.interp(new, old, sequence[:, i]) for i in range(sequence.shape[1])], axis=1)

def distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = resample(a), resample(b)
    scale = np.std(np.vstack([a, b]), axis=0) + 1e-5
    return float(np.mean(np.abs((a - b) / scale)))
