from __future__ import annotations

import torch


def log_mel(audio: torch.Tensor, sample_rate: int = 16_000, n_mels: int = 80) -> torch.Tensor:
    """Características log-Mel calculadas localmente, sin depender de Whisper."""
    if audio.ndim != 1:
        audio = audio.flatten()
    window = torch.hann_window(400, device=audio.device)
    spectrum = torch.stft(audio, n_fft=512, hop_length=160, win_length=400,
                          window=window, return_complex=True).abs().square()
    frequencies = torch.linspace(0, sample_rate / 2, spectrum.shape[0], device=audio.device)
    centers = torch.linspace(80, 7_600, n_mels, device=audio.device)
    bandwidth = max(1.0, float(centers[1] - centers[0]))
    filters = torch.clamp(1 - (frequencies[None, :] - centers[:, None]).abs() / bandwidth, min=0)
    return torch.log(filters @ spectrum + 1e-10).transpose(0, 1)
