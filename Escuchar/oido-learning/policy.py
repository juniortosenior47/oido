"""Política de comparación inspirada en el dilema del prisionero.

Los pesos producidos aquí ajustan decisiones de calidad y priorización de datos;
no modifican directamente los pesos neuronales de Whisper.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\wáéíóúüñ ]", "", texto.lower())).strip()


def similitud(prediccion: str, referencia: str) -> float:
    return SequenceMatcher(None, normalizar(prediccion), normalizar(referencia)).ratio()


@dataclass(frozen=True)
class Decision:
    similarity: float
    model_action: str
    reference_action: str
    payoff: float
    reward: float
    weight_delta: dict[str, float]


def evaluar(prediccion: str, referencia: str, confianza: float | None) -> Decision:
    """Calcula recompensa usando una matriz de pagos explícita.

    Cooperar del modelo significa una predicción suficientemente fiable y correcta.
    Cooperar de la referencia significa confirmar la predicción sin cambio material.
    Pagos: CC=+3, DC=+5, CD=-3, DD=-1.
    """
    score = similitud(prediccion, referencia)
    confianza = 0.0 if confianza is None else max(0.0, min(1.0, confianza))
    reference_action = "cooperate" if score >= 0.98 else "defect"
    model_action = "cooperate" if score >= 0.85 and confianza >= 0.50 else "defect"
    matrix = {
        ("cooperate", "cooperate"): 3.0,
        ("defect", "cooperate"): 5.0,
        ("cooperate", "defect"): -3.0,
        ("defect", "defect"): -1.0,
    }
    payoff = matrix[(model_action, reference_action)]
    reward = payoff * (0.5 + 0.5 * score)
    # Ajustes pequeños y acotados para evitar inestabilidad.
    delta = {
        "similarity": round(0.02 * (score - 0.5), 4),
        "confidence": round(0.01 * (confianza - 0.5), 4),
        "history": round(0.01 * (1.0 if reference_action == "cooperate" else -1.0), 4),
    }
    return Decision(score, model_action, reference_action, payoff, reward, delta)
