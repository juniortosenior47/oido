"""Procesa correcciones nuevas y registra sus eventos y pesos en PostgreSQL."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from policy import evaluar

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATABASE_URL = os.environ["DATABASE_URL"]
POLICY_VERSION = os.getenv("POLICY_VERSION", "pd-v1")


def clamp(value: float) -> float:
    return round(max(0.05, min(3.0, value)), 4)


def procesar(limite: int) -> int:
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, t.id, t.predicted_text, c.corrected_text, t.confidence
                FROM corrections c
                JOIN transcriptions t ON t.id = c.transcription_id
                LEFT JOIN learning_events e ON e.correction_id = c.id
                WHERE e.id IS NULL
                ORDER BY c.corrected_at
                LIMIT %s
                """,
                (limite,),
            )
            pendientes = cur.fetchall()
            if not pendientes:
                return 0

            cur.execute(
                "SELECT weights FROM weight_snapshots WHERE is_active ORDER BY created_at DESC LIMIT 1"
            )
            fila = cur.fetchone()
            pesos = dict(fila[0]) if fila else {"similarity": 1.0, "confidence": 0.4, "history": 0.2}

            for correction_id, transcription_id, prediccion, referencia, confianza in pendientes:
                decision = evaluar(prediccion, referencia, confianza)
                cur.execute(
                    """
                    INSERT INTO learning_events
                    (correction_id, predicted_text, reference_text, similarity, confidence,
                     model_action, reference_action, payoff, reward, weight_delta, policy_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (correction_id, prediccion, referencia, decision.similarity, confianza,
                     decision.model_action, decision.reference_action, decision.payoff,
                     decision.reward, json.dumps(decision.weight_delta), POLICY_VERSION),
                )
                cur.execute(
                    "UPDATE comparison_rounds SET is_human_verified = TRUE WHERE transcription_id = %s",
                    (transcription_id,),
                )
                for name, delta in decision.weight_delta.items():
                    pesos[name] = clamp(float(pesos.get(name, 0.0)) + delta)

            cur.execute("UPDATE weight_snapshots SET is_active = FALSE WHERE is_active")
            cur.execute(
                "INSERT INTO weight_snapshots (policy_version, weights, evaluation) VALUES (%s, %s::jsonb, %s::jsonb)",
                (POLICY_VERSION, json.dumps(pesos), json.dumps({"events_processed": len(pendientes)})),
            )
        conn.commit()
    return len(pendientes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(f"Eventos procesados: {procesar(args.limit)}")
