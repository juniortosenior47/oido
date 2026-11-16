# oido-db

PostgreSQL local para sesiones de audio, métricas, transcripciones, correcciones y aprendizaje.

## Inicio

1. Copie `.env.example` a `.env` y defina una contraseña propia.
2. Ejecute `docker compose up -d`.
3. Use esta URL desde los otros módulos:

```text
postgresql://oido_app:SU_CLAVE@localhost:5432/oido
```

`schema.sql` se ejecuta únicamente al crear el volumen por primera vez. Para reiniciar los datos de desarrollo: `docker compose down -v` (borra todos los datos).

Si la base ya existía antes de añadir el modelo ASR propio, aplique la migración:

```powershell
Get-Content .\migrations\002_own_asr_validator.sql | docker compose exec -T postgres psql -U oido_app -d oido
```

Los `weight_snapshots` guardan pesos de la política de comparación y calidad. No son pesos internos del modelo Whisper; para ajustar un modelo ASR se requiere un proceso de fine-tuning separado y datos de audio transcritos.
