# oido-stt

Aplicación local con interfaz gráfica que usa **oido-asr-core** como transcriptor principal. Whisper solo actúa como validador y ambas salidas se guardan junto con el estado CC/CD/DC/DD.

## Preparación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Primera ejecución: faster-whisper descarga el modelo configurado en `WHISPER_MODEL`. Requiere conexión a Internet solo para descargar ese modelo; después puede ejecutarse localmente. El modelo propio comienza sin entrenar y por eso mostrará confianza cero hasta ejecutar el entrenamiento con correcciones humanas.

Si el micrófono no es el predeterminado, asigne su índice de `sounddevice.query_devices()` a `INPUT_DEVICE` dentro de `.env`.

Use **Guardar corrección** después de seleccionar/corregir una frase reciente. Luego ejecute `oido-learning/learning_worker.py`; este registra la comparación, la recompensa y los pesos de la política.
