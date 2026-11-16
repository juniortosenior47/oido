# oido-learning

Ejecuta las comparaciones entre la predicción del modelo propio y una corrección humana. Cada corrección se procesa una sola vez, marca la comparación inicial como verificada y genera un `learning_event` más un nuevo `weight_snapshot`.

## Uso

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python learning_worker.py
```

La política `pd-v1` usa la matriz del dilema del prisionero para recompensar o penalizar la calidad de la predicción. Los pesos resultantes son pesos de decisión, no entrenamiento neuronal. Para entrenar/fine-tunear ASR se debe crear un conjunto de audio + texto corregido y evaluar un modelo separado.
