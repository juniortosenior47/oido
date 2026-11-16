# oido-asr-core

Modelo propio de reconocimiento de voz, entrenado desde cero. No usa pesos de Whisper: transforma audio a log-Mel y usa una red Conv + BiGRU con pérdida CTC para producir caracteres españoles.

Al principio no existe `models/latest.pt`, por lo que el modelo responde con confianza cero. Eso es correcto: primero se deben reunir audios y correcciones humanas.

El conjunto de entrenamiento se forma con los WAV de `oido-stt/recordings` y correcciones registradas en PostgreSQL. Whisper solo sirve para validación y etiquetas provisionales.
