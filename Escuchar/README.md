# Escuchar: flujo por momentos

Cada aplicación se usa cuando usted decide; ninguna inicia entrenamiento automáticamente.

1. `01-capturador-oaf`: capture una palabra: `python capture.py --label hola`.
2. Apruebe o rechace sus muestras en PostgreSQL cuando las revise.
3. `02-entrenador-oaf`: ejecute `python train.py` cuando quiera crear un modelo nuevo.
4. `03-ajustador-pesos`: ejecute `python weights.py --threshold 0.70` para crear otra configuración sin borrar la anterior.
5. `04-pantalla-escucha`: ejecute `python screen.py`; solo usa el modelo activo y escribe el resultado en pantalla.

Las muestras pendientes nunca entran al entrenamiento. El modelo en vivo no captura patrones ni ajusta pesos.
