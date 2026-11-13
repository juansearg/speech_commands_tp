# TP IA 2026 — Reconocimiento de comandos de voz

Trabajo práctico de Inteligencia Artificial sobre el conjunto de datos **Speech Commands v0.02**.

El notebook principal es:

```text
TP_IA2026_CAPUTO_CARRIZO_CARRO_GRUPO6.ipynb
```

## Conjunto de datos

Se usa Speech Commands v0.02 de Google:

- Página: https://arxiv.org/abs/1804.03209
- Descarga: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- Prueba: http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz

La estructura esperada es:

```text
data/
  train/
    one/
    two/
    ...
  test/
    ...
```

Cada archivo `.wav` dura aproximadamente 1 segundo y la etiqueta es el nombre de la carpeta.

## Entorno

Se recomienda Python 3.11 y una GPU con al menos 4 GB de memoria.

Instalación local:

```bash
pip install -r requirements.txt
```

En Google Colab, TensorFlow y PyTorch ya vienen instalados. Solo hacen falta las demás librerías:

```python
!pip install -q librosa h5py
```

## Cómo ejecutar el notebook

1. Abrir `TP_IA2026_CAPUTO_CARRIZO_CARRO_GRUPO6.ipynb`.
2. Ejecutar la celda de **entorno y rutas**. En local usa la carpeta del proyecto; en Colab monta Google Drive y usa `MyDrive/TP_IA2026`.
3. Ejecutar las celdas de preparación de datos. Si ya existen los archivos procesados, se pueden saltear.
4. Ejecutar la celda de **preprocesados**. Genera:
   - `preprocesados/split.npz`
   - `preprocesados/norm.json`
5. Ejecutar las celdas `%%writefile`:
   - `%%writefile metricas.py`
   - `%%writefile datos.py`
   - `%%writefile modelos.py`
   - `%%writefile train_model.py`
6. Para modificar una red, editar la celda de `modelos.py` y volver a ejecutarla.
7. Ejecutar las celdas de entrenamiento. Cada modelo se entrena en un subproceso separado.
8. Al final, ejecutar la celda de **tabla comparativa** y las celdas de evaluación.

Cada entrenamiento guarda sus resultados en:

```text
resultados/<run_id>/
  metricas.json
  historia.csv
  mejor.weights.h5
  curvas.png
  confusion_matrix_dev.png
  classification_report_dev.txt
  summary.txt
```

## Flujo de trabajo

Las celdas de entrenamiento usan `entrenar_y_evaluar(nombre, clave_modelo)`. Esa función:

- lanza `train_model.py` en un subproceso,
- libera la memoria de la GPU al terminar el proceso,
- carga el historial y las métricas en el notebook.

El baseline y las redes guardan sus métricas con el mismo formato en `resultados/<run_id>/metricas.json`. Los dos baselines quedan como `logreg_basico` y `logreg_extendido`.

Las redes están en la celda `%%writefile modelos.py`. Para cambiar una red, se edita esa celda y se vuelve a ejecutar. No hay automatizaciones ocultas ni generación automática de código.

## Entrenamiento por subproceso

Cada modelo se entrena con un proceso independiente:

```bash
python train_model.py --modelo n4_crnn --batch 128 --lr 3e-4 --run-id n4_crnn_b128_lr3e4
```

Al terminar el proceso, la memoria de la GPU se libera. Esto permite entrenar varios modelos seguidos sin reiniciar el kernel.

Para evaluar un modelo ya entrenado en el conjunto de prueba:

```bash
python train_model.py --modelo n4_crnn --run-id n4_crnn_b128_lr3e4 --evaluate-only --test
```

## Modelos

| Nombre | Descripción |
|---|---|
| `n0_mlp` | MLP de tres capas ocultas sobre el espectrograma aplanado |
| `n1_cnn_min` | CNN mínima, con un bloque convolucional |
| `n2_cnn` | CNN con tres capas convolucionales y cabeza densa |
| `n3_cnn` | CNN regularizada con tres bloques, BatchNorm y Dropout |
| `n4_crnn` | CNN + GRU (modelo final) |
| `logreg_basico` | Regresión logística con features básicas |
| `logreg_extendido` | Regresión logística con features extendidas |

## Métrica

La métrica principal es **F1 macro sobre las 35 palabras**, excluyendo `_silence_`.

Motivo: el conjunto de prueba está desbalanceado y `_silence_` no aparece en entrenamiento. La exactitud y el F1 ponderado se reportan como métricas secundarias.

El conjunto de **desarrollo** se usa para elegir el modelo. El conjunto de **prueba** se evalúa una sola vez, al final.

## División por hablante

Pendiente de implementación.

## Reproducibilidad

- Las particiones usan `random_state=42`.
- El script de entrenamiento fija la semilla con `--seed` (por defecto 42).
- Las versiones de las librerías están en `requirements.txt`.
- Los archivos `preprocesados/`, `modelos.py`, `datos.py` y `train_model.py` se generan desde el notebook.

## Estructura del proyecto

```text
.
├── TP_IA2026_CAPUTO_CARRIZO_CARRO_GRUPO6.ipynb
├── train_model.py
├── modelos.py
├── datos.py
├── metricas.py
├── requirements.txt
├── README.md
├── data/
├── preprocesados/
└── resultados/
```
