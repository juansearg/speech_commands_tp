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
8. Al final, ejecutar la celda de **tabla comparativa** y las celdas de evaluación. Para el split por hablante, el orden es: entrenar N6, ejecutar la celda `--evaluate-only --test` y recién después la tabla de hablantes.

Cada entrenamiento guarda sus resultados en:

```text
resultados/<run_id>/
  metricas.json
  metricas_test.json              # solo con --evaluate-only --test
  historia.csv
  mejor.weights.h5
  curvas.png
  confusion_matrix_dev.png
  classification_report_dev.txt
  confusion_matrix_test.png       # solo con --test
  classification_report_test.txt  # solo con --test
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

Ese comando no reentrena: carga `resultados/<run_id>/mejor.weights.h5`, evalúa Dev y Test y escribe `metricas_test.json`. En cambio, un entrenamiento normal con `--test` guarda los campos `test_*` dentro de `metricas.json`.

Para el split por hablante hay que indicar explícitamente `--split hablante`:

```bash
python train_model.py \
    --modelo n6_crnn_conv \
    --run-id n6_crnn_conv_hablante_b128_lr3e4 \
    --split hablante \
    --evaluate-only \
    --test
```

Sin `--split hablante`, el script usa el split aleatorio por defecto y evalúa el modelo con los datos equivocados.

## Modelos

| Nombre | Descripción |
|---|---|
| `logreg_basico` | Regresión logística con features básicas |
| `logreg_extendido` | Regresión logística con features extendidas |
| `n0_mlp` | MLP de tres capas ocultas sobre el espectrograma aplanado |
| `n1_cnn_min` | CNN mínima, con un bloque convolucional |
| `n2_cnn` | CNN con tres capas convolucionales y cabeza densa |
| `n3_cnn` | CNN regularizada con tres bloques, BatchNorm y Dropout |
| `n4_crnn` | CNN + GRU (modelo base final) |
| `n5_crnn_dropout` | N4 con Dropout 0.4 |
| `n6_crnn_conv` | N5 con un tercer bloque convolucional |


## Métrica

La métrica principal es **F1 macro sobre las 35 palabras**, excluyendo `_silence_`.

Motivo: el conjunto de prueba está desbalanceado y `_silence_` no aparece en entrenamiento. La exactitud y el F1 ponderado se reportan como métricas secundarias.

El conjunto de **desarrollo** se usa para elegir el modelo. El Test oficial se evalúa al final con el mejor modelo. En `resultados/` también quedaron evaluaciones de Test de N4 y N6, que se reportan como referencia y no se usan para elegir el modelo. El Test speaker-disjoint es un experimento aparte y no se compara directamente con el Test oficial.

## División por hablante

El notebook incluye una sección de generalización a hablantes no vistos:

- junta train, dev y test en un único `df_all`,
- excluye `_silence_` porque su `person` no es un hablante real,
- muestra la distribución completa de registros por hablante,
- separa 80/15/5 por hablante, sin solapamiento entre conjuntos,
- genera `data/hablante/X_train_spec.h5`, `X_dev_spec.h5` y `X_test_spec.h5`,
- guarda los índices y etiquetas en `preprocesados/hablante/split.npz`,
- guarda además `split_indices.npz` y `distribucion_hablantes.csv`,
- entrena N6 (`n6_crnn_conv`) con el split por hablante y lo evalúa en su propio Test speaker-disjoint.

El Test speaker-disjoint es balanceado y no contiene `_silence_`, mientras que el Test oficial está desbalanceado y sí lo incluye. Por eso los resultados de ambos Tests no se comparan directamente: el experimento por hablante muestra si la red generaliza a voces no vistas, no si supera al split aleatorio.

Para mostrar el Test del N6 hablante en la tabla hay que ejecutar primero el entrenamiento y después la celda `--evaluate-only --test --split hablante`; esa celda genera `metricas_test.json`.

## Reproducibilidad

- La partición aleatoria usa `random_state=42`.
- La partición por hablante usa la primera semilla que contiene las 35 clases en los tres conjuntos; actualmente son 0 y 1, y la semilla usada se imprime en el notebook.
- El script de entrenamiento fija la semilla con `--seed` (por defecto 42).
- Las versiones de las librerías están en `requirements.txt`.
- Los archivos `preprocesados/`, `modelos.py`, `datos.py`, `metricas.py` y `train_model.py` se generan desde el notebook.

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
│   ├── train/
│   ├── test/
│   ├── X_train_spec.h5
│   ├── X_dev_spec.h5
│   ├── X_test_spec.h5
│   └── hablante/
├── preprocesados/
│   ├── split.npz
│   ├── norm.json
│   └── hablante/
└── resultados/
```
