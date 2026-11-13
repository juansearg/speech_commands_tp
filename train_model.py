"""Entrenamiento de un modelo individual.

Este script se ejecuta como subproceso desde el notebook. Al terminar,
el proceso muere y libera la memoria de GPU.

Uso:
    python train_model.py --modelo n4_crnn --batch 128 --lr 3e-4 \
        --run-id prueba_crnn --epochs 60

Para evaluar un modelo ya entrenado:
    python train_model.py --modelo n4_crnn --run-id prueba_crnn \
        --evaluate-only --test
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Debe configurarse antes de importar TensorFlow.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = Path(os.environ.get("TP_BASE_DIR", Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE_DIR))

from datos import crear_generadores
from metricas import calcular_metricas, guardar_metricas
from modelos import MODELOS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", required=True, choices=sorted(MODELOS))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--run-id", required=True)
    p.add_argument("--test", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def predecir(modelo, generador):
    """Devuelve y_true e y_pred para un generador completo."""
    y_true, y_pred = [], []
    for i in range(len(generador)):
        xb, yb = generador[i]
        y_pred.append(modelo.predict_on_batch(xb).argmax(axis=1))
        y_true.append(yb)
    return np.concatenate(y_true), np.concatenate(y_pred)


def guardar_reporte(y_true, y_pred, clases, labels_palabras, out, prefijo="dev"):
    target_names = [clases[i] for i in labels_palabras]
    reporte = classification_report(
        y_true,
        y_pred,
        labels=labels_palabras,
        target_names=target_names,
        zero_division=0,
    )
    (out / f"classification_report_{prefijo}.txt").write_text(
        reporte, encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_palabras)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
    ax.set_title(f"Matriz de confusión ({prefijo})")
    plt.tight_layout()
    fig.savefig(out / f"confusion_matrix_{prefijo}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    base_dir = BASE_DIR
    resultados = base_dir / "resultados"
    out = resultados / args.run_id
    out.mkdir(parents=True, exist_ok=True)

    train_gen, dev_gen, test_gen, norm = crear_generadores(
        base_dir=base_dir,
        batch_size=args.batch,
    )

    clases = list(norm["classes"])
    num_classes = len(clases)
    labels_palabras = [i for i, c in enumerate(clases) if c != "_silence_"]

    modelo = MODELOS[args.modelo](tuple(norm["input_shape"]))

    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        modelo.summary(print_fn=lambda x: f.write(x + "\n"))

    hist = None
    seg = 0.0
    if not args.evaluate_only:
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
            callbacks.CSVLogger(str(out / "historia.csv")),
            callbacks.ModelCheckpoint(
                str(out / "mejor.weights.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ]
        t0 = time.time()
        hist = modelo.fit(
            train_gen,
            validation_data=dev_gen,
            epochs=args.epochs,
            callbacks=callbacks_list,
            verbose=2,
        )
        seg = time.time() - t0
    else:
        pesos = out / "mejor.weights.h5"
        if not pesos.exists():
            raise FileNotFoundError(f"No se encontró {pesos}")
        modelo.load_weights(str(pesos))

    # Evaluación en Dev
    y_true_dev, y_pred_dev = predecir(modelo, dev_gen)
    metricas_dev = calcular_metricas(y_true_dev, y_pred_dev, labels_palabras)
    guardar_reporte(y_true_dev, y_pred_dev, clases, labels_palabras, out, "dev")

    if hist is not None:
        pd.DataFrame(hist.history).to_csv(out / "historia.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist.history["loss"], label="train")
        axes[0].plot(hist.history["val_loss"], label="validación")
        axes[0].set_title("Pérdida")
        axes[0].set_xlabel("época")
        axes[0].legend()
        axes[1].plot(hist.history["accuracy"], label="train")
        axes[1].plot(hist.history["val_accuracy"], label="validación")
        axes[1].set_title("Exactitud")
        axes[1].set_xlabel("época")
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(out / "curvas.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Test: solo cuando se pide explícitamente.
    metricas_test = None
    if args.test:
        y_true_test, y_pred_test = predecir(modelo, test_gen)
        metricas_test = calcular_metricas(y_true_test, y_pred_test, labels_palabras)
        guardar_reporte(y_true_test, y_pred_test, clases, labels_palabras, out, "test")

    nombre = "metricas_test.json" if args.evaluate_only else "metricas.json"
    datos = guardar_metricas(
        out / nombre,
        run_id=args.run_id,
        modelo=args.modelo,
        params=modelo.count_params(),
        seg=seg,
        dev=metricas_dev,
        test=metricas_test,
        extra={
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
            "epocas_entrenadas": int(len(hist.history["loss"])) if hist else None,
        },
    )

    print(json.dumps(datos, indent=2, ensure_ascii=False))
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
