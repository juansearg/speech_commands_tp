import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def calcular_metricas(y_true, y_pred, labels):
    """Calcula las mismas métricas para el baseline y para las redes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)
    mask = np.isin(y_true, labels)

    return {
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_palabras": (
            float(accuracy_score(y_true[mask], y_pred[mask]))
            if mask.any()
            else float("nan")
        ),
    }


def guardar_metricas(ruta, run_id, modelo, params, seg, dev, test=None, extra=None):
    """Guarda las métricas en el mismo formato para todos los modelos."""
    datos = {
        "run_id": run_id,
        "modelo": modelo,
        "params": int(params),
        "seg": float(seg),
    }
    datos.update({f"dev_{k}": v for k, v in dev.items()})
    if test is not None:
        datos.update({f"test_{k}": v for k, v in test.items()})
    if extra:
        datos.update(extra)

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    return datos
