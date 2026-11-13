import json
from pathlib import Path
import numpy as np
import h5py
import tensorflow as tf


class GeneradorLotesHDF5(tf.keras.utils.Sequence):
    def __init__(self, h5_path, etiquetas, batch_size, mu, sd, rutas_esperadas=None):
        super().__init__()
        self.h5_path = h5_path
        self.etiquetas = np.asarray(etiquetas)
        self.batch_size = batch_size
        self.mu = mu
        self.sd = sd

        with h5py.File(self.h5_path, "r") as hf:
            self.num_samples = hf["X_spec"].shape[0]
            if rutas_esperadas is not None:
                if "paths" not in hf:
                    raise ValueError(f"{self.h5_path} no tiene el dataset 'paths'.")
                rutas_h5 = [
                    p.decode() if isinstance(p, bytes) else str(p)
                    for p in hf["paths"][:]
                ]
                rutas_esperadas = [str(r) for r in rutas_esperadas]
                if rutas_h5 != rutas_esperadas:
                    raise ValueError(
                        f"{self.h5_path}: las rutas no coinciden con las etiquetas."
                    )

        if len(self.etiquetas) != self.num_samples:
            raise ValueError(
                f"{self.h5_path}: hay {self.num_samples} espectrogramas "
                f"pero {len(self.etiquetas)} etiquetas."
            )

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, i):
        start = i * self.batch_size
        end = min((i + 1) * self.batch_size, self.num_samples)

        with h5py.File(self.h5_path, "r") as hf:
            X_lote = hf["X_spec"][start:end]

        X_lote = (X_lote - self.mu) / self.sd
        return X_lote, self.etiquetas[start:end]


def crear_generadores(base_dir, batch_size=128):
    base_dir = Path(base_dir)
    z = np.load(base_dir / "preprocesados" / "split.npz", allow_pickle=True)
    with open(base_dir / "preprocesados" / "norm.json", encoding="utf-8") as f:
        norm = json.load(f)

    mu, sd = norm["mu"], norm["sd"]
    data = base_dir / "data"

    train_gen = GeneradorLotesHDF5(
        data / "X_train_spec.h5", z["y_tr_num"], batch_size, mu, sd,
        rutas_esperadas=z["train_paths"],
    )
    dev_gen = GeneradorLotesHDF5(
        data / "X_dev_spec.h5", z["y_va_num"], batch_size, mu, sd,
        rutas_esperadas=z["dev_paths"],
    )
    test_gen = GeneradorLotesHDF5(
        data / "X_test_spec.h5", z["y_te_num"], batch_size, mu, sd,
        rutas_esperadas=z["test_paths"],
    )
    return train_gen, dev_gen, test_gen, norm
