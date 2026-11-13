# Redes del TP. Editar esta celda y ejecutarla para guardar modelos.py.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 36
num_classes = NUM_CLASSES

def bloque_conv_2d(filtros):
    return [
        layers.Conv2D(filtros, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
    ]

def bloque_conv_bn(filtros):
    return [
        layers.Conv2D(filtros, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
    ]

def nivel0_mlp(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="N0_MLP")

def nivel1_cnn_minima(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ], name="N1_CNN_minima")

def nivel2_cnn(input_shape, cabezal="flatten"):
    capas = [layers.Input(shape=input_shape)]
    for f in (16, 32):
        capas += bloque_conv_2d(f)
    capas += [layers.Conv2D(64, (3, 3), activation="relu", padding="same")]
    if cabezal == "flatten":
        capas += [layers.MaxPooling2D((2, 2)), layers.Flatten()]
    else:
        capas += [layers.GlobalAveragePooling2D()]
    capas += [layers.Dense(64, activation="relu"),
              layers.Dropout(0.3),
              layers.Dense(num_classes, activation="softmax")]
    return models.Sequential(capas, name=f"N2_{cabezal}")

def nivel2_cnn_gap(input_shape):
    return nivel2_cnn(input_shape, "gap")

def nivel3_cnn_regularizada(input_shape):
    capas = [layers.Input(shape=input_shape)]
    for f in (16, 32, 64):
        capas += bloque_conv_bn(f)
    capas += [layers.Flatten(),
              layers.Dense(64, activation="relu"),
              layers.Dropout(0.4),
              layers.Dense(num_classes, activation="softmax")]
    return models.Sequential(capas, name="N3_CNN_regularizada")

def nivel4_crnn(input_shape):
    T = input_shape[0] // 4   # tiempo después de dos pools de 2
    F = input_shape[1] // 4   # frecuencia después de dos pools de 2
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Reshape((T, F * 32)),   # secuencia de T pasos, cada uno con F*32 valores
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="N4_CRNN")

MODELOS = {
    "n0_mlp": nivel0_mlp,
    "n1_cnn_min": nivel1_cnn_minima,
    "n2_cnn": nivel2_cnn,
    "n2_gap": nivel2_cnn_gap,
    "n3_cnn": nivel3_cnn_regularizada,
    "n4_crnn": nivel4_crnn,
}
