#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dog-Vision en Linux (local)
Transfer-learning con EfficientNetV2B0 sobre Stanford Dogs Dataset.
Autor: tú
"""
import os, pathlib, random, datetime, tarfile, urllib.request, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # menos spam de TF

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"))

# --------------------------------------------------
# 1. CONFIGURACIÓN GLOBAL
# --------------------------------------------------
DATA_DIR   = pathlib.Path("data")            # datasets
MODEL_DIR  = pathlib.Path("models")          # .keras y labels
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. DESCARGA / DESCOMPRIME DATASET
# --------------------------------------------------
def download_stanford_dogs():
    url  = "http://vision.stanford.edu/aditya86/ImageNetDogs"
    files = {"images.tar": "757 MB", "annotation.tar": "21 MB", "lists.tar": "0.5 MB"}
    for fname in files:
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"[INFO] {fname} ya existe")
            continue
        print(f"[INFO] descargando {fname} ({files[fname]}) …")
        urllib.request.urlretrieve(f"{url}/{fname}", dest)
    # descomprimir
    for fname in files:
        with tarfile.open(DATA_DIR / fname) as tar:
            tar.extractall(DATA_DIR)
    print("[INFO] dataset listo en", DATA_DIR.resolve())

download_stanford_dogs()

# --------------------------------------------------
# 3. PREPARAR LISTAS TRAIN / TEST
# --------------------------------------------------
import scipy.io
train_list = scipy.io.loadmat(DATA_DIR / "train_list.mat")["file_list"]
test_list  = scipy.io.loadmat(DATA_DIR / "test_list.mat")["file_list"]
train_files = [item[0][0] for item in train_list]
test_files  = [item[0][0] for item in test_list]
print("Imágenes train:", len(train_files), "| test:", len(test_files))

# --------------------------------------------------
# 4. CREAR CARPETAS CLASIFICADORAS
# --------------------------------------------------
def build_folder_structure(split_name, file_list):
    base = DATA_DIR / "ready" / split_name
    for path in file_list:
        breed_folder = path.split("/")[0].split("-", 1)[1].lower().replace("-", "_")
        dest = base / breed_folder / pathlib.Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = DATA_DIR / "Images" / path
        if not dest.exists():
            dest.symlink_to(src.resolve())  # ahorra espacio
    print(f"[INFO] {split_name} enlazado en", base)

build_folder_structure("train", train_files)
build_folder_structure("test",  test_files)

# 10 % rápido
random.shuffle(train_files)
build_folder_structure("train_10pct", train_files[:len(train_files)//10])

# --------------------------------------------------
# 5. PIPELINE tf.data
# --------------------------------------------------
def make_ds(split_dir, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "ready" / split_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEED
    )
    # guardar los nombres antes de perderlos
    global class_names
    if not globals().get("class_names"):   # la primera vez que se llame
        class_names = ds.class_names
    return ds.cache().prefetch(tf.data.AUTOTUNE)

train_ds      = make_ds("train")
train_10pct_ds= make_ds("train_10pct")
test_ds       = make_ds("test", shuffle=False)


print("Clases:", len(class_names), "(primeras 5:", class_names[:5], ")")

# --------------------------------------------------
# 6. CREAR MODELO (transfer learning)
# --------------------------------------------------
def build_model(name="dogvision"):
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        include_preprocessing=True,
        pooling="avg"
    )
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --------------------------------------------------
# 7. ENTRENAR
# --------------------------------------------------
def train_quick():
    model = build_model("quick")
    hist = model.fit(train_10pct_ds, validation_data=test_ds, epochs=5)
    return model, hist

def train_full():
    model = build_model("full")
    hist = model.fit(train_ds, validation_data=test_ds, epochs=5)
    return model, hist

# --------------------------------------------------
# 8. GUARDAR / CARGAR
# --------------------------------------------------
def save_model(model, name):
    path = MODEL_DIR / f"{name}.keras"
    model.save(path)
    print("[INFO] modelo guardado en", path)

def load_model(name):
    path = MODEL_DIR / f"{name}.keras"
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)

# --------------------------------------------------
# 9. PREDECIR FOTO PROPIA
# --------------------------------------------------
def predict_image(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)  # batch
    probs = model.predict(arr)[0]
    idx = int(tf.argmax(probs))
    return class_names[idx], float(tf.reduce_max(probs))

# --------------------------------------------------
# 10. MENÚ SIMPLE
# --------------------------------------------------
def main():
    while True:
        print("\n=== Dog-Vision local ===")
        print("1) Entrenar rápido (10 %)")
        print("2) Entrenar completo (100 %)")
        print("3) Evaluar modelo guardado")
        print("4) Predecir foto propia")
        print("0) Salir")
        opt = input("Elige> ").strip()
        if opt == "0":
            break
        elif opt == "1":
            model, hist = train_quick()
            save_model(model, "quick")
        elif opt == "2":
            model, hist = train_full()
            save_model(model, "full")
        elif opt == "3":
            model = load_model("full") or load_model("quick")
            if model is None:
                print("No hay modelo guardado; entrena primero")
                continue
            loss, acc = model.evaluate(test_ds)
            print(f"Test accuracy: {acc:.3f}")
        elif opt == "4":
            model = load_model("full") or load_model("quick")
            if model is None:
                print("No hay modelo; entrena primero")
                continue
            path = input("Ruta a la foto (jpg/png): ").strip()
            if not pathlib.Path(path).exists():
                print("Archivo no encontrado")
                continue
            pred, prob = predict_image(path, model)
            print(f"Predicción: {pred}  (prob={prob:.3f})")
            img = plt.imread(path)
            plt.imshow(img)
            plt.title(f"{pred}  {prob:.3f}")
            plt.axis("off")
            plt.show()
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()