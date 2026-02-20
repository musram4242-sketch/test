# %% [markdown]
# Tavuk CQT Spektrogramları - 3 Sınıflı CNN
#
# Bu script/notebook-hücresi yapısı şunları yapar:
# 1) E:/tavuksesV1 içindeki sınıfları okur (hasta, none, saglikli)
# 2) Veriyi stratified %80 train / %20 test böler
# 3) Split'i E:/tavuksesV1_split altına kopyalar
# 4) CNN modelini eğitir
# 5) Confusion matrix + Accuracy/Precision/Recall/F1/Log_Loss/ROC_AUC üretir

# %%
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# %%
# ========== AYARLAR ==========
SOURCE_DIR = Path(r"E:/tavuksesV1")
SPLIT_DIR = SOURCE_DIR.parent / "tavuksesV1_split"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
TEST_RATIO = 0.20
SEED = 42

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# %%
# Tekrarlanabilirlik
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def collect_image_paths(source_dir: Path, allowed_ext: set[str]) -> pd.DataFrame:
    if not source_dir.exists():
        raise FileNotFoundError(f"Kaynak klasör bulunamadı: {source_dir}")

    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError("Kaynak klasörde sınıf alt klasörü yok.")

    rows: list[dict] = []
    for cdir in class_dirs:
        for f in sorted(cdir.iterdir()):
            if f.is_file() and f.suffix.lower() in allowed_ext:
                rows.append({"path": f, "label": cdir.name})

    if not rows:
        raise ValueError("Uygun uzantıda görsel bulunamadı.")

    df = pd.DataFrame(rows)

    # Çok küçük sınıflar stratify sırasında hata verir.
    counts = df["label"].value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Bazı sınıflarda 2'den az görsel var, stratified split yapılamaz: "
            + ", ".join([f"{k}={v}" for k, v in too_small.items()])
        )

    return df


def split_and_copy(df: pd.DataFrame, split_dir: Path, test_ratio: float, seed: int):
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=df["label"],
    )

    if split_dir.exists():
        shutil.rmtree(split_dir)

    train_root = split_dir / "train"
    test_root = split_dir / "test"

    for _, row in train_df.iterrows():
        dst = train_root / row["label"] / Path(row["path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], dst)

    for _, row in test_df.iterrows():
        dst = test_root / row["label"] / Path(row["path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], dst)

    return train_df, test_df, train_root, test_root


full_df = collect_image_paths(SOURCE_DIR, ALLOWED_EXT)
train_df, test_df, train_dir, test_dir = split_and_copy(full_df, SPLIT_DIR, TEST_RATIO, SEED)

print(f"Toplam görsel: {len(full_df)}")
print(full_df["label"].value_counts())
print(f"Eğitim: {len(train_df)} | Test: {len(test_df)}")
print(f"Train klasörü: {train_dir}")
print(f"Test klasörü : {test_dir}")

# %%
# Dataset yükleme
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Sınıflar:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# %%
# Model
model = keras.Sequential(
    [
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# %%
# Eğitim
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# %%
# Değerlendirme
y_true_onehot = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_true = np.argmax(y_true_onehot, axis=1)

y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Karmaşıklık Matrisi (Confusion Matrix)")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()

# %%
# Metrikler
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
logloss = log_loss(y_true, y_prob, labels=list(range(num_classes)))

try:
    roc_auc = roc_auc_score(y_true_onehot, y_prob, multi_class="ovr", average="macro")
except ValueError:
    roc_auc = np.nan
    print("Uyarı: ROC_AUC hesaplanamadı (test setinde bir sınıf eksik olabilir).")

metrics_df = pd.DataFrame(
    [
        {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Log_Loss": logloss,
            "ROC_AUC": roc_auc,
        }
    ]
)

print("Değerlendirme Metrikleri")
print(metrics_df.round(4))

# %%
# İsterseniz modeli kaydedin:
# model.save("tavuk_cnn_model.h5")
