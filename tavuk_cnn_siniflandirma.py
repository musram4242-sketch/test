# %% [markdown]
# Tavuk CQT Spektrogramları - 3 Sınıflı CNN
#
# Bu sürümde veri tekrar bölünmez.
# - Eğitim verisi: E:/tavuksesV2_split/train
# - Test verisi  : E:/tavuksesV2_split/test
#
# Çıktılar:
# - Karmaşıklık matrisi
# - Accuracy, Precision, Recall, F1-Score, Log_Loss, ROC_AUC

# %%
import random
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
from tensorflow import keras
from tensorflow.keras import layers

# %%
# ========== AYARLAR ==========
TRAIN_DIR = Path(r"E:/tavuksesV2_split/train")
TEST_DIR = Path(r"E:/tavuksesV2_split/test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42
LEARNING_RATE = 0.0001  # her epoch boyunca sabit kalır

# %%
# Tekrarlanabilirlik
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Eğitim klasörü bulunamadı: {TRAIN_DIR}")
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test klasörü bulunamadı: {TEST_DIR}")

print(f"Train klasörü: {TRAIN_DIR}")
print(f"Test klasörü : {TEST_DIR}")

# %%
# Dataset yükleme
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
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
# Model (BatchNormalization eklendi, parametre sayısı ciddi artırıldı)
model = keras.Sequential(
    [
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(512, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(512, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print(f"Sabit learning rate: {LEARNING_RATE}")

# %%
# Eğitim
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
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
# model.save("tavuk_cnn_model_v2.h5")
