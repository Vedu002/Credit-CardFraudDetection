"""
AI-driven credit card fraud detection with a hybrid stacking ensemble.

Base learners : 1D-CNN, LSTM, Transformer encoder   (Keras / TensorFlow)
Meta-learner  : XGBoost trained on the base learners' predicted probabilities

Based on: Ileberi & Sun, "A Hybrid Deep Learning Ensemble Model for Credit Card
Fraud Detection", IEEE Access, vol. 12, 2024.

Dataset: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud) -> creditcard.csv

Usage:
    python fraud_detection.py --data creditcard.csv --epochs 15
    python fraud_detection.py --data creditcard.csv --sample 50000 --epochs 3   # quick smoke test
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models
from xgboost import XGBClassifier


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_and_split(path, seed, sample=None):
    """Load CSV and make a stratified 60/20/20 train/validation/test split.

    Train      -> fits the three base learners
    Validation -> fits the XGBoost meta-learner (on base-learner predictions)
    Test       -> untouched until final evaluation (avoids the leakage that
                  arises if the meta-learner is scored on data it was trained on)
    """
    df = pd.read_csv(path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed)
    y = df["Class"].values.astype("float32")
    X = df.drop(columns=["Class"]).values.astype("float32")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    # Fit the scaler on TRAIN only, then apply to all splits.
    scaler = StandardScaler().fit(X_tr)
    feature_names = df.drop(columns=["Class"]).columns.tolist()
    return (scaler.transform(X_tr).astype("float32"), y_tr,
            scaler.transform(X_val).astype("float32"), y_val,
            scaler.transform(X_te).astype("float32"), y_te,
            scaler, feature_names)


# ----------------------------------------------------------------------------
# Base learners
# ----------------------------------------------------------------------------
def build_cnn(n_features):
    inp = layers.Input(shape=(n_features,))
    x = layers.Reshape((n_features, 1))(inp)
    for f in (32, 64, 128):
        x = layers.Conv1D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return _compile(models.Model(inp, out), lr=1e-3)


def build_lstm(n_features):
    inp = layers.Input(shape=(n_features,))
    x = layers.Reshape((n_features, 1))(inp)  # features treated as a sequence
    x = layers.LSTM(50, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(100)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return _compile(models.Model(inp, out), lr=1e-3)


class PositionEmbedding(layers.Layer):
    """Learned positional embedding added to token embeddings."""

    def __init__(self, length, dim, **kw):
        super().__init__(**kw)
        self.length, self.dim = length, dim
        self.emb = layers.Embedding(length, dim)

    def call(self, x):
        return x + self.emb(tf.range(self.length))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(length=self.length, dim=self.dim)
        return cfg


def build_transformer(n_features, d_model=32, heads=4, n_layers=2, ff_dim=128):
    """Each feature becomes a 'token'; self-attention models feature interactions.
    (Paper uses 8 heads / 6 layers / 2048 FFN; scaled down here for CPU training.)"""
    inp = layers.Input(shape=(n_features,))
    x = layers.Reshape((n_features, 1))(inp)
    x = layers.Dense(d_model)(x)
    x = PositionEmbedding(n_features, d_model)(x)
    for _ in range(n_layers):
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(0.1)(a))
        f = layers.Dense(ff_dim, activation="relu")(x)
        f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(0.1)(f))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return _compile(models.Model(inp, out), lr=1e-4 * 5)


def _compile(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.AUC(name="prauc", curve="PR")])
    return model


def train_keras(model, X_tr, y_tr, X_val, y_val, epochs, batch_size):
    cw = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=y_tr)
    stop = callbacks.EarlyStopping(monitor="val_prauc", mode="max", patience=4,
                                   restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=batch_size, class_weight={0: cw[0], 1: cw[1]},
              callbacks=[stop], verbose=2)
    return model


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"Sensitivity": sens, "Specificity": spec, "Precision": prec,
            "F1": f1, "AUC-ROC": roc_auc_score(y_true, prob),
            "AUC-PR": average_precision_score(y_true, prob),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="creditcard.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=None,
                    help="use a random subset of rows (fast smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    X_tr, y_tr, X_val, y_val, X_te, y_te, scaler, feature_names = load_and_split(
        args.data, args.seed, args.sample)
    n_feat = X_tr.shape[1]
    print(f"train={len(X_tr)} val={len(X_val)} test={len(X_te)} "
          f"| frauds train/val/test = {int(y_tr.sum())}/{int(y_val.sum())}/{int(y_te.sum())}")

    # 1) Base learners -------------------------------------------------------
    builders = {"CNN": build_cnn, "LSTM": build_lstm, "Transformer": build_transformer}
    val_preds, test_preds, trained = {}, {}, {}
    for name, build in builders.items():
        print(f"\n=== Training {name} ===")
        m = train_keras(build(n_feat), X_tr, y_tr, X_val, y_val,
                        args.epochs, args.batch_size)
        val_preds[name] = m.predict(X_val, batch_size=4096, verbose=0).ravel()
        test_preds[name] = m.predict(X_te, batch_size=4096, verbose=0).ravel()
        trained[name] = m

    # 2) XGBoost baseline (raw features) ------------------------------------
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb_kw = dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, gamma=0, eval_metric="aucpr",
                  random_state=args.seed, n_jobs=-1)
    xgb_base = XGBClassifier(scale_pos_weight=spw, **xgb_kw).fit(X_tr, y_tr)
    val_preds["XGBoost"] = xgb_base.predict_proba(X_val)[:, 1]
    test_preds["XGBoost"] = xgb_base.predict_proba(X_te)[:, 1]

# 3) Meta-learner: XGBoost on stacked base-learner probabilities ---------
# Includes XGBoost's own prediction alongside CNN/LSTM/Transformer, so the
# meta-learner can lean on whichever base learner is actually reliable.
    names = list(builders) + ["XGBoost"]
    Z_val = np.column_stack([val_preds[n] for n in names])
    Z_te = np.column_stack([test_preds[n] for n in names])
    spw_meta = (y_val == 0).sum() / max((y_val == 1).sum(), 1)
    meta = XGBClassifier(scale_pos_weight=spw_meta, **xgb_kw).fit(Z_val, y_val)
    test_preds["Stacking Ensemble"] = meta.predict_proba(Z_te)[:, 1]

    # 4) Evaluate on the untouched test set ---------------------------------
    rows = {n: evaluate(y_te, p, args.threshold) for n, p in test_preds.items()}
    res = pd.DataFrame(rows).T
    res[["TP", "FP", "FN", "TN"]] = res[["TP", "FP", "FN", "TN"]].astype(int)
    print("\n=== TEST SET RESULTS ===")
    print(res.round(4).to_string())
    res.round(4).to_csv(os.path.join(args.out, "results.csv"))

    # 5) ROC and Precision-Recall plots -------------------------------------
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n, p in test_preds.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_te, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{n} (AP={average_precision_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curves")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves")
    for a in ax:
        a.legend(loc="lower right" if a is ax[0] else "lower left")
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "curves.png"), dpi=150)

    # 6) Save everything the Streamlit demo needs ----------------------------
    models_dir = os.path.join(args.out, "models")
    os.makedirs(models_dir, exist_ok=True)
    for name, m in trained.items():
        m.save(os.path.join(models_dir, f"{name.lower()}.keras"))
    xgb_base.save_model(os.path.join(models_dir, "xgb_base.json"))
    meta.save_model(os.path.join(models_dir, "meta_xgb.json"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nSaved results, plots and all models (for the Streamlit demo) to '{args.out}/'")


if __name__ == "__main__":
    main()