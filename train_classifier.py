import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

CSV_DEFAULT = "aquaculture_stream.csv"
MODEL_PATH = Path("aq_dnn.keras")
ANALYSIS_FIG = Path("analysis_report.png")
TRAINING_CURVE_FIG = Path("training_curves.png")

# Thresholds tuned for tilapia as defaults
PH_LOW, PH_HIGH = 7.1, 8.5
DO_LOW = 7.64
TEMP_OPT_HIGH = 26.3

LABELS_ZH = {
    "pH_low": "pH 太低",
    "pH_high": "pH 太高",
    "DO_low": "溶氧不足",
    "Temp_high": "水溫過高",
    "normal": "狀態正常",
}

# Target epoch ranges for each fold to create a continuous timeline
EPOCH_RANGES = {
    1: (1, 15),
    2: (20, 30),
    3: (35, 45),
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train K-fold classifier for aquaculture telemetry")
    parser.add_argument("--csv", default=CSV_DEFAULT, help="Path to the aggregated telemetry CSV")
    parser.add_argument("--splits", type=int, default=3, help="Number of K-Fold splits")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--ph-low", type=float, default=PH_LOW, help="Lower pH threshold")
    parser.add_argument("--ph-high", type=float, default=PH_HIGH, help="Upper pH threshold")
    parser.add_argument("--do-low", type=float, default=DO_LOW, help="Dissolved oxygen low threshold")
    parser.add_argument("--temp-high", type=float, default=TEMP_OPT_HIGH, help="Temperature high threshold")
    parser.add_argument("--no-figure", action="store_true", help="Skip generating analysis figure")
    return parser.parse_args()


def auto_label(row, ph_low, ph_high, do_low, temp_high):
    ph, dissolved, temp = row["ph"], row["o2"], row["temp"]
    if ph < ph_low:
        return "pH_low"
    if ph > ph_high:
        return "pH_high"
    if dissolved < do_low:
        return "DO_low"
    if temp > temp_high:
        return "Temp_high"
    return "normal"


def build_model(num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def resolve_fonts() -> tuple[list[str], bool]:
    candidates = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "PMingLiU",
        "SimHei",
        "SimSun",
        "DFKai-SB",
        "Noto Sans CJK TC",
        "Arial Unicode MS",
    ]
    available = []
    for family in candidates:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            available.append(family)
        except (ValueError, RuntimeError):
            continue
    if available:
        return available + ["Segoe UI", "sans-serif"], True
    return ["DejaVu Sans", "Arial", "sans-serif"], False


def plot_analysis(classes, class_counts, confusion_matrix_raw, confusion_norm) -> None:
    font_families, support_chinese = resolve_fonts()
    plt.rcParams["font.family"] = font_families
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if support_chinese:
        xtick_labels = [f"{cls}\n{LABELS_ZH.get(cls, cls)}" for cls in classes]
        axis_labels = [LABELS_ZH.get(cls, cls) for cls in classes]
    else:
        xtick_labels = classes
        axis_labels = classes

    axes[0].bar(range(len(classes)), class_counts, color="#3f74e3")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(xtick_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Samples")
    axes[0].set_title("Label Distribution")

    im = axes[1].imshow(confusion_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(axis_labels, rotation=20, ha="right")
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_yticklabels(axis_labels)
    axes[1].set_title("Confusion Matrix (Recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1].text(
                j,
                i,
                f"{confusion_matrix_raw[i, j]}\n{confusion_norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if confusion_norm[i, j] < 0.6 else "white",
            )

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Recall")

    fig.suptitle("Aquaculture Telemetry Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ANALYSIS_FIG, dpi=150)
    plt.close(fig)


def linspace_epochs(fold: int, length: int) -> np.ndarray:
    start, end = EPOCH_RANGES.get(fold, (1, length))
    if length == 1:
        return np.array([start])
    return np.linspace(start, end, num=length)


def plot_training_curves(fold_histories: list[dict], production_metrics: dict[str, float]) -> None:
    if not fold_histories:
        return

    font_families, _ = resolve_fonts()
    plt.rcParams["font.family"] = font_families
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    max_epoch = max(end for end in (rng[1] for rng in EPOCH_RANGES.values()))

    for axis in axes:
        axis.axvspan(*EPOCH_RANGES[1], color="#cbe8ff", alpha=0.3)
        axis.axvspan(*EPOCH_RANGES[2], color="#cbe8ff", alpha=0.2)
        axis.axvspan(*EPOCH_RANGES[3], color="#cbe8ff", alpha=0.3)

    axes[0].text(8, axes[0].get_ylim()[1] * 0.95, "1st train split", ha="center", va="top", fontsize=11)
    axes[0].text(25, axes[0].get_ylim()[1] * 0.95, "2nd train split", ha="center", va="top", fontsize=11)
    axes[0].text(40, axes[0].get_ylim()[1] * 0.95, "3rd train split", ha="center", va="top", fontsize=11)

    for fold_info in fold_histories:
        fold = fold_info["fold"]
        history = fold_info["history"]
        epochs = linspace_epochs(fold, len(history["accuracy"]))

        axes[0].plot(epochs, history["accuracy"], color="#3f74e3", label=f"Fold {fold} Train" if fold == 1 else None)
        if "val_accuracy" in history:
            axes[0].plot(epochs, history["val_accuracy"], color="#f05d5e", linestyle="--", label=f"Fold {fold} Val" if fold == 1 else None)

        axes[1].plot(epochs, history["loss"], color="#3f74e3", label=f"Fold {fold} Train" if fold == 1 else None)
        if "val_loss" in history:
            axes[1].plot(epochs, history["val_loss"], color="#f05d5e", linestyle="--", label=f"Fold {fold} Val" if fold == 1 else None)

    test_epochs = [max_epoch + 1.5]
    axes[0].bar(test_epochs, [production_metrics.get("accuracy")], width=1.2, color="#16c79a", label="Full-set accuracy")
    axes[1].bar(test_epochs, [production_metrics.get("loss")], width=1.2, color="#16c79a", label="Full-set loss")

    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train vs Validation Accuracy (per fold)")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].set_xlim(1, max_epoch + 5)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Train vs Validation Loss (per fold)")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].set_xlim(1, max_epoch + 5)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(TRAINING_CURVE_FIG, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path).dropna(subset=["ph", "o2", "temp"])
    if df.empty:
        raise ValueError("Dataset is empty after dropping NaNs")

    df["label"] = df.apply(
        auto_label,
        axis=1,
        args=(args.ph_low, args.ph_high, args.do_low, args.temp_high),
    )

    features = df[["ph", "o2", "temp"]].astype("float32").values
    labels = df["label"].values

    classes = sorted(np.unique(labels))
    if len(classes) < 2:
        raise ValueError(
            "Only one label present after auto labelling. Adjust thresholds or provide more diverse data."
        )
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    y = np.array([class_to_id[label] for label in labels], dtype="int32")

    kf = KFold(n_splits=args.splits, shuffle=True, random_state=42)
    reports = []
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    fold_histories: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features), start=1):
        scaler = StandardScaler().fit(features[train_idx])
        X_train, X_val = scaler.transform(features[train_idx]), scaler.transform(features[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(len(classes))
        history = model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            validation_data=(X_val, y_val),
        )

        fold_histories.append({"fold": fold, "history": history.history})

        predictions = model.predict(X_val, verbose=0).argmax(axis=1)
        confusion += confusion_matrix(y_val, predictions, labels=range(len(classes)))
        reports.append(
            classification_report(
                y_val,
                predictions,
                labels=range(len(classes)),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            )
        )

        print(f"Fold {fold} accuracy: {reports[-1]['accuracy']:.4f}")

    avg_accuracy = float(np.mean([report["accuracy"] for report in reports]))
    macro_f1 = float(np.mean([report["macro avg"]["f1-score"] for report in reports]))

    print("Classes:", classes)
    print(f"Average accuracy: {avg_accuracy:.4f}")
    print(f"Macro F1 score: {macro_f1:.4f}")
    print("Confusion matrix:\n", confusion)

    full_scaler = StandardScaler().fit(features)
    X_full = full_scaler.transform(features)
    production_model = build_model(len(classes))
    prod_history = production_model.fit(
        X_full,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    prod_loss, prod_acc = production_model.evaluate(X_full, y, verbose=0)

    production_model.save(MODEL_PATH, include_optimizer=False)
    joblib.dump({"scaler": full_scaler, "classes": classes, "class_to_id": class_to_id}, "aq_meta.pkl")
    print(f"Saved model to {MODEL_PATH} and metadata to aq_meta.pkl")

    if not args.no_figure:
        class_counts = df["label"].value_counts().reindex(classes, fill_value=0).values
        row_sums = confusion.sum(axis=1, keepdims=True).clip(min=1)
        confusion_norm = confusion / row_sums
        plot_analysis(classes, class_counts, confusion, confusion_norm)
        print(f"Saved analysis figure to {ANALYSIS_FIG}")

        plot_training_curves(
            fold_histories,
            {"accuracy": prod_acc, "loss": prod_loss},
        )
        print(f"Saved training curves to {TRAINING_CURVE_FIG}")


if __name__ == "__main__":
    main()
