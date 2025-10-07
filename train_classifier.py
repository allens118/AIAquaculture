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

LABELS_EN = {
    "pH_low": "pH Low",
    "pH_high": "pH High",
    "DO_low": "Dissolved Oxygen Low",
    "Temp_high": "Temperature High",
    "normal": "Normal",
}

# Bilingual label presentation
LABELS_DISPLAY = {key: f"{LABELS_EN[key]} ({LABELS_ZH[key]})" for key in LABELS_ZH}
LABELS_MULTILINE = {key: f"{LABELS_EN[key]}\n{LABELS_ZH[key]}" for key in LABELS_ZH}

# Predefined epoch ranges to form a continuous x-axis per fold
EPOCH_RANGES = {
    1: (1, 15),
    2: (20, 30),
    3: (35, 45),
}

SPLIT_LABELS = {
    1: "1st Train Split\n第1折訓練",
    2: "2nd Train Split\n第2折訓練",
    3: "3rd Train Split\n第3折訓練",
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
    available: list[str] = []
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
        xtick_labels = [LABELS_MULTILINE.get(cls, cls) for cls in classes]
        axis_labels = xtick_labels
    else:
        xtick_labels = [LABELS_DISPLAY.get(cls, cls) for cls in classes]
        axis_labels = xtick_labels

    axes[0].bar(range(len(classes)), class_counts, color="#3f74e3")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_xticklabels(xtick_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Samples / 筆數")
    axes[0].set_title("Label Distribution / 標籤分佈")

    im = axes[1].imshow(confusion_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(axis_labels, rotation=20, ha="right")
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_yticklabels(axis_labels)
    axes[1].set_title("Confusion Matrix (Recall) / 混淆矩陣")
    axes[1].set_xlabel("Predicted / 預測")
    axes[1].set_ylabel("Actual / 實際")

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
    cbar.set_label("Recall / 召回率")

    fig.suptitle("Aquaculture Telemetry Analysis / 水產養殖遙測分析", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ANALYSIS_FIG, dpi=150)
    plt.close(fig)


def linspace_epochs(fold: int, length: int) -> np.ndarray:
    start, end = EPOCH_RANGES.get(fold, (1, length))
    if length <= 1:
        return np.array([start], dtype=float)
    return np.linspace(start, end, num=length, endpoint=True)


def plot_training_curves(fold_histories: list[dict], production_metrics: dict[str, float]) -> None:
    if not fold_histories:
        return

    font_families, _ = resolve_fonts()
    plt.rcParams["font.family"] = font_families
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    min_epoch = min(rng[0] for rng in EPOCH_RANGES.values())
    max_epoch = max(rng[1] for rng in EPOCH_RANGES.values())

    shades = [0.22, 0.12, 0.22]
    for idx, (fold, (start, end)) in enumerate(EPOCH_RANGES.items(), start=1):
        color = "#cbe8ff"
        for axis in axes:
            alpha = shades[(idx - 1) % len(shades)]
            axis.axvspan(start, end, color=color, alpha=alpha, linewidth=0)

    train_acc_x: list[float] = []
    train_acc_y: list[float] = []
    val_acc_x: list[float] = []
    val_acc_y: list[float] = []
    train_loss_x: list[float] = []
    train_loss_y: list[float] = []
    val_loss_x: list[float] = []
    val_loss_y: list[float] = []

    for fold_info in fold_histories:
        fold = fold_info["fold"]
        history = fold_info["history"]
        epochs = linspace_epochs(fold, len(history.get("accuracy", [])))

        train_acc_x.extend(epochs)
        train_acc_y.extend(history.get("accuracy", []))
        train_loss_x.extend(epochs)
        train_loss_y.extend(history.get("loss", []))

        if "val_accuracy" in history:
            val_acc_x.extend(epochs)
            val_acc_y.extend(history["val_accuracy"])
        if "val_loss" in history:
            val_loss_x.extend(epochs)
            val_loss_y.extend(history["val_loss"])

    axes[0].plot(train_acc_x, train_acc_y, color="#3f74e3", linewidth=1.6, marker="o", label="Train Accuracy / 訓練")
    if val_acc_x:
        axes[0].plot(val_acc_x, val_acc_y, color="#f05d5e", linewidth=1.6, linestyle="--", marker="s", label="Validation Accuracy / 驗證")

    axes[1].plot(train_loss_x, train_loss_y, color="#3f74e3", linewidth=1.6, marker="o", label="Train Loss / 訓練")
    if val_loss_x:
        axes[1].plot(val_loss_x, val_loss_y, color="#f05d5e", linewidth=1.6, linestyle="--", marker="s", label="Validation Loss / 驗證")

    test_epoch = max_epoch + 2
    axes[0].bar([test_epoch], [production_metrics.get("accuracy", np.nan)], width=1.5, color="#16c79a", label="Full-set Accuracy / 完整資料")
    axes[1].bar([test_epoch], [production_metrics.get("loss", np.nan)], width=1.5, color="#16c79a", label="Full-set Loss / 完整資料")

    axes[0].set_ylabel("Accuracy / 準確率")
    axes[0].set_title("Train vs Validation Accuracy / 訓練與驗證準確率")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].set_xlim(min_epoch, test_epoch + 3)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Epoch / 訓練輪次")
    axes[1].set_ylabel("Loss / 損失")
    axes[1].set_title("Train vs Validation Loss / 訓練與驗證損失")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].set_xlim(min_epoch, test_epoch + 3)
    axes[1].legend(loc="upper right")

    # Add split labels after plotting so we know the axis limits
    acc_ylim = axes[0].get_ylim()
    loss_ylim = axes[1].get_ylim()
    acc_y = acc_ylim[1] - (acc_ylim[1] - acc_ylim[0]) * 0.05
    loss_y = loss_ylim[1] - (loss_ylim[1] - loss_ylim[0]) * 0.05
    for idx, (fold, (start, end)) in enumerate(EPOCH_RANGES.items(), start=1):
        midpoint = (start + end) / 2
        label = SPLIT_LABELS.get(fold, f"Fold {fold}")
        axes[0].text(midpoint, acc_y, label, ha="center", va="top", fontsize=11, color="#34495e")
        axes[1].text(midpoint, loss_y, label, ha="center", va="top", fontsize=11, color="#34495e")

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
    production_model.fit(
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
