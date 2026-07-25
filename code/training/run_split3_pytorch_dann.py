from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"

TASKS = [
    ("three_class_with_air", ["air", "alcohol", "acetone"]),
    ("binary_without_air", ["alcohol", "acetone"]),
]

VARIANTS = [
    {"name": "PyTorch MLP source only", "use_target_labels": False, "use_domain_loss": False, "lambdas": [0.0]},
    {"name": "PyTorch MLP source + Day1", "use_target_labels": True, "use_domain_loss": False, "lambdas": [0.0]},
    {"name": "PyTorch DANN-UDA", "use_target_labels": False, "use_domain_loss": True, "lambdas": [0.05, 0.2, 0.5]},
    {"name": "PyTorch DANN-semi", "use_target_labels": True, "use_domain_loss": True, "lambdas": [0.05, 0.2, 0.5]},
]

SEEDS = [7, 11, 23]
EPOCHS = 500
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


class DANN(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(16, n_classes)
        self.domain = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward_class(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature(x))

    def forward_domain(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        h = self.feature(x)
        return self.domain(grad_reverse(h, lambd))

    def domain_no_grl(self, x: torch.Tensor) -> torch.Tensor:
        return self.domain(self.feature(x))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[:2] in {f"s{i}" for i in range(1, 7)} and c[2:3] == "_"]


def encode(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(classes)}
    return np.array([mapping[str(v)] for v in labels], dtype=np.int64)


def split_data(df: pd.DataFrame, classes: list[str]) -> dict[str, pd.DataFrame]:
    task = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    return {
        "source": task[task["period"].eq("Period_I")].copy(),
        "target_cal": task[task["period"].eq("Period_II") & task["day_num"].eq(1)].copy(),
        "val": task[task["period"].eq("Period_II") & task["day_num"].between(2, 3)].copy(),
        "test": task[
            (task["period"].eq("Period_II") & task["day_num"].between(4, 17))
            | (task["period"].eq("Period_III") & task["day_num"].between(18, 19))
        ].copy(),
    }


def standardize(x_ref: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.nanmean(x_ref, axis=0)
    std = np.nanstd(x_ref, axis=0)
    std = np.nan_to_num(std, nan=1.0)
    std[std < 1e-9] = 1.0
    return tuple(np.nan_to_num((arr - mean) / std).astype(np.float32) for arr in arrays)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def to_long(y: np.ndarray) -> torch.Tensor:
    return torch.tensor(y, dtype=torch.long)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = len(y) / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    recalls, f1s = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        precision = tp / max(1, cm[:, c].sum())
        recall = tp / max(1, cm[c, :].sum())
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(1e-12, precision + recall))
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def predict(model: DANN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.forward_class(x).argmax(dim=1).cpu().numpy()


def predict_domain(model: DANN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.domain_no_grl(x).argmax(dim=1).cpu().numpy()


def train_model(
    x_cls: np.ndarray,
    y_cls: np.ndarray,
    x_source: np.ndarray,
    x_target: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    variant: dict,
    lambda_value: float,
    seed: int,
) -> tuple[DANN, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DANN(x_cls.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    x_cls_t = to_tensor(x_cls)
    y_cls_t = to_long(y_cls)
    x_source_t = to_tensor(x_source)
    x_target_t = to_tensor(x_target)
    x_dom_t = torch.cat([x_source_t, x_target_t], dim=0)
    y_dom_t = torch.cat(
        [torch.zeros(len(x_source), dtype=torch.long), torch.ones(len(x_target), dtype=torch.long)],
        dim=0,
    )
    x_val_t = to_tensor(x_val)
    y_weight = class_weights(y_cls, n_classes)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0)
    best_epoch = 0
    last_class_loss = 0.0
    last_domain_loss = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        class_logits = model.forward_class(x_cls_t)
        class_loss = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)
        loss = class_loss
        domain_loss = torch.tensor(0.0)
        if variant["use_domain_loss"] and len(x_target) > 0:
            domain_logits = model.forward_domain(x_dom_t, lambda_value)
            domain_loss = F.cross_entropy(domain_logits, y_dom_t)
            loss = loss + domain_loss
        loss.backward()
        optimizer.step()
        last_class_loss = float(class_loss.detach().cpu())
        last_domain_loss = float(domain_loss.detach().cpu())

        if epoch % 10 == 0 or epoch == EPOCHS:
            val_pred = predict(model, x_val_t)
            val_metrics = metric_dict(y_val, val_pred, n_classes)
            score = (val_metrics["macro_f1"], val_metrics["accuracy"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "class_loss_last": last_class_loss,
        "domain_loss_last": last_domain_loss,
    }


def evaluate_variant(data: dict[str, pd.DataFrame], cols: list[str], classes: list[str], variant: dict) -> list[dict]:
    x_source_raw = data["source"][cols].to_numpy(float)
    x_target_raw = data["target_cal"][cols].to_numpy(float)
    x_val_raw = data["val"][cols].to_numpy(float)
    x_test_raw = data["test"][cols].to_numpy(float)
    x_ref = np.vstack([x_source_raw, x_target_raw])
    x_source, x_target, x_val, x_test = standardize(x_ref, x_source_raw, x_target_raw, x_val_raw, x_test_raw)

    y_source = encode(data["source"]["label"].to_numpy(str), classes)
    y_target = encode(data["target_cal"]["label"].to_numpy(str), classes)
    y_val = encode(data["val"]["label"].to_numpy(str), classes)
    y_test = encode(data["test"]["label"].to_numpy(str), classes)

    if variant["use_target_labels"]:
        x_cls = np.vstack([x_source, x_target])
        y_cls = np.concatenate([y_source, y_target])
        label_usage = "source labels + Day1 target labels"
    else:
        x_cls = x_source
        y_cls = y_source
        label_usage = "source labels only"

    rows = []
    for lambda_value in variant["lambdas"]:
        for seed in SEEDS:
            model, info = train_model(
                x_cls, y_cls, x_source, x_target, x_val, y_val, len(classes), variant, lambda_value, seed
            )
            train_pred = predict(model, to_tensor(x_cls))
            val_pred = predict(model, to_tensor(x_val))
            test_pred = predict(model, to_tensor(x_test))
            train_m = metric_dict(y_cls, train_pred, len(classes))
            val_m = metric_dict(y_val, val_pred, len(classes))
            test_m = metric_dict(y_test, test_pred, len(classes))
            x_dom = np.vstack([x_source, x_target])
            y_dom = np.concatenate([np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)])
            domain_pred = predict_domain(model, to_tensor(x_dom))
            domain_acc = float((domain_pred == y_dom).mean())
            rows.append(
                {
                    "variant": variant["name"],
                    "lambda": lambda_value,
                    "seed": seed,
                    "label_usage": label_usage,
                    "best_epoch": int(info["best_epoch"]),
                    "class_loss_last": info["class_loss_last"],
                    "domain_loss_last": info["domain_loss_last"],
                    "train_n": len(y_cls),
                    "source_n": len(y_source),
                    "target_calibration_n": len(y_target),
                    "validation_n": len(y_val),
                    "test_n": len(y_test),
                    "train_accuracy": train_m["accuracy"],
                    "validation_accuracy": val_m["accuracy"],
                    "test_accuracy": test_m["accuracy"],
                    "train_balanced_accuracy": train_m["balanced_accuracy"],
                    "validation_balanced_accuracy": val_m["balanced_accuracy"],
                    "test_balanced_accuracy": test_m["balanced_accuracy"],
                    "train_macro_f1": train_m["macro_f1"],
                    "validation_macro_f1": val_m["macro_f1"],
                    "test_macro_f1": test_m["macro_f1"],
                    "train_test_accuracy_gap": train_m["accuracy"] - test_m["accuracy"],
                    "domain_discriminator_accuracy": domain_acc,
                    "domain_invariance_note": "closer_to_0.5_is_more_invariant",
                }
            )
    return rows


def choose_best(results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (_task, _variant), sub in results.groupby(["task", "variant"], sort=False):
        ordered = sub.sort_values(
            ["validation_macro_f1", "validation_accuracy", "test_accuracy"],
            ascending=[False, False, False],
        )
        selected.append(ordered.iloc[0])
    out = pd.DataFrame(selected).reset_index(drop=True)
    best_by_task = out.groupby("task")["test_accuracy"].transform("max")
    out["judgment"] = np.where(np.isclose(out["test_accuracy"], best_by_task), "Best test accuracy", "Candidate")
    out.loc[out["train_test_accuracy_gap"] > 0.25, "judgment"] += "; high train-test gap"
    out.loc[out["domain_discriminator_accuracy"].between(0.45, 0.60), "judgment"] += "; domain-invariant features"
    out.loc[out["domain_discriminator_accuracy"] > 0.75, "judgment"] += "; domain still separable"
    return out


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def write_svg(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        ("Task", 170),
        ("Variant", 185),
        ("Lambda", 65),
        ("Seed", 45),
        ("Train Acc", 76),
        ("Val Acc", 68),
        ("Test Acc", 72),
        ("Test BA", 68),
        ("Test F1", 68),
        ("Gap", 60),
        ("Domain Acc", 78),
        ("Judgment", 300),
    ]
    row_h = 34
    header_h = 38
    width = 32 + sum(w for _, w in cols) + 32
    height = 112 + header_h + row_h * len(summary) + 58
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="26" y="34" font-family="Arial" font-size="22" font-weight="700">PyTorch DANN on Split 3</text>',
        '<text x="26" y="58" font-family="Arial" font-size="13" fill="#555">Source = Period I; target calibration = Period II Day1; validation = Day2-Day3; test = Day4-Day19. Domain accuracy near 0.5 means more domain-invariant features.</text>',
    ]
    x0, y = 26, 88
    x = x0
    for label, w in cols:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{header_h}" fill="#f4f7fb" stroke="#d7dde7"/>')
        lines.append(f'<text x="{x+6}" y="{y+24}" font-family="Arial" font-size="11" font-weight="700">{label}</text>')
        x += w
    y += header_h
    for _, row in summary.iterrows():
        values = [
            "three-class" if row["task"] == "three_class_with_air" else "binary",
            row["variant"].replace("PyTorch ", ""),
            fmt(row["lambda"]),
            str(int(row["seed"])),
            fmt(row["train_accuracy"]),
            fmt(row["validation_accuracy"]),
            fmt(row["test_accuracy"]),
            fmt(row["test_balanced_accuracy"]),
            fmt(row["test_macro_f1"]),
            fmt(row["train_test_accuracy_gap"]),
            fmt(row["domain_discriminator_accuracy"]),
            row["judgment"],
        ]
        x = x0
        for (label, w), value in zip(cols, values):
            fill = "#e2f4e7" if label == "Judgment" and "Best" in value else ("#fdebd8" if label == "Judgment" and "high" in value else "white")
            lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="{fill}" stroke="#d7dde7"/>')
            lines.append(f'<text x="{x+6}" y="{y+22}" font-family="Arial" font-size="10">{value}</text>')
            x += w
        y += row_h
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_PATH)
    cols = feature_columns(df)
    all_rows = []
    split_rows = []
    for task_name, classes in TASKS:
        data = split_data(df, classes)
        for name, part in data.items():
            split_rows.append(
                {
                    "task": task_name,
                    "partition": name,
                    "n": len(part),
                    "class_counts": "; ".join(f"{c}:{int(part['label'].eq(c).sum())}" for c in classes),
                }
            )
        for variant in VARIANTS:
            rows = evaluate_variant(data, cols, classes, variant)
            for row in rows:
                row["task"] = task_name
                row["framework"] = "PyTorch"
            all_rows.extend(rows)
    results = pd.DataFrame(all_rows)
    summary = choose_best(results)
    results.to_csv(TABLE_DIR / "split3_pytorch_dann_all_runs.csv", index=False)
    summary.to_csv(TABLE_DIR / "split3_pytorch_dann_selected_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(TABLE_DIR / "split3_pytorch_dann_splits.csv", index=False)
    write_svg(summary, FIG_DIR / "split3_pytorch_dann_summary.svg")
    print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())
    print("\nSplits")
    print(pd.DataFrame(split_rows).to_string(index=False))
    print("\nSelected summary")
    print(
        summary[
            [
                "task",
                "variant",
                "lambda",
                "seed",
                "train_accuracy",
                "validation_accuracy",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "train_test_accuracy_gap",
                "domain_discriminator_accuracy",
                "judgment",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
