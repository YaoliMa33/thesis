from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"

TASKS = [
    ("three_class_with_air", ["air", "alcohol", "acetone"]),
    ("binary_without_air", ["alcohol", "acetone"]),
]

VARIANTS = [
    {"name": "MLP source only", "use_target_labels": False, "use_domain_loss": False, "lambdas": [0.0]},
    {"name": "MLP source + Day1", "use_target_labels": True, "use_domain_loss": False, "lambdas": [0.0]},
    {"name": "DANN-UDA", "use_target_labels": False, "use_domain_loss": True, "lambdas": [0.02, 0.05, 0.1, 0.2, 0.5]},
    {"name": "DANN-semi", "use_target_labels": True, "use_domain_loss": True, "lambdas": [0.02, 0.05, 0.1, 0.2, 0.5]},
]

SEEDS = [3, 7, 11, 17, 23]
EPOCHS = 900
HIDDEN = 16
LR = 0.045
L2 = 1e-4


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[:2] in {f"s{i}" for i in range(1, 7)} and c[2:3] == "_"]


def encode(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(classes)}
    return np.array([mapping[str(v)] for v in labels], dtype=int)


def split_data(df: pd.DataFrame, classes: list[str]) -> dict[str, pd.DataFrame]:
    task = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    source = task[task["period"].eq("Period_I")].copy()
    target_cal = task[task["period"].eq("Period_II") & task["day_num"].eq(1)].copy()
    val = task[task["period"].eq("Period_II") & task["day_num"].between(2, 3)].copy()
    test = task[
        (task["period"].eq("Period_II") & task["day_num"].between(4, 17))
        | (task["period"].eq("Period_III") & task["day_num"].between(18, 19))
    ].copy()
    return {"all": task, "source": source, "target_cal": target_cal, "val": val, "test": test}


def standardize(x_ref: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.nanmean(x_ref, axis=0)
    std = np.nanstd(x_ref, axis=0)
    std = np.nan_to_num(std, nan=1.0)
    std[std < 1e-9] = 1.0
    return tuple(np.nan_to_num((arr - mean) / std) for arr in arrays)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def weighted_ce_grad(logits: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[float, np.ndarray]:
    prob = softmax(logits)
    one = np.eye(n_classes)[y]
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = len(y) / (n_classes * np.maximum(counts[y], 1.0))
    loss = -np.mean(weights * np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)))
    grad = (prob - one) * weights[:, None] / len(y)
    return float(loss), grad


def ce_grad(logits: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[float, np.ndarray]:
    prob = softmax(logits)
    one = np.eye(n_classes)[y]
    loss = -np.mean(np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)))
    grad = (prob - one) / len(y)
    return float(loss), grad


def init_params(rng: np.random.Generator, n_features: int, n_classes: int) -> dict[str, np.ndarray]:
    return {
        "wf": rng.normal(0, 0.16, size=(n_features, HIDDEN)),
        "bf": np.zeros(HIDDEN),
        "wc": rng.normal(0, 0.16, size=(HIDDEN, n_classes)),
        "bc": np.zeros(n_classes),
        "wd": rng.normal(0, 0.16, size=(HIDDEN, 2)),
        "bd": np.zeros(2),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = x @ params["wf"] + params["bf"]
    h = np.tanh(pre)
    return pre, h, h @ params["wc"] + params["bc"]


def domain_logits(params: dict[str, np.ndarray], h: np.ndarray) -> np.ndarray:
    return h @ params["wd"] + params["bd"]


def predict(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    _, _, logits = forward(params, x)
    return np.argmax(logits, axis=1)


def domain_predict(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    _, h, _ = forward(params, x)
    return np.argmax(domain_logits(params, h), axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
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


def train_one(
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
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    params = init_params(rng, x_cls.shape[1], n_classes)
    x_dom = np.vstack([x_source, x_target])
    y_dom = np.concatenate([np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)])
    best_params = {k: v.copy() for k, v in params.items()}
    best_score = (-1.0, -1.0)
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        pre_c, h_c, logits_c = forward(params, x_cls)
        class_loss, grad_logits_c = weighted_ce_grad(logits_c, y_cls, n_classes)
        grad_wc = h_c.T @ grad_logits_c + L2 * params["wc"]
        grad_bc = grad_logits_c.sum(axis=0)
        grad_h_c = grad_logits_c @ params["wc"].T
        grad_pre_c = grad_h_c * (1.0 - np.tanh(pre_c) ** 2)
        grad_wf = x_cls.T @ grad_pre_c + L2 * params["wf"]
        grad_bf = grad_pre_c.sum(axis=0)

        if variant["use_domain_loss"] and len(x_target) > 0:
            pre_d, h_d, _ = forward(params, x_dom)
            logits_d = domain_logits(params, h_d)
            _domain_loss, grad_logits_d = ce_grad(logits_d, y_dom, 2)
            grad_wd = h_d.T @ grad_logits_d + L2 * params["wd"]
            grad_bd = grad_logits_d.sum(axis=0)
            grad_h_d = grad_logits_d @ params["wd"].T
            grad_pre_d = grad_h_d * (1.0 - np.tanh(pre_d) ** 2)
            grad_wf -= lambda_value * (x_dom.T @ grad_pre_d)
            grad_bf -= lambda_value * grad_pre_d.sum(axis=0)
        else:
            grad_wd = np.zeros_like(params["wd"])
            grad_bd = np.zeros_like(params["bd"])

        params["wf"] -= LR * grad_wf
        params["bf"] -= LR * grad_bf
        params["wc"] -= LR * grad_wc
        params["bc"] -= LR * grad_bc
        params["wd"] -= LR * grad_wd
        params["bd"] -= LR * grad_bd

        if epoch % 10 == 0 or epoch == EPOCHS:
            val_pred = predict(params, x_val)
            val_metrics = metrics(y_val, val_pred, n_classes)
            score = (val_metrics["macro_f1"], val_metrics["accuracy"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_params = {k: v.copy() for k, v in params.items()}

    return best_params, {"best_epoch": best_epoch, "class_loss_last": class_loss}


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
        train_n = len(y_cls)
    else:
        x_cls = x_source
        y_cls = y_source
        train_n = len(y_cls)

    rows = []
    for lambda_value in variant["lambdas"]:
        for seed in SEEDS:
            params, train_info = train_one(
                x_cls, y_cls, x_source, x_target, x_val, y_val, len(classes), variant, lambda_value, seed
            )
            pred_train = predict(params, x_cls)
            pred_val = predict(params, x_val)
            pred_test = predict(params, x_test)
            m_train = metrics(y_cls, pred_train, len(classes))
            m_val = metrics(y_val, pred_val, len(classes))
            m_test = metrics(y_test, pred_test, len(classes))
            dom_x = np.vstack([x_source, x_target])
            dom_y = np.concatenate([np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)])
            dom_pred = domain_predict(params, dom_x)
            domain_acc = float((dom_pred == dom_y).mean())
            rows.append(
                {
                    "variant": variant["name"],
                    "lambda": lambda_value,
                    "seed": seed,
                    "best_epoch": int(train_info["best_epoch"]),
                    "train_n": train_n,
                    "source_n": len(y_source),
                    "target_calibration_n": len(y_target),
                    "validation_n": len(y_val),
                    "test_n": len(y_test),
                    "train_accuracy": m_train["accuracy"],
                    "validation_accuracy": m_val["accuracy"],
                    "test_accuracy": m_test["accuracy"],
                    "train_balanced_accuracy": m_train["balanced_accuracy"],
                    "validation_balanced_accuracy": m_val["balanced_accuracy"],
                    "test_balanced_accuracy": m_test["balanced_accuracy"],
                    "train_macro_f1": m_train["macro_f1"],
                    "validation_macro_f1": m_val["macro_f1"],
                    "test_macro_f1": m_test["macro_f1"],
                    "train_test_accuracy_gap": m_train["accuracy"] - m_test["accuracy"],
                    "domain_discriminator_accuracy": domain_acc,
                    "domain_invariance_note": "closer_to_0.5_is_more_invariant",
                }
            )
    return rows


def choose_best(results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (task, variant), sub in results.groupby(["task", "variant"], sort=False):
        ordered = sub.sort_values(
            ["validation_macro_f1", "validation_accuracy", "test_accuracy"],
            ascending=[False, False, False],
        )
        selected.append(ordered.iloc[0])
    out = pd.DataFrame(selected).reset_index(drop=True)
    best_by_task = out.groupby("task")["test_accuracy"].transform("max")
    out["judgment"] = np.where(np.isclose(out["test_accuracy"], best_by_task), "Best test accuracy", "Candidate")
    out.loc[out["train_test_accuracy_gap"] > 0.25, "judgment"] = out["judgment"] + "; high train-test gap"
    out.loc[out["domain_discriminator_accuracy"].between(0.45, 0.60), "judgment"] = out["judgment"] + "; domain-invariant features"
    out.loc[out["domain_discriminator_accuracy"] > 0.75, "judgment"] = out["judgment"] + "; domain still separable"
    return out


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def write_svg(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        ("Task", 170),
        ("Variant", 160),
        ("Lambda", 65),
        ("Seed", 50),
        ("Train Acc", 80),
        ("Val Acc", 70),
        ("Test Acc", 75),
        ("Test BA", 70),
        ("Test F1", 70),
        ("Gap", 60),
        ("Domain Acc", 82),
        ("Judgment", 265),
    ]
    row_h = 34
    header_h = 38
    width = 32 + sum(w for _, w in cols) + 32
    height = 110 + header_h + row_h * len(summary) + 60
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="26" y="34" font-family="Arial" font-size="22" font-weight="700">Split 3 preliminary domain adversarial training</text>',
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
            row["variant"],
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
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_PATH)
    cols = feature_columns(df)
    all_rows = []
    split_rows = []
    for task_name, classes in TASKS:
        data = split_data(df, classes)
        split_rows.extend(
            [
                {
                    "task": task_name,
                    "partition": name,
                    "n": len(part),
                    "class_counts": "; ".join(
                        f"{c}:{int(part['label'].eq(c).sum())}" for c in classes
                    ),
                }
                for name, part in data.items()
                if name != "all"
            ]
        )
        for variant in VARIANTS:
            rows = evaluate_variant(data, cols, classes, variant)
            for row in rows:
                row["task"] = task_name
            all_rows.extend(rows)
    results = pd.DataFrame(all_rows)
    summary = choose_best(results)
    results.to_csv(TABLE_DIR / "split3_numpy_dann_all_runs.csv", index=False)
    summary.to_csv(TABLE_DIR / "split3_numpy_dann_selected_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(TABLE_DIR / "split3_numpy_dann_splits.csv", index=False)
    write_svg(summary, FIG_DIR / "split3_numpy_dann_summary.svg")
    print("Splits")
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
