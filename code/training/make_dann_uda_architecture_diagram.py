from __future__ import annotations

import html
from pathlib import Path

import shared_experiment_protocol as protocol


OUT_DIR = Path(r"D:\thesis\figures\dann_learning")


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text(
    x: int,
    y: int,
    value: str,
    size: int = 18,
    weight: int = 400,
    color: str = "#111827",
    anchor: str = "middle",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def multiline(
    x: int,
    y: int,
    lines: list[str],
    size: int = 16,
    color: str = "#111827",
    anchor: str = "middle",
    gap: int = 22,
    weight: int = 400,
) -> str:
    return "\n".join(
        text(x, y + i * gap, line, size=size, color=color, anchor=anchor, weight=weight)
        for i, line in enumerate(lines)
    )


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: int = 2, rx: int = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#374151", sw: int = 2, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
        f'marker-end="url(#arrow)"{dash}/>'
    )


def label_note(parts: list[str], x: int, y: int, lines: list[str], color: str = "#9333ea", w: int = 260, h: int = 86) -> None:
    parts += [
        rect(x, y, w, h, "#faf5ff", color, 2, 8),
        multiline(x + w // 2, y + 26, lines, 13, "#111827", gap=18),
    ]


def settings_note(parts: list[str], x: int, y: int, w: int = 420, include_alpha: bool = False) -> None:
    seed_text = ", ".join(str(seed) for seed in protocol.MODEL_SEEDS)
    last_line = f"seeds: {seed_text}; lambda: 0.05, 0.2, 0.5"
    if include_alpha:
        last_line = f"seeds: {seed_text}; lambda: 0.05, 0.2, 0.5; alpha: 0.05, 0.2, 0.5"
    parts += [
        rect(x, y, w, 112, "#f9fafb", "#6b7280", 1, 8),
        multiline(
            x + w // 2,
            y + 24,
            [
                "Training settings in code",
                "epochs=400, Adam, lr=0.01, weight_decay=1e-4",
                last_line,
                "with validation: select by val macro-F1 / BA / acc",
                "S1 has no validation: fixed defaults; val metrics = NA",
            ],
            12,
            "#374151",
            gap=16,
        ),
    ]


def svg_start(title: str, subtitle: str, w: int = 1700, h: int = 980) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        '<path d="M2,2 L10,6 L2,10 Z" fill="#374151"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(w // 2, 46, title, 28, 700),
        text(w // 2, 76, subtitle, 15, 400, "#4b5563"),
    ]


def write_svg(name: str, parts: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")
    print(path)


def add_common_feature_extractor(parts: list[str], x: int = 520, y: int = 235) -> None:
    parts += [
        rect(x, y, 280, 78, "#eef2ff", "#4f46e5"),
        multiline(x + 140, y + 30, ["Feature extractor G_f", "Linear r -> 32, ReLU", "r = 54 in this report"], 14, gap=18),
        rect(x, y + 135, 280, 78, "#eef2ff", "#4f46e5"),
        multiline(x + 140, y + 165, ["Dropout 0.05", "Linear 32 -> 16, ReLU"], 15),
        rect(x, y + 290, 280, 88, "#ede9fe", "#7c3aed"),
        multiline(x + 140, y + 323, ["Latent feature", "h = G_f(x) in R^16"], 17, weight=700),
        arrow(x + 140, y + 78, x + 140, y + 135),
        arrow(x + 140, y + 213, x + 140, y + 290),
    ]


def add_classifier(parts: list[str], x: int = 930, y: int = 250, loss_y: int = 635, semi: bool = False) -> None:
    label_line = "source + labeled target" if semi else "source labels only"
    parts += [
        rect(x, y, 285, 95, "#dbeafe", "#2563eb"),
        multiline(x + 142, y + 35, ["Gas classifier G_y", "Linear 16 -> C", "C = 3"], 15, gap=18),
        rect(x, y + 165, 285, 110, "#eff6ff", "#2563eb"),
        multiline(x + 142, y + 203, ["Gas logits z_y", "softmax -> gas probability", "prediction: air / alcohol / acetone"], 14),
        arrow(x + 142, y + 95, x + 142, y + 165, "#2563eb"),
        rect(x - 10, loss_y, 305, 170, "#ffffff", "#2563eb"),
        multiline(
            x + 142,
            loss_y + 30,
            [
                "Classification loss",
                "z_y = G_y(G_f(x_i))",
                "p_i,k = softmax(z_y)_k",
                "L = -1/N_cls sum_i log p_i,y_i",
                f"labels: {label_line}",
            ],
            13,
            gap=19,
        ),
        arrow(x + 142, y + 275, x + 142, loss_y, "#2563eb"),
    ]


def add_domain_classifier(parts: list[str], x: int = 1290, y: int = 225, loss_y: int = 665) -> None:
    parts += [
        rect(x, y, 275, 82, "#fee2e2", "#dc2626"),
        multiline(x + 137, y + 32, ["Gradient reversal layer", "GRL_lambda(h): forward = h"], 15),
        rect(x, y + 135, 275, 96, "#fee2e2", "#dc2626"),
        multiline(x + 137, y + 170, ["Domain classifier G_d", "Linear 16 -> 16, ReLU", "Linear 16 -> 2"], 15),
        rect(x, y + 300, 275, 110, "#fff1f2", "#dc2626"),
        multiline(x + 137, y + 338, ["Domain logits z_d", "source vs target", "domain labels: 0 / 1"], 14),
        arrow(x + 137, y + 82, x + 137, y + 135, "#dc2626"),
        arrow(x + 137, y + 231, x + 137, y + 300, "#dc2626"),
        rect(x - 10, loss_y, 295, 160, "#ffffff", "#dc2626"),
        multiline(
            x + 137,
            loss_y + 30,
            [
                "Domain loss",
                "q_i,d = softmax(G_d(GRL(h_i)))_d",
                "y_d: source=0, target=1",
                "Ld = -1/N_d sum_i log q_i,y_d,i",
                "gas labels not used",
            ],
            12,
            gap=18,
        ),
        arrow(x + 137, y + 410, x + 137, loss_y, "#dc2626"),
    ]


def make_mlp() -> None:
    parts = svg_start(
        "Source-only MLP Baseline",
        "Three-class only: air / alcohol / acetone. No domain adaptation; test labels are used only after training.",
        h=990,
    )
    parts += [
        rect(45, 118, 360, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(465, 118, 385, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(905, 118, 355, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(1320, 118, 320, 740, "#f8fafc", "#d1d5db", 1, 12),
        text(225, 150, "1. Labeled Source", 20, 700),
        text(657, 150, "2. MLP Feature Layers", 20, 700),
        text(1082, 150, "3. Gas Prediction", 20, 700),
        text(1480, 150, "4. Evaluation", 20, 700),
        rect(85, 235, 280, 125, "#e0f2fe", "#0284c7"),
        multiline(225, 265, ["Training input", "X_s in R^(N_s x r)", "r=54, labels y_s available"], 16),
        rect(85, 435, 280, 120, "#f3f4f6", "#6b7280"),
        multiline(225, 470, ["Optional validation", "source/target val labels", "S1 has no validation -> NA"], 15),
        arrow(365, 295, 520, 295, "#0284c7"),
        arrow(365, 495, 520, 455, "#6b7280", dashed=True),
    ]
    add_common_feature_extractor(parts, 520, 250)
    parts += [arrow(800, 585, 930, 300, "#2563eb")]
    add_classifier(parts, 930, 250, 635)
    parts += [
        rect(1345, 255, 270, 125, "#ecfdf5", "#059669"),
        multiline(1480, 288, ["Test input", "X_t,test in R^(N_t x 54)", "test labels hidden during training"], 15),
        rect(1345, 425, 270, 90, "#ecfeff", "#0891b2"),
        multiline(1480, 458, ["Predicted labels", "y_hat_t = argmax softmax(z_y)"], 14),
        rect(1345, 585, 270, 135, "#ffffff", "#059669"),
        multiline(1480, 620, ["Report metrics", "compare y_hat_t with y_t,test", "accuracy, balanced acc.", "macro-F1, confusion matrix"], 14),
        arrow(1215, 470, 1345, 470, "#0891b2"),
        arrow(1480, 515, 1480, 585, "#059669"),
        rect(405, 885, 890, 58, "#f9fafb", "#111827"),
        text(850, 922, "Optimized objective in code: minimize L only. There is no source-target alignment term.", 18, 700),
    ]
    label_note(parts, 70, 615, ["Label usage", "y_s -> classification loss L", "validation labels -> model selection"], "#9333ea", 310, 98)
    arrow(350, 655, 920, 705, "#9333ea", dashed=True)
    label_note(parts, 1328, 735, ["Evaluation labels", "y_t,test is used only after training", "metrics = f(y_hat_t, y_t,test)"], "#059669", 295, 92)
    arrow(1480, 735, 1480, 720, "#059669", dashed=True)
    settings_note(parts, 40, 865, 360)
    write_svg("architecture_1_mlp_source_only.svg", parts)


def make_mlp_source_target_labels() -> None:
    parts = svg_start(
        "MLP Baseline with Source + Target Labels",
        "Three-class only. Supervised target-label calibration baseline; no adversarial domain alignment.",
        h=990,
    )
    parts += [
        rect(45, 118, 390, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(485, 118, 385, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(925, 118, 355, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(1335, 118, 320, 740, "#f8fafc", "#d1d5db", 1, 12),
        text(240, 150, "1. Labeled Source + Target", 20, 700),
        text(677, 150, "2. MLP Feature Layers", 20, 700),
        text(1102, 150, "3. Gas Prediction", 20, 700),
        text(1495, 150, "4. Evaluation", 20, 700),
        rect(88, 210, 305, 112, "#e0f2fe", "#0284c7"),
        multiline(240, 240, ["Source training labels", "X_s in R^(N_s x r)", "y_s available for L"], 15),
        rect(88, 370, 305, 128, "#dcfce7", "#16a34a"),
        multiline(240, 400, ["Selected target labels", "X_t,labeled, y_t,labeled", "Day subset defined by split", "used for L"], 14),
        rect(88, 565, 305, 118, "#f3f4f6", "#6b7280"),
        multiline(240, 598, ["Validation labels", "used for model selection only", "S1 has no validation -> NA"], 14),
        arrow(393, 265, 540, 290, "#0284c7"),
        arrow(393, 430, 540, 310, "#16a34a"),
        arrow(393, 620, 540, 455, "#6b7280", dashed=True),
    ]
    add_common_feature_extractor(parts, 540, 250)
    parts += [arrow(820, 585, 950, 300, "#2563eb")]
    add_classifier(parts, 950, 250, 635, semi=True)
    parts += [
        rect(1360, 255, 270, 125, "#ecfdf5", "#059669"),
        multiline(1495, 288, ["Held-out target test", "X_t,test in R^(N_t x 54)", "test labels hidden during training"], 15),
        rect(1360, 425, 270, 90, "#ecfeff", "#0891b2"),
        multiline(1495, 458, ["Predicted labels", "y_hat_t = argmax softmax(z_y)"], 14),
        rect(1360, 585, 270, 135, "#ffffff", "#059669"),
        multiline(1495, 620, ["Report metrics", "compare y_hat_t with y_t,test", "accuracy, balanced acc.", "macro-F1, confusion matrix"], 14),
        arrow(1235, 470, 1360, 470, "#0891b2"),
        arrow(1495, 515, 1495, 585, "#059669"),
        rect(405, 885, 920, 58, "#f9fafb", "#111827"),
        text(865, 922, "Optimized objective in code: minimize L on source + selected labeled target samples only.", 17, 700),
    ]
    label_note(parts, 70, 725, ["Classification labels", "y_cls = concat(y_s, y_t,labeled)", "used in L"], "#9333ea", 340, 82)
    arrow(410, 755, 940, 705, "#9333ea", dashed=True)
    label_note(parts, 1348, 735, ["No domain adaptation", "no G_d, no GRL, no Ld", "not a DANN model"], "#dc2626", 295, 92)
    label_note(parts, 72, 810, ["Target-test labels", "y_t,test used only after training", "final metrics only"], "#059669", 335, 76)
    settings_note(parts, 40, 865, 360)
    write_svg("architecture_2_mlp_source_target_labels.svg", parts)


def make_dann_uda() -> None:
    parts = svg_start(
        "DANN-UDA: Unsupervised Domain Adversarial Training",
        "Three-class only. Matches use_domain_loss=True and classifier_label_usage='source labels only'.",
    )
    parts += [
        rect(35, 118, 380, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(455, 118, 390, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(895, 118, 345, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(1285, 118, 345, 740, "#f8fafc", "#d1d5db", 1, 12),
        text(225, 150, "1. Source + Target", 20, 700),
        text(650, 150, "2. Shared G_f", 20, 700),
        text(1068, 150, "3. Gas Head", 20, 700),
        text(1458, 150, "4. Domain Head", 20, 700),
        rect(80, 210, 285, 120, "#e0f2fe", "#0284c7"),
        multiline(222, 240, ["Source", "X_s in R^(N_s x r)", "r=54, gas labels y_s -> L"], 15),
        rect(80, 390, 285, 120, "#ecfdf5", "#059669"),
        multiline(222, 420, ["Target", "X_t in R^(N_t x r)", "all S1-S7: features for Ld only"], 15),
        rect(80, 575, 285, 135, "#fef3c7", "#d97706"),
        multiline(222, 605, ["Domain feature batch", "X_d = concat(X_s, X_t)", "features only -> G_f"], 15),
        arrow(365, 270, 520, 282, "#0284c7"),
        arrow(365, 450, 520, 303, "#059669"),
        arrow(365, 640, 520, 322, "#d97706"),
    ]
    add_common_feature_extractor(parts, 520, 250)
    parts += [arrow(800, 585, 930, 300, "#2563eb"), arrow(800, 585, 1290, 265, "#dc2626")]
    add_classifier(parts, 930, 250, 665)
    add_domain_classifier(parts, 1290, 225, 665)
    parts += [
        rect(420, 890, 860, 58, "#f9fafb", "#111827"),
        text(850, 927, "Code computes backward scalar L + Ld; GRL makes G_f effectively minimize L - lambda * Ld.", 17, 700),
    ]
    label_note(parts, 908, 810, ["Gas labels", "y_cls = y_s", "used only in L"], "#2563eb", 300, 74)
    arrow(1060, 810, 1060, 792, "#2563eb", dashed=True)
    label_note(parts, 1278, 810, ["Domain labels", "y_dom = [0 for source, 1 for target]", "not input to G_f; used only in Ld"], "#dc2626", 340, 74)
    arrow(1450, 810, 1450, 800, "#dc2626", dashed=True)
    label_note(parts, 68, 735, ["Target gas labels", "not used in training", "only used later for test metrics"], "#059669", 310, 86)
    settings_note(parts, 25, 862, 430)
    write_svg("architecture_3_dann_uda.svg", parts)


def make_dann_semi() -> None:
    parts = svg_start(
        "DANN-semi: Semi-supervised Domain Adversarial Training",
        "Three-class only. Matches classifier_label_usage='source labels + selected target labels'.",
    )
    parts += [
        rect(35, 118, 390, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(465, 118, 385, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(895, 118, 345, 740, "#f8fafc", "#d1d5db", 1, 12),
        rect(1285, 118, 345, 740, "#f8fafc", "#d1d5db", 1, 12),
        text(230, 150, "1. Source + Labeled Target", 20, 700),
        text(657, 150, "2. Shared G_f", 20, 700),
        text(1068, 150, "3. Gas Head", 20, 700),
        text(1458, 150, "4. Domain Head", 20, 700),
        rect(80, 195, 300, 112, "#e0f2fe", "#0284c7"),
        multiline(230, 225, ["Source labels", "X_s, y_s", "always used for L"], 15),
        rect(80, 350, 300, 125, "#dcfce7", "#16a34a"),
        multiline(230, 380, ["Selected target labels", "X_t,labeled, y_t,labeled", "S5-S7 only; into y_cls"], 15),
        rect(80, 540, 300, 132, "#fef3c7", "#d97706"),
        multiline(230, 570, ["Domain feature batch", "X_d = concat(source, target)", "features only -> G_f", "no labels enter G_f"], 14),
        rect(80, 720, 300, 88, "#f3f4f6", "#6b7280"),
        multiline(230, 752, ["Target validation labels", "used for model selection", "not for final test"], 14),
        arrow(380, 250, 520, 282, "#0284c7"),
        arrow(380, 410, 520, 303, "#16a34a"),
        arrow(380, 600, 520, 322, "#d97706"),
    ]
    add_common_feature_extractor(parts, 520, 250)
    parts += [arrow(800, 585, 930, 300, "#2563eb"), arrow(800, 585, 1290, 265, "#dc2626")]
    add_classifier(parts, 930, 250, 665, semi=True)
    add_domain_classifier(parts, 1290, 225, 665)
    parts += [
        rect(395, 890, 910, 58, "#f9fafb", "#111827"),
        text(850, 927, "Code computes L + Ld; through GRL, G_f effectively minimizes L(source + labeled target) - lambda * Ld.", 16, 700),
    ]
    label_note(parts, 902, 810, ["Gas labels", "y_cls = concat(y_s, y_t,labeled)", "used in classification loss L"], "#2563eb", 320, 74)
    arrow(1060, 810, 1060, 792, "#2563eb", dashed=True)
    label_note(parts, 1268, 810, ["Domain labels", "source/target labels y_dom", "not input to G_f; used only in Ld"], "#dc2626", 350, 74)
    arrow(1450, 810, 1450, 800, "#dc2626", dashed=True)
    label_note(parts, 80, 812, ["Validation labels", "used every 10 epochs", "select best macro-F1 then accuracy"], "#6b7280", 300, 76)
    settings_note(parts, 20, 862, 430)
    write_svg("architecture_4_dann_semi.svg", parts)


def make_cdann() -> None:
    parts = svg_start(
        "C-DANN: Class-conditional Domain Alignment",
        "Three-class only. DANN loss plus class-wise latent centroid alignment Lc.",
        w=1900,
        h=1040,
    )
    parts += [
        rect(35, 118, 380, 770, "#f8fafc", "#d1d5db", 1, 12),
        rect(455, 118, 385, 770, "#f8fafc", "#d1d5db", 1, 12),
        rect(885, 118, 320, 770, "#f8fafc", "#d1d5db", 1, 12),
        rect(1245, 118, 300, 770, "#f8fafc", "#d1d5db", 1, 12),
        rect(1585, 118, 280, 770, "#f8fafc", "#d1d5db", 1, 12),
        text(225, 150, "1. Source + Target", 20, 700),
        text(647, 150, "2. Shared G_f", 20, 700),
        text(1045, 150, "3. Gas Head", 20, 700),
        text(1395, 150, "4. Domain Head", 20, 700),
        text(1725, 150, "5. Conditional Alignment", 20, 700),
        rect(80, 205, 285, 118, "#e0f2fe", "#0284c7"),
        multiline(222, 235, ["Source", "X_s in R^(N_s x r)", "r=54, gas labels y_s"], 15),
        rect(80, 375, 285, 118, "#ecfdf5", "#059669"),
        multiline(222, 405, ["Target", "X_t in R^(N_t x r)", "UDA: predicted class prob. for Lc"], 15),
        rect(80, 550, 285, 130, "#fef3c7", "#d97706"),
        multiline(222, 580, ["Domain feature batch", "X_d = concat(X_s, X_t)", "features only -> G_f"], 15),
        rect(80, 730, 285, 95, "#dcfce7", "#16a34a"),
        multiline(222, 762, ["Semi option", "S5-S7 labeled target subset", "can replace predictions"], 14),
        arrow(365, 265, 520, 282, "#0284c7"),
        arrow(365, 435, 520, 303, "#059669"),
        arrow(365, 615, 520, 322, "#d97706"),
        arrow(365, 775, 520, 326, "#16a34a", dashed=True),
    ]
    add_common_feature_extractor(parts, 520, 250)
    parts += [arrow(800, 585, 915, 300, "#2563eb"), arrow(800, 585, 1275, 265, "#dc2626"), arrow(800, 585, 1615, 320, "#7c3aed", dashed=True)]
    add_classifier(parts, 915, 250, 675)
    add_domain_classifier(parts, 1275, 225, 675)
    parts += [
        rect(1615, 260, 220, 115, "#f5f3ff", "#7c3aed"),
        multiline(1725, 296, ["Class-wise latent", "centroid matching", "mu_s,k vs mu_t,k"], 15),
        rect(1615, 455, 220, 155, "#faf5ff", "#7c3aed"),
        multiline(
            1725,
            488,
            [
                "Centroid loss",
                "mu_s,k = mean h_s | y_s=k",
                "mu_t,k = weighted mean h_t",
                "Lc = 1/|K| sum_k ||mu_s,k",
                "- mu_t,k||_2^2",
            ],
            12,
            gap=18,
        ),
        arrow(1725, 375, 1725, 465, "#7c3aed", dashed=True),
        rect(1615, 675, 220, 130, "#ffffff", "#7c3aed"),
        multiline(1725, 710, ["Target class weights", "UDA: predicted prob.", "semi: known labels"], 14),
        arrow(1725, 590, 1725, 675, "#7c3aed", dashed=True),
        rect(440, 930, 1020, 65, "#f9fafb", "#111827"),
        text(950, 967, "Code computes L + Ld + alpha * Lc; through GRL, G_f effectively minimizes L - lambda * Ld + alpha * Lc.", 16, 700),
        text(950, 992, "Code selects lambda/alpha by validation metrics averaged over seeds; S1 uses fixed defaults and shows validation as NA.", 14, 400, "#374151"),
    ]
    label_note(parts, 895, 810, ["Gas labels for L", "UDA: y_cls = y_s", "semi: concat(y_s, y_t,labeled)"], "#2563eb", 330, 82)
    arrow(1045, 810, 1045, 792, "#2563eb", dashed=True)
    label_note(parts, 1248, 810, ["Domain labels for Ld", "y_dom = source/target", "not input to G_f", "no gas labels in domain loss"], "#dc2626", 330, 82)
    arrow(1395, 810, 1395, 800, "#dc2626", dashed=True)
    label_note(parts, 1588, 835, ["Class weights for Lc", "source uses one-hot y_s", "target uses predicted prob.; semi can use target labels"], "#7c3aed", 276, 86)
    arrow(1725, 835, 1725, 805, "#7c3aed", dashed=True)
    settings_note(parts, 35, 900, 500, include_alpha=True)
    write_svg("architecture_5_cdann_conditional.svg", parts)


def main() -> None:
    make_mlp()
    make_mlp_source_target_labels()
    make_dann_uda()
    make_dann_semi()
    make_cdann()


if __name__ == "__main__":
    main()
