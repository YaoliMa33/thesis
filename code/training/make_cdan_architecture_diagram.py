from __future__ import annotations

import html
from pathlib import Path

import shared_experiment_protocol as protocol


OUT = Path(r"D:\thesis\figures\dann_learning\architecture_6_official_cdan.svg")


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text(x: int, y: int, value: str, size: int = 18, weight: int = 400, color: str = "#111827", anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def multiline(x: int, y: int, lines: list[str], size: int = 16, color: str = "#111827", gap: int = 22, weight: int = 400) -> str:
    return "\n".join(text(x, y + i * gap, line, size, weight, color) for i, line in enumerate(lines))


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: int = 2, rx: int = 8) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#374151", sw: int = 2, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
        f'marker-end="url(#arrow)"{dash}/>'
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1980, 1040
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
        '<path d="M2,2 L10,6 L2,10 Z" fill="#374151"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(990, 46, "Official CDAN-style Conditional Adversarial Domain Adaptation", 28, 700),
        text(990, 76, "Three-class only. CDAN conditions the domain discriminator on latent feature h and gas probability g: G_cd(T(h,g)).", 15, 400, "#4b5563"),
        rect(30, 115, 350, 800, "#f8fafc", "#d1d5db", 1, 12),
        rect(420, 115, 330, 800, "#f8fafc", "#d1d5db", 1, 12),
        rect(790, 115, 300, 800, "#f8fafc", "#d1d5db", 1, 12),
        rect(1130, 115, 360, 800, "#f8fafc", "#d1d5db", 1, 12),
        rect(1530, 115, 390, 800, "#f8fafc", "#d1d5db", 1, 12),
        text(205, 150, "1. Inputs", 20, 700),
        text(585, 150, "2. Shared Feature Extractor", 20, 700),
        text(940, 150, "3. Gas Head", 20, 700),
        text(1310, 150, "4. CDAN Conditioning", 20, 700),
        text(1725, 150, "5. Conditional Domain Head", 20, 700),
        rect(65, 205, 280, 112, "#e0f2fe", "#0284c7"),
        multiline(205, 235, ["Source features", "X_s in R^(N_s x r)", "r=54, gas labels y_s -> L"], 15),
        rect(65, 375, 280, 112, "#ecfdf5", "#059669"),
        multiline(205, 405, ["Target features", "X_t in R^(N_t x r)", "all S1-S7: features for Ld_cdan"], 15),
        rect(65, 555, 280, 128, "#fef3c7", "#d97706"),
        multiline(205, 585, ["Domain feature batch", "X_d = concat(X_s, X_t)", "features only -> G_f", "labels do not enter G_f"], 14),
        rect(65, 745, 280, 92, "#dcfce7", "#16a34a"),
        multiline(205, 778, ["Semi option", "S5-S7 X_t,labeled, y_t,labeled", "can be added to L"], 14),
        rect(455, 230, 260, 78, "#eef2ff", "#4f46e5"),
        multiline(585, 260, ["G_f layer 1", "Linear r -> 32, ReLU", "r = 54 in this report"], 14, gap=18),
        rect(455, 370, 260, 78, "#eef2ff", "#4f46e5"),
        multiline(585, 400, ["G_f layer 2", "Dropout 0.05, Linear 32 -> 16, ReLU"], 14),
        rect(455, 540, 260, 88, "#ede9fe", "#7c3aed"),
        multiline(585, 574, ["Latent feature", "h = G_f(x) in R^16", "h_s, h_t, h_d"], 16, weight=700),
        arrow(585, 308, 585, 370),
        arrow(585, 448, 585, 540),
        arrow(345, 260, 455, 270, "#0284c7"),
        arrow(345, 430, 455, 292, "#059669"),
        arrow(345, 615, 455, 306, "#d97706"),
        rect(820, 245, 240, 90, "#dbeafe", "#2563eb"),
        multiline(940, 278, ["Gas classifier G_y", "Linear 16 -> C", "C = 3"], 15, gap=18),
        rect(820, 420, 240, 112, "#eff6ff", "#2563eb"),
        multiline(940, 455, ["Gas logits z_y", "g = softmax(z_y)", "g in R^C"], 15),
        rect(805, 650, 270, 160, "#ffffff", "#2563eb"),
        multiline(
            940,
            680,
            [
                "Classification loss",
                "z_y = G_y(G_f(x_i))",
                "p_i,k = softmax(z_y)_k",
                "L = -1/N_cls sum_i log p_i,y_i",
                "UDA: y_i from source only",
                "semi: source + labeled target",
            ],
            11,
            gap=18,
        ),
        arrow(715, 585, 820, 285, "#2563eb"),
        arrow(940, 335, 940, 420, "#2563eb"),
        arrow(940, 532, 940, 660, "#2563eb"),
        rect(1170, 245, 280, 108, "#f5f3ff", "#7c3aed"),
        multiline(1310, 280, ["Official CDAN map", "T(h,g) = g outer h", "shape: N x (C*16)"], 16),
        rect(1170, 455, 280, 126, "#faf5ff", "#7c3aed"),
        multiline(1310, 492, ["Three-class report", "C=3", "T in R^(N x 48)", "task = three_class_with_air"], 15),
        rect(1170, 690, 280, 92, "#fdf4ff", "#7c3aed"),
        multiline(1310, 722, ["Prediction handling", "g is detached for Ld_cdan", "L still updates G_y normally"], 14),
        arrow(715, 585, 1170, 300, "#7c3aed", dashed=True),
        arrow(1060, 475, 1170, 300, "#7c3aed", dashed=True),
        arrow(1310, 353, 1310, 455, "#7c3aed"),
        arrow(1310, 581, 1310, 690, "#7c3aed"),
        rect(1575, 230, 300, 82, "#fee2e2", "#dc2626"),
        multiline(1725, 262, ["Gradient reversal", "GRL_lambda(T): forward = T"], 15),
        rect(1575, 370, 300, 105, "#fee2e2", "#dc2626"),
        multiline(1725, 407, ["Conditional domain discriminator", "G_cd: C*16 -> 32, ReLU", "32 -> 2"], 15),
        rect(1575, 550, 300, 108, "#fff1f2", "#dc2626"),
        multiline(1725, 587, ["Domain logits z_cd", "source vs target", "domain labels y_dom: 0 / 1"], 15),
        rect(1565, 720, 320, 160, "#ffffff", "#dc2626"),
        multiline(
            1725,
            750,
            [
                "CDAN domain loss",
                "u_i = GRL_lambda(T(h_i,g_i))",
                "q_i,d = softmax(G_cd(u_i))_d",
                "Ld_cdan = -1/N_d sum_i log q_i,y_dom,i",
                "gas labels not used in Ld_cdan",
            ],
            12,
            gap=18,
        ),
        arrow(1450, 300, 1575, 270, "#dc2626"),
        arrow(1725, 312, 1725, 370, "#dc2626"),
        arrow(1725, 475, 1725, 550, "#dc2626"),
        arrow(1725, 658, 1725, 735, "#dc2626"),
        rect(1120, 925, 620, 62, "#f9fafb", "#111827"),
        text(1430, 950, "Code computes L + Ld_cdan; through GRL, G_f effectively minimizes L - lambda * Ld_cdan.", 16, 700),
        text(1430, 978, "CDAN+C-DANN in code adds the ordinary DANN domain head G_d and centroid loss Lc from the C-DANN diagram.", 14, 400, "#374151"),
        rect(35, 910, 520, 112, "#f9fafb", "#6b7280", 1, 8),
        multiline(
            295,
            934,
            [
                "Training settings",
                "epochs=400, Adam, lr=0.01, weight_decay=1e-4",
                f"seeds: {', '.join(str(seed) for seed in protocol.MODEL_SEEDS)}; lambda: 0.05, 0.2, 0.5",
                "with validation: grid search and validation selection",
                "S1 has no validation: fixed defaults; val metrics = NA",
            ],
            12,
            "#374151",
            16,
        ),
        "</svg>",
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
