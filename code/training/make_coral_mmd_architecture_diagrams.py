"""Generate Deep CORAL and MK-MMD research flow diagrams in the DANN style."""

from __future__ import annotations

from make_dann_uda_architecture_diagram import (
    OUT_DIR,
    arrow,
    label_note,
    multiline,
    rect,
    svg_start,
    text,
    write_svg,
)
import shared_experiment_protocol as protocol


WIDTH = 1900
HEIGHT = 1040


def add_columns(parts: list[str], alignment_title: str) -> None:
    columns = [
        (35, 380, "1. Source + Target"),
        (455, 385, "2. Shared Feature Extractor"),
        (885, 320, "3. Gas Head"),
        (1245, 300, "4. Latent Distributions"),
        (1585, 280, alignment_title),
    ]
    for x, width, title in columns:
        parts.append(rect(x, 118, width, 770, "#f8fafc", "#d1d5db", 1, 12))
        parts.append(text(x + width // 2, 150, title, 20, 700))


def add_inputs(parts: list[str], alignment_symbol: str) -> None:
    parts += [
        rect(80, 205, 285, 118, "#e0f2fe", "#0284c7"),
        multiline(222, 235, ["Labeled source", "X_s in R^(N_s x 54)", "gas labels y_s -> L"], 15),
        rect(80, 375, 285, 118, "#ecfdf5", "#059669"),
        multiline(222, 405, ["Target domain", "X_t in R^(N_t x 54)", f"UDA: features only -> {alignment_symbol}"], 15),
        rect(80, 550, 285, 130, "#fef3c7", "#d97706"),
        multiline(222, 580, ["Shared feature inputs", "X_s and X_t use the same G_f", "no domain classifier", "no domain labels"], 14),
        rect(80, 730, 285, 95, "#dcfce7", "#16a34a"),
        multiline(222, 760, ["Semi option", "labeled target trials join L", "alignment still uses h_s and h_t"], 14),
        arrow(365, 265, 500, 280, "#0284c7"),
        arrow(365, 435, 500, 305, "#059669"),
        arrow(365, 615, 500, 325, "#d97706"),
    ]


def add_feature_extractor(parts: list[str]) -> None:
    parts += [
        rect(500, 225, 295, 98, "#eef2ff", "#4f46e5"),
        multiline(647, 255, ["Shared feature extractor G_f", "one-layer: 54 -> 16", "two-layer: 54 -> 32 -> 16"], 14, gap=19),
        rect(500, 385, 295, 88, "#eef2ff", "#4f46e5"),
        multiline(647, 418, ["ReLU + Dropout 0.05", "same parameters for both domains"], 14),
        rect(500, 555, 295, 110, "#ede9fe", "#7c3aed"),
        multiline(647, 588, ["Latent representations", "h_s = G_f(X_s) in R^(N_s x 16)", "h_t = G_f(X_t) in R^(N_t x 16)"], 14, gap=19, weight=700),
        arrow(647, 323, 647, 385),
        arrow(647, 473, 647, 555),
    ]


def add_gas_head(parts: list[str]) -> None:
    parts += [
        rect(910, 245, 270, 98, "#dbeafe", "#2563eb"),
        multiline(1045, 278, ["Gas classifier G_y", "Linear 16 -> C", "C = 3"], 15, gap=19),
        rect(910, 410, 270, 112, "#eff6ff", "#2563eb"),
        multiline(1045, 446, ["Gas logits z_y", "softmax -> gas probability", "air / alcohol / acetone"], 14, gap=19),
        rect(900, 650, 290, 165, "#ffffff", "#2563eb"),
        multiline(
            1045,
            682,
            [
                "Classification loss L",
                "p_i = softmax(G_y(h_i))",
                "L = CE(p_s, y_s) in UDA",
                "Semi: CE over source + labeled target",
                "unlabeled target gas labels are not used",
            ],
            13,
            gap=20,
        ),
        arrow(1045, 343, 1045, 410, "#2563eb"),
        arrow(1045, 522, 1045, 650, "#2563eb"),
        arrow(795, 610, 910, 292, "#2563eb"),
    ]


def add_latent_distributions(parts: list[str]) -> None:
    parts += [
        rect(1270, 245, 250, 105, "#ede9fe", "#7c3aed"),
        multiline(1395, 278, ["Source latent set", "H_s = {h_s,i}", "shape N_s x 16"], 15, gap=19),
        rect(1270, 430, 250, 105, "#ecfdf5", "#059669"),
        multiline(1395, 463, ["Target latent set", "H_t = {h_t,j}", "shape N_t x 16"], 15, gap=19),
        rect(1270, 630, 250, 135, "#faf5ff", "#7c3aed"),
        multiline(1395, 663, ["Alignment inputs", "computed from h_s and h_t", "N_s and N_t may differ", "gas labels not required"], 13, gap=19),
        arrow(795, 598, 1270, 300, "#7c3aed", dashed=True),
        arrow(795, 625, 1270, 482, "#059669", dashed=True),
        arrow(1395, 350, 1395, 630, "#7c3aed", dashed=True),
        arrow(1395, 535, 1395, 630, "#059669", dashed=True),
    ]


def add_settings(parts: list[str], lambda_values: str, source: str) -> None:
    seeds = ", ".join(str(seed) for seed in protocol.MODEL_SEEDS)
    parts += [
        rect(35, 905, 510, 105, "#f9fafb", "#6b7280", 1, 8),
        multiline(
            290,
            928,
            [
                "Training settings in unified suite",
                "epochs=400, Adam, lr=0.01, weight_decay=1e-4",
                f"seeds: {seeds}; lambda candidates: {lambda_values}",
                "UDA: source gas labels only; Semi: Source+Target refit",
                source,
            ],
            12,
            "#374151",
            gap=16,
        ),
    ]


def make_deep_coral() -> None:
    parts = svg_start(
        "Deep CORAL: Correlation Alignment for Sensor-drift Adaptation",
        "Shared G_f learns gas-discriminative features while matching source and target covariance matrices.",
        w=WIDTH,
        h=HEIGHT,
    )
    add_columns(parts, "5. CORAL Alignment")
    add_inputs(parts, "L_coral")
    add_feature_extractor(parts)
    add_gas_head(parts)
    add_latent_distributions(parts)
    parts += [
        rect(1608, 220, 235, 120, "#f5f3ff", "#7c3aed"),
        multiline(1725, 252, ["Domain covariance", "C_s = cov(H_s)", "C_t = cov(H_t)", "each matrix: 16 x 16"], 14, gap=19),
        rect(1608, 420, 235, 175, "#faf5ff", "#7c3aed"),
        multiline(
            1725,
            452,
            [
                "Deep CORAL loss",
                "L_coral = ||C_s - C_t||_F^2",
                "/ (4 d^2)",
                "d = 16",
                "aligns second-order statistics",
            ],
            13,
            gap=21,
        ),
        arrow(1520, 300, 1608, 280, "#7c3aed", dashed=True),
        arrow(1520, 482, 1608, 300, "#059669", dashed=True),
        arrow(1725, 340, 1725, 420, "#7c3aed"),
        rect(1600, 680, 250, 125, "#ffffff", "#7c3aed"),
        multiline(1725, 716, ["Optimized scalar", "L_total = L + lambda_coral L_coral", "no GRL", "no domain discriminator"], 14, gap=20, weight=700),
        arrow(1725, 595, 1725, 680, "#7c3aed"),
        rect(565, 918, 1285, 62, "#f9fafb", "#111827"),
        text(1207, 950, "G_f minimizes classification error and covariance mismatch jointly; target gas labels are unnecessary in UDA.", 16, 700),
    ]
    label_note(parts, 895, 825, ["Gas-label usage", "UDA: y_s only -> L", "Semi: source + labeled target -> L"], "#2563eb", 300, 76)
    label_note(parts, 1598, 825, ["Alignment-label usage", "L_coral uses H_s and H_t only", "no gas/domain labels"], "#7c3aed", 255, 76)
    add_settings(parts, "0.001, 0.01, 0.1, 1.0", "loss follows VisionLearningGroup/Deep CORAL")
    write_svg("architecture_7_deep_coral.svg", parts)


def make_mk_mmd() -> None:
    parts = svg_start(
        "MK-MMD: Multi-kernel Distribution Alignment for Sensor Drift",
        "Shared G_f minimizes a multi-kernel maximum mean discrepancy between source and target latent distributions.",
        w=WIDTH,
        h=HEIGHT,
    )
    add_columns(parts, "5. MK-MMD Alignment")
    add_inputs(parts, "L_mmd")
    add_feature_extractor(parts)
    add_gas_head(parts)
    add_latent_distributions(parts)
    parts += [
        rect(1608, 205, 235, 145, "#f5f3ff", "#7c3aed"),
        multiline(1725, 237, ["Gaussian kernel bank", "k = sum_u k_u", "alpha = 0.5, 1.0, 2.0", "bandwidth from latent", "pairwise distances"], 13, gap=20),
        rect(1608, 420, 235, 190, "#faf5ff", "#7c3aed"),
        multiline(
            1725,
            451,
            [
                "MK-MMD loss",
                "L_mmd = mean K_ss",
                "+ mean K_tt",
                "- 2 mean K_st",
                "matches kernel mean embeddings",
                "supports N_s != N_t",
            ],
            13,
            gap=21,
        ),
        arrow(1520, 300, 1608, 270, "#7c3aed", dashed=True),
        arrow(1520, 482, 1608, 310, "#059669", dashed=True),
        arrow(1725, 350, 1725, 420, "#7c3aed"),
        rect(1600, 680, 250, 125, "#ffffff", "#7c3aed"),
        multiline(1725, 716, ["Optimized scalar", "L_total = L + lambda_mmd L_mmd", "no GRL", "no domain discriminator"], 14, gap=20, weight=700),
        arrow(1725, 610, 1725, 680, "#7c3aed"),
        rect(565, 918, 1285, 62, "#f9fafb", "#111827"),
        text(1207, 950, "G_f minimizes classification error and multi-kernel distribution distance jointly; target labels are unnecessary in UDA.", 16, 700),
    ]
    label_note(parts, 895, 825, ["Gas-label usage", "UDA: y_s only -> L", "Semi: source + labeled target -> L"], "#2563eb", 300, 76)
    label_note(parts, 1598, 825, ["Alignment-label usage", "L_mmd uses H_s and H_t only", "no gas/domain labels"], "#7c3aed", 255, 76)
    add_settings(parts, "0.01, 0.1, 1.0", "kernel design follows THU DAN / Transfer-Learning-Library")
    write_svg("architecture_8_mk_mmd.svg", parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_deep_coral()
    make_mk_mmd()


if __name__ == "__main__":
    main()
