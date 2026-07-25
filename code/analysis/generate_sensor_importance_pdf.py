from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import PIL.JpegImagePlugin
import PIL.PdfImagePlugin


def locate_output_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "tables").exists() and (parent / "processed").exists():
            return parent
        if (parent / "thesis_out" / "tables").exists() and (parent / "thesis_out" / "processed").exists():
            return parent / "thesis_out"
    return path.parents[2]


OUT = locate_output_root()
TABLES = OUT / "tables"
PROCESSED = OUT / "processed"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

PDF_PATH = REPORTS / "sensor_importance_detailed_report_cn.pdf"
MD_PATH = REPORTS / "sensor_importance_detailed_report_cn.md"
PREVIEW_DIR = REPORTS / "sensor_importance_pdf_pages"

PAGE_W, PAGE_H = 1240, 1754
MARGIN_X = 92
MARGIN_TOP = 82
MARGIN_BOTTOM = 76
CONTENT_W = PAGE_W - 2 * MARGIN_X

INK = "#1f2937"
MUTED = "#4b5563"
BLUE = "#1f4e79"
LIGHT_BLUE = "#e8f0f7"
LIGHT_GRAY = "#f3f4f6"
BORDER = "#cbd5e1"
RED = "#b91c1c"
GREEN = "#047857"


def font_path() -> str:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "arial.ttf"


FONT = font_path()


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        for candidate in [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\arialbd.ttf"]:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
    return ImageFont.truetype(FONT, size=size)


FONT_TITLE = f(38, True)
FONT_H1 = f(27, True)
FONT_H2 = f(22, True)
FONT_BODY = f(18)
FONT_BODY_BOLD = f(18, True)
FONT_SMALL = f(15)
FONT_TABLE = f(15)
FONT_TABLE_BOLD = f(15, True)
FONT_FOOTER = f(13)


class PdfReport:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP

    def _new_page(self) -> Image.Image:
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(page)
        draw.rectangle([0, 0, PAGE_W, 34], fill=BLUE)
        return page

    def new_page(self) -> None:
        self._footer()
        self.pages.append(self.page)
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP

    def ensure(self, h: int) -> None:
        if self.y + h > PAGE_H - MARGIN_BOTTOM:
            self.new_page()

    def _footer(self) -> None:
        n = len(self.pages) + 1
        self.draw.line([MARGIN_X, PAGE_H - 58, PAGE_W - MARGIN_X, PAGE_H - 58], fill=BORDER, width=1)
        self.draw.text((MARGIN_X, PAGE_H - 46), "Air / Alcohol / Acetone eNose sensor importance report", font=FONT_FOOTER, fill=MUTED)
        self.draw.text((PAGE_W - MARGIN_X - 42, PAGE_H - 46), str(n), font=FONT_FOOTER, fill=MUTED)

    def save(self, pdf_path: Path, preview_dir: Path) -> None:
        self._footer()
        self.pages.append(self.page)
        preview_dir.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(self.pages, start=1):
            page.save(preview_dir / f"page-{i:02d}.png")
        self.pages[0].save(pdf_path, save_all=True, append_images=self.pages[1:], resolution=150.0)

    def text_width(self, text: str, font: ImageFont.FreeTypeFont) -> float:
        return self.draw.textlength(text, font=font)

    def wrap(self, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
        lines: list[str] = []
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                lines.append("")
                continue
            current = ""
            for ch in para:
                candidate = current + ch
                if self.text_width(candidate, font) <= width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = ch
            if current:
                lines.append(current)
        return lines

    def title(self, text: str, subtitle: str) -> None:
        self.ensure(210)
        self.draw.text((MARGIN_X, self.y), text, font=FONT_TITLE, fill=BLUE)
        self.y += 58
        for line in self.wrap(subtitle, FONT_BODY, CONTENT_W):
            self.draw.text((MARGIN_X, self.y), line, font=FONT_BODY, fill=MUTED)
            self.y += 28
        self.y += 20
        self.draw.rectangle([MARGIN_X, self.y, PAGE_W - MARGIN_X, self.y + 4], fill=BLUE)
        self.y += 34

    def h1(self, text: str) -> None:
        self.ensure(58)
        self.y += 10
        self.draw.text((MARGIN_X, self.y), text, font=FONT_H1, fill=BLUE)
        self.y += 42

    def h2(self, text: str) -> None:
        self.ensure(42)
        self.y += 8
        self.draw.text((MARGIN_X, self.y), text, font=FONT_H2, fill=INK)
        self.y += 34

    def p(self, text: str, color: str = INK, gap: int = 10) -> None:
        lines = self.wrap(text, FONT_BODY, CONTENT_W)
        self.ensure(len(lines) * 27 + gap)
        for line in lines:
            self.draw.text((MARGIN_X, self.y), line, font=FONT_BODY, fill=color)
            self.y += 27
        self.y += gap

    def bullet(self, text: str) -> None:
        lines = self.wrap(text, FONT_BODY, CONTENT_W - 38)
        self.ensure(len(lines) * 27 + 8)
        self.draw.ellipse([MARGIN_X + 5, self.y + 10, MARGIN_X + 13, self.y + 18], fill=BLUE)
        for i, line in enumerate(lines):
            self.draw.text((MARGIN_X + 34, self.y), line, font=FONT_BODY, fill=INK)
            self.y += 27
        self.y += 5

    def callout(self, title: str, text: str) -> None:
        lines = self.wrap(text, FONT_BODY, CONTENT_W - 44)
        h = 46 + len(lines) * 27
        self.ensure(h + 16)
        x0, y0 = MARGIN_X, self.y
        self.draw.rectangle([x0, y0, PAGE_W - MARGIN_X, y0 + h], fill=LIGHT_BLUE, outline="#9fb9d2", width=2)
        self.draw.text((x0 + 22, y0 + 16), title, font=FONT_BODY_BOLD, fill=BLUE)
        yy = y0 + 48
        for line in lines:
            self.draw.text((x0 + 22, yy), line, font=FONT_BODY, fill=INK)
            yy += 27
        self.y += h + 18

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
        col_w = [int(CONTENT_W * w) for w in widths]
        col_w[-1] += CONTENT_W - sum(col_w)
        x_positions = [MARGIN_X]
        for w in col_w[:-1]:
            x_positions.append(x_positions[-1] + w)

        def row_height(cells: list[str], font: ImageFont.FreeTypeFont) -> int:
            max_lines = 1
            for cell, w in zip(cells, col_w):
                max_lines = max(max_lines, len(self.wrap(str(cell), font, w - 18)))
            return max(42, max_lines * 22 + 18)

        header_h = row_height(headers, FONT_TABLE_BOLD)
        self.ensure(header_h + 20)
        self._draw_row(headers, x_positions, col_w, header_h, FONT_TABLE_BOLD, fill=LIGHT_GRAY)
        for row in rows:
            h = row_height(row, FONT_TABLE)
            self.ensure(h + 10)
            self._draw_row(row, x_positions, col_w, h, FONT_TABLE, fill="white")
        self.y += 18

    def _draw_row(self, cells, x_positions, col_w, h, font, fill) -> None:
        y0 = self.y
        for i, cell in enumerate(cells):
            x0 = x_positions[i]
            self.draw.rectangle([x0, y0, x0 + col_w[i], y0 + h], fill=fill, outline=BORDER, width=1)
            yy = y0 + 9
            for line in self.wrap(str(cell), font, col_w[i] - 18):
                self.draw.text((x0 + 9, yy), line, font=font, fill=INK)
                yy += 21
        self.y += h

    def image(self, path: Path, max_h: int = 410, caption: str | None = None) -> None:
        img = Image.open(path).convert("RGB")
        scale = min(CONTENT_W / img.width, max_h / img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        self.ensure(new_size[1] + (42 if caption else 18))
        x = MARGIN_X + (CONTENT_W - new_size[0]) // 2
        self.page.paste(img, (x, self.y))
        self.y += new_size[1] + 8
        if caption:
            for line in self.wrap(caption, FONT_SMALL, CONTENT_W):
                self.draw.text((MARGIN_X, self.y), line, font=FONT_SMALL, fill=MUTED)
                self.y += 22
        self.y += 12


def draw_bar_chart(summary: pd.DataFrame, path: Path) -> None:
    df = summary.copy()
    df["delta"] = df["random_holdout_lda_delta_bacc"].astype(float)
    df = df.sort_values("delta", ascending=True)
    img = Image.new("RGB", (1040, 430), "white")
    d = ImageDraw.Draw(img)
    d.text((28, 20), "Leave-one-sensor-out marginal contribution (LDA, random blind holdout)", font=f(24, True), fill=BLUE)
    x0, y0, w, h = 230, 78, 720, 270
    d.line([x0 + w // 2, y0 - 8, x0 + w // 2, y0 + h], fill="#111827", width=2)
    max_abs = max(0.01, float(df["delta"].abs().max()))
    for i, row in enumerate(df.itertuples(index=False)):
        y = y0 + i * 42
        val = float(row.delta)
        bar_w = int(abs(val) / max_abs * (w * 0.45))
        if val >= 0:
            x = x0 + w // 2
            color = "#2563eb"
        else:
            x = x0 + w // 2 - bar_w
            color = "#dc2626"
        d.text((52, y + 5), str(row.removed_sensor).upper(), font=f(18, True), fill=INK)
        d.rectangle([x, y, x + bar_w, y + 25], fill=color)
        d.text((x + bar_w + 8 if val >= 0 else x - 75, y + 3), f"{val:+.3f}", font=f(16), fill=INK)
    d.text((28, 378), "Positive means removing that sensor hurts balanced accuracy. Negative means removal does not hurt or slightly helps.", font=f(15), fill=MUTED)
    img.save(path)


def draw_pca(scores: pd.DataFrame, path: Path) -> None:
    img = Image.new("RGB", (1040, 650), "white")
    d = ImageDraw.Draw(img)
    d.text((28, 20), "Baseline-corrected feature space, PCA projection (PC1 vs PC2)", font=f(24, True), fill=BLUE)
    x0, y0, w, h = 100, 75, 820, 460
    d.rectangle([x0, y0, x0 + w, y0 + h], fill="#f9fafb", outline=BORDER, width=2)
    colors = {"air": "#111827", "alcohol": "#2563eb", "acetone": "#dc2626"}
    xs = scores["PC1"].to_numpy(float)
    ys = scores["PC2"].to_numpy(float)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    def sx(v: float) -> int:
        return int(x0 + (v - xmin) / max(1e-9, xmax - xmin) * w)

    def sy(v: float) -> int:
        return int(y0 + h - (v - ymin) / max(1e-9, ymax - ymin) * h)

    for row in scores.itertuples(index=False):
        x, y = sx(row.PC1), sy(row.PC2)
        c = colors[row.label]
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=c, outline="white")
    d.text((x0 + w // 2 - 20, y0 + h + 22), "PC1", font=f(16), fill=INK)
    d.text((34, y0 + h // 2), "PC2", font=f(16), fill=INK)
    lx = 140
    for label in ["air", "alcohol", "acetone"]:
        d.ellipse([lx, 585, lx + 14, 599], fill=colors[label])
        d.text((lx + 22, 580), label, font=f(16), fill=INK)
        lx += 150
    img.save(path)


def fmt(x: float) -> str:
    return f"{x:.3f}"


def build_markdown(summary: pd.DataFrame) -> str:
    return f"""# Air / Alcohol / Acetone 电子鼻传感器重要性分析报告

本报告使用 `C:\\Users\\yaoli\\Documents\\DCAE\\enose_data` 中 air、alcohol、acetone 的全部 198 条 CSV 数据。所有传感器数据均进行了 baseline correction，并按照 180/125/130 秒采集协议划分 clean air、gas exposure、gas removal/recovery 三个阶段。

核心结论：S2 和 S3 是本任务中最重要的边际传感器；S1 和 S5 有明显响应但在全模型中存在冗余；S4 边际价值较低；S6 对区分三类气体基本不重要，默认应从训练和 raw 3D 坐标图中排除。

主要输出文件：

- `reports/sensor_importance_detailed_report_cn.pdf`
- `reports/sensor_importance_detailed_report_cn.md`
- `code/analysis/analyze_sensor_importance.py`
- `code/analysis/generate_sensor_importance_pdf.py`
- `tables/sensor_importance_summary.csv`
- `processed/sample_features_baseline_corrected.csv`
- `processed/pca_3d_feature_space_scores.csv`
"""


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(TABLES / "sensor_importance_summary.csv")
    pca = pd.read_csv(PROCESSED / "pca_3d_feature_space_scores.csv")
    triplet = pd.read_csv(TABLES / "sensor_triplet_summary.csv")
    response = pd.read_csv(TABLES / "sensor_response_summary.csv")
    overview = (TABLES / "analysis_overview.txt").read_text(encoding="utf-8")

    bar_png = FIGURES / "sensor_ablation_importance.png"
    pca_png = FIGURES / "pca_feature_space_pc1_pc2.png"
    draw_bar_chart(summary, bar_png)
    draw_pca(pca, pca_png)

    report = PdfReport()
    report.title(
        "Air / Alcohol / Acetone 电子鼻传感器重要性分析报告",
        "基于 baseline-corrected 特征、传感器消融实验、随机盲测、日期留出与新瓶留出的训练前预处理建议",
    )
    report.callout(
        "一句话结论",
        "S2 与 S3 是区分 air / alcohol / acetone 时最有稳定边际贡献的传感器；S1 与 S5 有响应但在全模型中较冗余；S4 贡献较低；S6 可以视为不重要，默认不建议参与训练或 raw 3D feature-space 作图。",
    )
    report.h1("1. 数据与任务")
    report.p("本次分析使用 enose_data 中三个类别的全部文件：air 52 条、alcohol 73 条、acetone 73 条，共 198 条记录。CSV 中的 time_s 不是稳定秒轴，因此使用 arduino_time 重建相对时间。")
    report.p("采集协议按用户说明处理：180 s 记录为 60 s clean air + 60 s gas exposure + 60 s removal；125 s 记录为 60 s clean air + 5 s gas exposure + 60 s removal；130 s 记录为 60 s clean air + 10 s gas exposure + 60 s removal。")
    report.h2("Baseline correction")
    report.p("每条记录、每个传感器单独估计 baseline：S0 为 clean-air 段的中位数。校正公式为 (S(t)-S0)/S0。这样做的意义是把每次实验的起始漂移、环境差异和传感器绝对量程差异压低，让模型关注暴露和恢复过程中的相对变化。")

    report.h1("2. 相关论文如何处理多传感器电子鼻")
    lit_rows = [
        ["Zhang et al., 2014", "MOS 阵列，多气体识别", "研究所有传感器组合，用 KPCA + FLDA 评估组合分类准确率；把组合性能作为传感器贡献依据。", "选择小尺寸阵列的原因是成本、便携性和冗余。"],
        ["Borowik et al., 2020", "Reduced sensor array", "从 adsorption/desorption transient response 提取特征，用 leave-one-group-out 和 group shuffle validation 验证。", "强调不是所有传感器都必须保留；特征选择可降低阵列规模。"],
        ["Peng et al., 2023", "EWM-TOPSIS 阵列优化", "同时评估 sensitivity、selectivity、correlation、repeatability，再用 PCA 分离度和 SVM 准确率验证。", "单一指标不够，弱响应、强冗余或重复性差的传感器可以舍弃。"],
        ["MOX e-nose review, 2026", "综述", "总结 PCA 是电子鼻中常用的可视化、正交性评估和分类可行性检查工具。", "3D feature space 通常画 PC1/PC2/PC3，而不是任意选三个 raw sensor。"],
    ]
    report.table(["论文/来源", "对象", "他们怎么操作", "对本实验的启发"], lit_rows, [0.18, 0.17, 0.31, 0.34])
    report.p("文献依据的共同点是：先做预处理和特征提取，再用 PCA/KPCA/LDA 或分类器验证；是否舍弃传感器要看消融、组合性能、冗余和稳定性，而不是只看传感器响应幅度。")

    report.h1("3. 本次实验怎么做")
    report.h2("特征提取")
    report.p("对每个传感器提取 exposure 和 recovery 两阶段特征，包括 exposure mean、absolute max、span、tail mean、absolute AUC proxy、recovery mean、recovery tail、recovery slope 和整体标准差。训练时不直接使用 baseline 原值，以避免模型学到采集日期或初始漂移。")
    report.h2("三个验证方式")
    report.bullet("随机盲测：重复 100 次分层随机 70/30 train/test 切分。每次测试集文件在训练时不可见，所以可估计一般情况下的平均性能。")
    report.bullet("日期留出：每次把某一天采集的所有样本作为测试集，其余日期训练。这比随机盲测更严格，因为测试集包含同一天的环境、瓶子状态、操作批次和潜在漂移。")
    report.bullet("新瓶留出：README 说明 2026-06-04 和 2026-06-05 使用新液体瓶，因此训练集只用此前日期，测试集用这两天。这用于观察换瓶后是否还能泛化。")
    report.h2("什么是边际贡献")
    report.p("边际贡献指：全传感器模型的性能减去“去掉某个传感器后”的性能。如果差值为正，说明去掉它会变差，它有贡献；如果差值接近 0，说明它可有可无；如果差值为负，说明去掉它反而更好，可能存在噪声或冗余。")

    report.h1("4. 分类器与全阵列性能")
    perf_rows = [
        ["Stratified random holdout", "0.962", "0.957", "0.959"],
        ["Leave collection date out", "0.931", "0.922", "0.948"],
        ["New bottle holdout", "1.000", "1.000", "1.000"],
    ]
    report.table(["验证方式", "LDA balanced accuracy", "Macro F1", "Accuracy"], perf_rows, [0.35, 0.22, 0.20, 0.23])
    report.p("LDA 是主报告指标，因为它在三种实现的分类器中整体最好，也符合电子鼻文献中常见的 PCA/LDA 分析传统。Gaussian Naive Bayes 和 nearest centroid 作为稳定性参考。")

    report.h1("5. 传感器消融结果")
    rows = []
    interp = {
        "s2": "最重要；尤其帮助 alcohol / acetone 分离",
        "s3": "重要；帮助气体样本不被判成 air",
        "s1": "有用但冗余",
        "s5": "有用但冗余，单传感器表现较强",
        "s4": "边际价值较低",
        "s6": "不重要；可默认排除",
    }
    for row in summary.itertuples(index=False):
        s = str(row.removed_sensor)
        rows.append([
            s.upper(),
            fmt(float(row.random_holdout_lda_delta_bacc)),
            fmt(float(row.leave_date_lda_delta_bacc)),
            fmt(float(row.new_bottle_lda_delta_bacc)),
            fmt(float(row.single_sensor_lda_bacc)),
            interp.get(s, ""),
        ])
    report.table(["传感器", "随机盲测 delta", "日期留出 delta", "新瓶留出 delta", "单传感器 BAcc", "解释"], rows, [0.10, 0.17, 0.17, 0.16, 0.16, 0.24])
    report.image(bar_png, caption="图 1：LDA leave-one-sensor-out 消融。正值表示移除该传感器会降低 balanced accuracy，因此传感器有正向边际贡献。")
    report.p("关键混淆差异：移除 S2 后，随机盲测中的 acetone -> alcohol 错误从 4 个聚合计数增加到 171，alcohol -> acetone 从 4 增至 63，因此 S2 是 alcohol 与 acetone 区分的关键。移除 S3 后，acetone -> air 错误从 68 增至 168，alcohol -> air 从 152 增至 198，因此 S3 对防止气体样本塌缩为 air 有帮助。")

    report.h1("6. 响应幅度为什么不能单独决定重要性")
    resp_pivot = response.pivot(index="sensor", columns="label", values="median_exp_absmax").reset_index()
    rows = []
    for row in resp_pivot.itertuples(index=False):
        rows.append([str(row.sensor).upper(), fmt(float(row.air)), fmt(float(row.alcohol)), fmt(float(row.acetone))])
    report.table(["传感器", "Air", "Alcohol", "Acetone"], rows, [0.18, 0.27, 0.27, 0.28])
    report.p("S1 和 S5 的响应幅度很大，单独分类也不错，但在全模型里去掉它们的性能下降很小，说明它们提供的信息与其他传感器部分重复。S2 的响应幅度没有 S1/S5 大，却有最高边际贡献；这说明电子鼻传感器选择要看类别可分性与消融验证，而不是只看峰值大小。S6 在 air/alcohol/acetone 上没有形成稳定可分的模式，单传感器和消融结果都弱。")

    report.h1("7. 3D feature space 怎么画")
    report.p("根据文献，最稳妥的 thesis 图是用保留传感器的 baseline-corrected 特征做 PCA，然后画 PC1/PC2/PC3。这样三个坐标轴是最大方差信息方向，而不是主观挑选三个 raw sensor。")
    report.image(pca_png, caption="图 2：全部 baseline-corrected 特征的 PCA 投影示意。正式 3D 图可使用 processed/pca_3d_feature_space_scores.csv 中的 PC1/PC2/PC3。")
    trip = triplet[(triplet["mode"] == "stratified_random_holdout") & (triplet["classifier"] == "lda")].sort_values("mean_balanced_accuracy", ascending=False).head(6)
    trip_d = triplet[(triplet["mode"] == "leave_collection_date_out") & (triplet["classifier"] == "lda")].sort_values("mean_balanced_accuracy", ascending=False).head(6)
    rows = [[r.sensors.upper(), fmt(float(r.mean_balanced_accuracy)), "随机盲测"] for r in trip.itertuples(index=False)]
    rows += [[r.sensors.upper(), fmt(float(r.mean_balanced_accuracy)), "日期留出"] for r in trip_d.itertuples(index=False)]
    report.table(["三传感器组合", "Mean BAcc", "验证方式"], rows, [0.36, 0.24, 0.40])
    report.p("如果必须画 raw sensor 3D 坐标，则 S1+S2+S3 在随机盲测中最好，S2+S3+S5 在日期留出中最好。两者都包含 S2/S3，因此建议 raw 3D 轴以 S2、S3 为核心，第三轴在 S1 和 S5 间根据展示目标选择。")

    report.h1("8. 最终建议")
    original_summary_path = TABLES / "original_points_sensor_importance_summary.csv"
    if original_summary_path.exists():
        original_summary = pd.read_csv(original_summary_path)
        report.h1("8. 原始数据点对照实验")
        report.p("为了按“除 baseline correction 外不做任何处理”的口径验证，额外做了一版 original-points 对照：每条 CSV 只做 baseline correction，然后保留每个传感器在文件中的原始采样点序列。不提取 exposure/recovery 特征，不计算峰值、均值、AUC、斜率，也不做时间轴重采样。")
        report.p("由于不同记录长度不完全一样，短记录后面的点用 NaN 占位；分类器计算距离或统计量时只比较实际存在的原始点列。这个 NaN padding 只是矩阵存储方式，不改变原始采样点本身。")
        report.p("这版实验使用 original_points_centroid 和 original_points_gnb 两个简单评价器。它仍然不是最终模型训练，而是用原始点序列直接做传感器消融验证。")
        rows = []
        for clf in ["original_points_centroid", "original_points_gnb"]:
            part = original_summary[
                (original_summary["mode"] == "original_points_stratified_random_holdout")
                & (original_summary["classifier"] == clf)
            ].sort_values("mean_delta_balanced_accuracy", ascending=False)
            for row in part.itertuples(index=False):
                rows.append([
                    clf,
                    str(row.removed_sensor).upper(),
                    fmt(float(row.mean_delta_balanced_accuracy)),
                    fmt(float(row.single_sensor_mean_balanced_accuracy)),
                ])
        report.table(["Original-points evaluator", "去掉传感器", "随机盲测 delta BAcc", "单传感器 BAcc"], rows, [0.32, 0.18, 0.25, 0.25])
        report.p("Original-points 对照的结论与特征版不完全相同，但主趋势一致：centroid 评价器中 S3 和 S2 的边际贡献最高；GNB 评价器中 S2 最高，S6 是明显负贡献。由于完全不做响应特征工程，整体分类性能低于特征版，这是正常的。它适合作为最干净的方法学对照。")

    report.h1("9. 最终建议")
    report.callout(
        "训练前预处理建议",
        "默认训练传感器：S1, S2, S3, S5。精简训练传感器：S1, S2, S3。默认排除：S6。S4 可以作为可选传感器，但不应作为核心传感器。3D feature space 优先使用 PCA 3D scores；raw 3D 图优先考虑 S2+S3+S5 或 S1+S2+S3。",
    )
    report.h2("输出文件")
    report.bullet("baseline-corrected 特征：processed/sample_features_baseline_corrected.csv")
    report.bullet("PCA 3D 坐标：processed/pca_3d_feature_space_scores.csv")
    report.bullet("主排名表：tables/sensor_importance_summary.csv")
    report.bullet("完整消融结果：tables/sensor_ablation_all_results.csv")
    report.bullet("三传感器组合表：tables/sensor_triplet_summary.csv")
    report.bullet("正式分析代码：code/analysis/analyze_sensor_importance.py 与 code/analysis/generate_sensor_importance_pdf.py")

    report.h1("10. 参考文献与网页来源")
    refs = [
        "Zhang et al. A novel sensor selection using pattern recognition in electronic nose. Measurement, 2014. https://www.sciencedirect.com/science/article/abs/pii/S0263224114001559",
        "Borowik et al. Odor Detection Using an E-Nose With a Reduced Sensor Array. Sensors, 2020. https://pubmed.ncbi.nlm.nih.gov/32585850/",
        "Peng et al. A Comprehensive Evaluation Model for Optimizing the Sensor Array of Electronic Nose. Applied Sciences, 2023. https://www.mdpi.com/2076-3417/13/4/2338",
        "In-Vehicle Gas Sensing and Monitoring Using Electronic Noses Based on Metal Oxide Semiconductor MEMS Sensor Arrays: A Critical Review. Chemosensors, 2026. https://www.mdpi.com/2227-9040/14/1/16",
        "Hybrid Feature Selection for Optimizing Sensor Array Reduction in Coffee Aroma Electronic Nose Systems. Journal of Robotics and Control, 2025. https://journal.umy.ac.id/index.php/jrc/article/view/28470",
    ]
    for ref in refs:
        report.bullet(ref)

    report.save(PDF_PATH, PREVIEW_DIR)
    MD_PATH.write_text(build_markdown(summary), encoding="utf-8")
    print(PDF_PATH)
    print(MD_PATH)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
