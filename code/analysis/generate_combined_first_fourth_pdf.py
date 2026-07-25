from __future__ import annotations

import pandas as pd

from generate_sensor_importance_pdf import REPORTS, TABLES, PdfReport, fmt


PDF_PATH = REPORTS / "sensor_importance_combined_feature_and_original_points_cn.pdf"
MD_PATH = REPORTS / "sensor_importance_combined_feature_and_original_points_cn.md"
PREVIEW_DIR = REPORTS / "sensor_importance_combined_pages"


def table_rows_feature_delta(summary: pd.DataFrame) -> list[list[str]]:
    rows = []
    for row in summary.sort_values("random_holdout_lda_delta_bacc", ascending=False).itertuples(index=False):
        rows.append([
            str(row.removed_sensor).upper(),
            fmt(float(row.random_holdout_lda_delta_bacc)),
            fmt(float(row.leave_date_lda_delta_bacc)),
            fmt(float(row.single_sensor_lda_bacc)),
        ])
    return rows


def table_rows_original_delta(summary: pd.DataFrame) -> list[list[str]]:
    rows = []
    for clf in ["original_points_centroid", "original_points_gnb"]:
        part = summary[
            (summary["mode"] == "original_points_stratified_random_holdout")
            & (summary["classifier"] == clf)
        ].sort_values("mean_delta_balanced_accuracy", ascending=False)
        for row in part.itertuples(index=False):
            rows.append([
                clf.replace("original_points_", ""),
                str(row.removed_sensor).upper(),
                fmt(float(row.mean_delta_balanced_accuracy)),
                fmt(float(row.positive_delta_rate)),
                fmt(float(row.single_sensor_mean_balanced_accuracy)),
            ])
    return rows


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    feature_summary = pd.read_csv(TABLES / "sensor_importance_summary.csv")
    original_summary = pd.read_csv(TABLES / "original_points_sensor_importance_summary.csv")
    feature_ablation = pd.read_csv(TABLES / "sensor_ablation_all_results.csv")
    original_ablation = pd.read_csv(TABLES / "original_points_sensor_ablation_all_results.csv")

    report = PdfReport()
    report.title(
        "Air / Alcohol / Acetone 传感器重要性合并报告",
        "合并第 1 版 feature-extraction 消融与第 4 版 original-points 消融；中间 raw-resampled / mixed 报告不作为最终依据",
    )
    report.callout(
        "最终保留口径",
        "本报告只保留两组数据：第一次的 baseline-corrected feature-extraction 消融，以及第四次的 baseline-corrected original-points 消融。第二次 raw-resampled 和第三次混合报告已作为中间版本删除/归档，不作为最终判断依据。",
    )

    report.h1("1. 两组实验为什么结果会不一样")
    report.p("第一次实验使用 baseline correction 后的响应特征，例如 exposure/recovery 阶段的均值、幅度、尾部响应和恢复趋势。它更接近电子鼻文献中常见的特征工程 + 传感器选择流程，因此分类性能更高。")
    report.p("第四次实验严格按原始点口径：除 baseline correction 外，不做特征提取、不重采样、不切 exposure/recovery 特征，直接使用 CSV 中各传感器的原始采样点。由于没有人工响应特征，它的整体性能低一些，但方法口径更干净。")
    report.p("所以两组结果不完全相同是正常的。更稳健的结论应看两组是否共同支持某些传感器。两组都支持 S2/S3 是核心信号来源，S4/S6 不适合作为核心传感器，尤其 S6 需要谨慎使用或排除。")

    report.h1("2. LDA + delta 消融这个 idea 来自哪里")
    report.p("这里的 idea 分成两层：第一层是用 LDA 做 sensor selection / sensor contribution，第二层是通过删除某个传感器后性能变化，即 delta，判断该传感器的边际贡献。")
    report.p("Guo, Zhang and Zhang 在 Sensors and Actuators B: Chemical 2011 的论文 An LDA based sensor selection approach used in breath analysis system 中提出 LDASS，即用 LDA 寻找让两类样本分开的方向，并把 LDA 权重解释为各传感器的贡献权重，用来判断哪个传感器对分类贡献更大。")
    report.p("Zhang et al. 2014 的 A novel sensor selection using pattern recognition in electronic nose 使用 KPCA + FLDA，对电子鼻传感器组合进行模式识别评估，用不同传感器组合的分类表现判断阵列中传感器的有效性。")
    report.p("Borowik et al. 2020 的 Odor Detection Using an E-Nose With a Reduced Sensor Array 也把 reduced sensor array / feature selection 放在分类性能验证框架中讨论，包括 forward/backward selection 和不同子集性能比较。因此，用“全传感器表现 - 去掉某传感器后的表现”作为 delta，是 wrapper-style selection / backward ablation 的直接实现。")
    report.callout(
        "本报告中的 delta 定义",
        "delta BAcc = 全传感器 Mean BAcc - 去掉某传感器后的 Mean BAcc。delta > 0 表示移除该传感器会变差，因此它有正边际贡献；delta <= 0 表示它边际价值低、冗余，甚至可能引入噪声。",
    )

    report.h1("3. 第一次：Feature-extraction 消融结果")
    perf_rows = []
    for mode in ["stratified_random_holdout", "leave_collection_date_out", "new_bottle_holdout"]:
        part = feature_ablation[(feature_ablation["mode"] == mode) & (feature_ablation["classifier"] == "lda") & (feature_ablation["case"] == "all")]
        if len(part):
            perf_rows.append([mode, fmt(float(part["balanced_accuracy"].mean())), fmt(float(part["macro_f1"].mean())), fmt(float(part["accuracy"].mean()))])
    report.table(["验证方式", "LDA Mean BAcc", "Mean Macro F1", "Mean Accuracy"], perf_rows, [0.40, 0.22, 0.19, 0.19])
    report.table(["去掉传感器", "随机盲测 delta BAcc", "日期留出 delta BAcc", "单传感器 BAcc"], table_rows_feature_delta(feature_summary), [0.20, 0.28, 0.28, 0.24])
    report.p("第一次实验的主结论：S2 的边际贡献最高，S3 次之；S1/S5 有响应但在全模型中较冗余；S4/S6 边际贡献低，S6 在随机盲测和日期留出中呈现负贡献。")

    report.h1("4. 第四次：Original-points 消融结果")
    perf_rows = []
    for mode in [
        "original_points_stratified_random_holdout",
        "original_points_leave_collection_date_out",
        "original_points_new_bottle_holdout",
    ]:
        for clf in ["original_points_centroid", "original_points_gnb"]:
            part = original_ablation[(original_ablation["mode"] == mode) & (original_ablation["classifier"] == clf) & (original_ablation["case"] == "all")]
            if len(part):
                perf_rows.append([
                    mode.replace("original_points_", ""),
                    clf.replace("original_points_", ""),
                    fmt(float(part["balanced_accuracy"].mean())),
                    fmt(float(part["macro_f1"].mean())),
                    fmt(float(part["accuracy"].mean())),
                ])
    report.table(["验证方式", "评价器", "Mean BAcc", "Mean Macro F1", "Mean Accuracy"], perf_rows, [0.27, 0.22, 0.17, 0.17, 0.17])
    report.table(["评价器", "去掉传感器", "随机盲测 delta BAcc", "正贡献比例", "单传感器 BAcc"], table_rows_original_delta(original_summary), [0.24, 0.16, 0.25, 0.17, 0.18])
    report.p("第四次实验的主结论：centroid 评价器中 S3 的边际贡献最高，其次是 S2；GNB 评价器中 S2 最高，S6 是明显负贡献。该结果在不做特征提取的口径下仍支持 S2/S3 是关键传感器。")

    report.h1("5. 合并结论")
    report.callout(
        "传感器排序建议",
        "核心：S2、S3。辅助：S1、S5。低优先级：S4。默认谨慎或排除：S6。若当前阶段坚持不做特征提取，请优先引用 original_points_* 结果；若进入后续机器学习建模，可把 feature-extraction 结果作为特征工程后传感器筛选依据。",
    )
    report.p("为什么 S2/S3 稳健：第一次 feature-extraction 消融中，去掉 S2/S3 后 LDA balanced accuracy 下降最大；第四次 original-points 消融中，centroid 和 GNB 仍分别把 S3/S2 排在最高贡献位置。")
    report.p("为什么 S6 不重要：第一次中去掉 S6 不降低甚至略提升性能；第四次 GNB 中去掉 S6 的 delta 为 -0.167，说明 S6 在原始点空间中可能引入噪声或不稳定差异。")

    report.h1("6. 最终保留文件")
    report.bullet("第一次数据：tables/sensor_importance_summary.csv, tables/sensor_ablation_all_results.csv, processed/sample_features_baseline_corrected.csv")
    report.bullet("第四次数据：tables/original_points_sensor_importance_summary.csv, tables/original_points_sensor_ablation_all_results.csv, processed/original_points_baseline_corrected_matrix.csv")
    report.bullet("合并 PDF：reports/sensor_importance_combined_feature_and_original_points_cn.pdf")
    report.bullet("合并 PDF 生成脚本：code/analysis/generate_combined_first_fourth_pdf.py")

    report.h1("7. 参考来源")
    refs = [
        "Guo, D., Zhang, D., & Zhang, L. (2011). An LDA based sensor selection approach used in breath analysis system. Sensors and Actuators B: Chemical, 157(1), 265-274. https://doi.org/10.1016/j.snb.2011.03.061",
        "Zhang et al. (2014). A novel sensor selection using pattern recognition in electronic nose. Measurement. https://www.sciencedirect.com/science/article/abs/pii/S0263224114001559",
        "Borowik et al. (2020). Odor Detection Using an E-Nose With a Reduced Sensor Array. Sensors. https://www.mdpi.com/1424-8220/20/12/3542",
    ]
    for ref in refs:
        report.bullet(ref)

    report.save(PDF_PATH, PREVIEW_DIR)
    MD_PATH.write_text(
        "# Combined sensor-importance report\n\n"
        "Keeps only the first feature-extraction experiment and the fourth original-points experiment.\n",
        encoding="utf-8",
    )
    print(PDF_PATH)
    print(MD_PATH)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
