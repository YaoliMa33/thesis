from __future__ import annotations

from generate_sensor_importance_pdf import REPORTS, PdfReport


PDF_PATH = REPORTS / "dcae_thesis_research_plan_cn.pdf"
MD_PATH = REPORTS / "dcae_thesis_research_plan_cn.md"
PREVIEW_DIR = REPORTS / "dcae_thesis_research_plan_pages"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    report = PdfReport()
    report.title(
        "DCAE 气体传感器漂移补偿论文研究路线报告",
        "面向后续新聊天框继续工作的完整参考：数据可视化、原始点 vs 特征、传统机器学习、5s 动态窗口、Domain Adaptation 与 DCAE",
    )

    report.callout(
        "论文核心主线",
        "本论文不是单纯追求三分类准确率，而是要证明：气体传感器数据存在 drift/domain shift；drift 会降低跨日期、跨批次、跨新瓶条件下的识别能力；domain adaptation，尤其 DCAE，可学习更稳定的表示，从而提升 air/alcohol/acetone 的鲁棒判别。",
    )

    report.h1("1. 论文目标与核心问题")
    report.p("论文题目为 Compensating Gas Sensor Drift via Domain Adaptation for Robust Gas Discrimination。核心任务是使用现有 air、alcohol、acetone 电子鼻数据，围绕 sensor drift compensation 建立完整实验链。")
    report.p("需要回答的问题包括：同一气体在不同日期或不同采集条件下是否出现分布偏移？这种偏移是否会伤害传统分类模型？原始数据点和人工特征哪一种更适合后续分析？传统机器学习模型在三分类任务上表现如何？迁移学习和 DCAE 是否能在 source-to-target 场景中提升 target domain 的分类鲁棒性？")

    report.h1("2. 数据与 Domain 定义")
    report.p("数据包括 air、alcohol、acetone 三类 CSV。所有后续实验均应至少进行 baseline correction。Domain 可以按日期、早期/后期、新瓶/旧瓶、距离、风速或 duration 定义。最适合当前论文主线的是按日期和新瓶条件定义 domain。")
    report.table(
        ["Domain 划分", "建议定义", "用途"],
        [
            ["日期 domain", "每个采集日期作为一个 domain", "分析时间漂移和日期留出泛化"],
            ["早期 vs 后期", "早期日期为 source，后期日期为 target", "模拟 sensor aging 或 temporal drift"],
            ["新瓶 domain", "2026-06-04 和 2026-06-05 作为 target", "测试换瓶/批次变化下的 domain shift"],
            ["条件 domain", "距离、风速、duration 作为辅助因素", "分析环境和采集协议对响应分布的影响"],
        ],
        [0.18, 0.39, 0.43],
    )
    report.callout(
        "第一步最重要的决定",
        "先固定 source/target split，再做所有模型实验。否则每次实验用不同划分，无法清楚证明 domain adaptation 是否真的补偿了 drift。",
    )

    report.h1("3. 数据可视化：先证明 Drift 存在")
    report.p("论文第一部分应先做可视化，而不是直接训练模型。目标是证明同一类气体在不同日期、不同批次、新瓶条件下的分布发生变化。")
    report.bullet("Baseline-corrected 原始响应曲线：按气体类别和日期画 S1-S6 的动态响应。")
    report.bullet("3D feature space：使用 S1/S2/S3 和 S2/S3/S5，直接画 baseline-corrected 原始采样点，可 360° 旋转。")
    report.bullet("PCA/UMAP/t-SNE：用样本级特征或窗口级特征观察同一气体是否按日期或 domain 分离。")
    report.bullet("传感器随日期的响应统计：观察相同气体的响应幅度、baseline、恢复趋势是否随时间漂移。")
    report.p("这一章的结论应该是：air/alcohol/acetone 在同分布随机划分下可能较容易区分，但跨日期或新瓶后分布偏移明显，因此需要 drift compensation。")

    report.h1("4. 原始数据点 vs 特征提取")
    report.p("你需要比较两种数据表达，以决定后续传统机器学习和 domain adaptation 使用哪一种作为主要输入。")
    report.table(
        ["表达方式", "定义", "优点", "缺点"],
        [
            ["Original points", "只做 baseline correction，保留 CSV 原始采样点", "最少假设，最接近真实传感器流", "维度高、长度不一致、分类性能可能较低"],
            ["5s raw windows", "把原始时序切成 5s 窗口，保留窗口内原始点", "模拟动态电子鼻，适合在线判别", "需要严格防止同一 CSV 窗口泄漏"],
            ["Handcrafted features", "提取 mean/max/std/slope/AUC/tail/recovery 等特征", "维度低、可解释、传统 ML 表现通常更稳", "引入人工假设，可能丢失动态曲线细节"],
            ["5s window features", "每个 5s 窗口提取局部统计/动态特征", "兼顾动态检测和可解释性", "需要比较窗口长度和特征组合"],
        ],
        [0.18, 0.30, 0.27, 0.25],
    )
    report.p("判断标准不能只看随机划分准确率。更重要的是日期留出、新瓶留出、source-to-target 表现。如果某种表达在 target domain 上更稳，就更适合 drift compensation 研究。")

    report.h1("5. 5s 窗口：模拟动态电子鼻")
    report.p("5s 窗口是一个非常适合本论文的设计，因为真实电子鼻通常不会等完整 180s 采集结束后才分类，而是持续接收传感器流并做在线判断。5s window 可以把离线实验变成动态识别任务。")
    report.bullet("主实验建议使用 5s non-overlapping windows，即 0-5s、5-10s、10-15s。")
    report.bullet("敏感性分析可以比较 2s、10s、20s 窗口，以及 2.5s step 的重叠窗口。")
    report.bullet("窗口样本必须保留 sample_id、window_start_s、window_end_s、phase、domain_date、domain_group。")
    report.callout(
        "严禁数据泄漏",
        "不能把同一条 CSV 的不同窗口同时放进训练集和测试集。正确做法是先按 sample_id/date/domain 划分 train/test，再在各自集合内部切窗口。",
    )
    report.h2("窗口标签建议")
    report.p("主实验建议先用最干净的标签策略：gas exposure 窗口标为对应气体，clean-air 窗口标为 air。Recovery 阶段存在残留响应，标签有争议，可以作为扩展实验或单独 phase-aware 分析。")
    report.table(
        ["方案", "标签方式", "推荐程度"],
        [
            ["A 主实验", "只用 clean-air 和 gas exposure 窗口", "最高，标签最可靠"],
            ["B 动态扩展", "recovery 也标为原气体残留", "可做扩展，模拟真实残留检测"],
            ["C phase-aware", "同时预测 gas_label 与 phase", "适合深入分析，但复杂度更高"],
        ],
        [0.18, 0.48, 0.34],
    )

    report.h1("6. 传统机器学习三分类 Baseline")
    report.p("在 domain adaptation 之前，必须建立 non-adaptation baseline。否则无法证明 DCAE 的提升来自 drift compensation。")
    report.table(
        ["模型", "作用", "备注"],
        [
            ["LDA", "线性判别与可解释 baseline", "也可用于传感器消融"],
            ["Logistic Regression", "线性分类 baseline", "适合标准化特征"],
            ["SVM", "强传统模型", "可比较 linear/RBF kernel"],
            ["Random Forest", "非线性、抗噪 baseline", "可给 feature importance"],
            ["kNN", "局部分布 baseline", "对 domain shift 敏感，可作为漂移对照"],
            ["Gaussian NB", "简单概率 baseline", "适合快速对照"],
        ],
        [0.22, 0.42, 0.36],
    )
    report.p("评估方式包括 random split、leave-date-out、old-bottle to new-bottle、early-to-late source-to-target。关键是展示：random split 表现不错，但 source-to-target 明显下降。这个下降就是 domain adaptation 的动机。")

    report.h1("7. Domain Adaptation 与 DCAE")
    report.p("Domain adaptation 的目标是学习一个表示空间，使 source domain 和 target domain 的分布更接近，同时保留气体类别可分性。对于本论文，可以先实现简单方法，再实现 DCAE。")
    report.table(
        ["方法层级", "方法", "用途"],
        [
            ["无迁移", "Source-only classifier", "必须有，用来证明 drift 伤害"],
            ["简单校正", "target/domain-wise standardization", "低成本 drift correction baseline"],
            ["传统 DA", "CORAL, TCA, BDA, JDA", "对齐均值/协方差/子空间/类别分布"],
            ["深度 DA", "DANN, MMD-AE, CORAL-AE", "学习 domain-invariant latent representation"],
            ["重点方法", "DCAE", "用 autoencoder 学 drift-corrected representation"],
        ],
        [0.18, 0.34, 0.48],
    )
    report.h2("DCAE 设计")
    report.p("DCAE 可以设计为 encoder-decoder-classifier 结构。Encoder 把输入映射到 latent representation z；decoder 重构输入；classifier 在 source label 上预测 gas class；domain alignment loss 让 source 和 target 的 z 更接近。")
    report.p("推荐 loss：reconstruction loss + source classification loss + domain alignment loss。Domain alignment 可以从 MMD 或 CORAL 开始，之后再尝试 adversarial domain discriminator。")

    report.h1("8. 推荐实验矩阵")
    report.table(
        ["阶段", "输入", "方法", "目的"],
        [
            ["Visualization", "baseline-corrected raw/5s windows", "curves, 3D, PCA", "证明 drift/domain shift"],
            ["Representation", "original points vs features", "LDA/SVM/RF", "选择后续输入表达"],
            ["Traditional ML", "选定表达", "LDA/SVM/RF/kNN", "建立 non-adaptation baseline"],
            ["Domain Adaptation", "选定表达或 latent input", "CORAL/TCA/BDA/DCAE", "证明 drift compensation"],
            ["Ablation", "DCAE variants", "remove losses/sensors", "证明每个模块有用"],
        ],
        [0.18, 0.25, 0.25, 0.32],
    )

    report.h1("9. 下一步执行清单")
    report.bullet("固定 domain split：建议 old bottle/early dates 作为 source，2026-06-04/2026-06-05 新瓶作为 target，同时保留 leave-date-out。")
    report.bullet("生成 5s window 数据集：original-window points 与 window features 两套都保留。")
    report.bullet("做可视化：曲线、3D feature space、PCA by date/domain。")
    report.bullet("跑 traditional ML：比较 original points、5s raw windows、window features、full-record features。")
    report.bullet("确认 source-to-target 性能下降后，再上 CORAL/TCA/BDA/DCAE。")
    report.bullet("DCAE 做 ablation：无 alignment、MMD alignment、CORAL alignment、不同 latent dimension、不同窗口长度。")

    report.h1("10. 参考文献与链接")
    refs = [
        "Ke Yan and David Zhang. Correcting Instrumental Variation and Time-Varying Drift: A Transfer Learning Approach With Autoencoders / DCAE preprint. https://yanke23.com/papers/preprint_DCAE.pdf",
        "Ke Yan, Lu Kou, David Zhang. Learning Domain-Invariant Subspace using Domain Features and Independence Maximization. https://arxiv.org/abs/1603.04535",
        "Balanced Distribution Adaptation for Metal Oxide Semiconductor Gas Sensor Array Drift Compensation. Sensors, 2021. https://www.mdpi.com/1424-8220/21/10/3403",
        "Wasserstein Distance Learned Feature Representations for Drift Compensation of E-Nose. Sensors, 2019. https://www.mdpi.com/1424-8220/19/17/3703",
        "A comprehensive gas recognition algorithm with label-free drift compensation based on domain adversarial network. Sensors and Actuators B, 2023. https://www.sciencedirect.com/science/article/pii/S0925400523004240",
        "TDACNN: Target-domain-free Domain Adaptation Convolutional Neural Network for Drift Compensation in Gas Sensors. https://arxiv.org/abs/2110.07509",
    ]
    for ref in refs:
        report.bullet(ref)

    report.save(PDF_PATH, PREVIEW_DIR)
    MD_PATH.write_text(
        "# DCAE thesis research plan\n\n"
        "Chinese planning report for continuing the thesis work in a new chat.\n",
        encoding="utf-8",
    )
    print(PDF_PATH)
    print(MD_PATH)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
