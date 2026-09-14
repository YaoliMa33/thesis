# EnoseData: short-term electronic-nose dataset and frozen splits

本目录公开实验使用的原始传感器序列、逐样本采集信息、54 维特征，以及 S1-S7 和 A/B/C/D 的**冻结样本成员表**。无需运行特征提取或划分脚本即可按 `sample_id` 加载数据、训练自己的模型。这里不包含训练代码、模型权重或结果筛选。

## 1. 文件与数据边界

| 路径 | 内容 |
| --- | --- |
| `raw/air/`, `raw/alcohol/`, `raw/acetone/` | 303 个原始 CSV，每个 CSV 为一次采集的时间序列。 |
| `metadata/sample_manifest.csv` | 每个样本的 `sample_id`、类别、采集日期、原始 batch、记录时长、原始文件相对路径等。 |
| `metadata/acquisition_inventory.csv` | 清单与实际阶段审计的逐样本合表；可直接筛选类别、时长和 `profile`。 |
| `metadata/phase_audit.csv` | 每个样本实际采用的空气、响应、恢复窗口及各窗口点数。 |
| `metadata/provenance.json` | 本次公开包的输入表哈希与样本数。 |
| `processed/features_54d.csv` | 与实验相同的 303 × 54 特征表，另有样本元数据列。 |
| `splits/s1_s7_assignments.csv` | S1-S7 的逐样本角色标志，7 × 303 行。 |
| `splits/abcd_assignments.csv` | A1-A5、B1-B4、C1-C5、D 的逐样本角色标志，15 × 303 行。 |
| `splits/definitions.json` | 各阶段可见数据、测试对象、数量和原始 batch 到论文 Batch 的映射。 |

主键为 `sample_id`。`sample_manifest.csv` 的 `target_file` 是相对于 `raw/` 的路径，文件中可能使用 Windows 反斜杠；跨平台读取时应将其视为路径分隔符。两个成员表中，每个 split/stage 都列出全部 303 个样本，角色列使用 `1` 表示属于该角色、`0` 表示不属于。一个样本可同时属于 `target_domain` 和 `target_test`，这是声明的**无标签转导式适应**，并非允许训练时使用其测试标签。

## 2. 样本、气体和时间批次

共有 303 个样本，六个传感器通道 `s1`-`s6`，三类标签：`air` 87、`alcohol` 108、`acetone` 108。源域为 Period I 的 30 个样本；Period II 的 Day 1-Day 24 共 273 个样本，按采集时间分为五个目标 batch。Day 25/26 不在此数据包的当前实验范围内。

| 论文名称 | 原始 `batch` 值 | 采集区间 | 总数 | air | alcohol | acetone |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Batch1 | `source` | Period I | 30 | 6 | 12 | 12 |
| Batch2 | `batch1` | Day 1-Day 2 | 28 | 8 | 10 | 10 |
| Batch3 | `batch2` | Day 3-Day 9 | 84 | 18 | 33 | 33 |
| Batch4 | `batch3` | Day 10-Day 17 | 81 | 31 | 25 | 25 |
| Batch5 | `batch4` | Day 18-Day 21 | 46 | 14 | 16 | 16 |
| Batch6 | `batch5` | Day 22-Day 24 | 34 | 10 | 12 | 12 |

原始清单中的 `source_kind` 保留了样本来源：6 个 `baseline_as_air` 和 9 个 `reference_as_air` 按既定实验标注计入 `air`。这些标签未在本数据包中重新解释或修改。`record_date` 和 `duration_s` 可用于检查采集条件，不能作为实验分类模型的输入特征。

## 3. 采集时长与阶段划分

阶段选择依据清单中的**标称采集时长及已记录的手动事件**，不依据 `air` / `alcohol` / `acetone` 类别。时间轴优先使用每个原始文件的 `arduino_time`：减去该文件首个读数，再除以 1000 得到秒；仅在没有该列时才使用 `time_s`。手动操作导致实际末时刻可能不等于标称时长，窗口按所列时间边界与实际存在的采样点截取。

| `profile` | 样本数 | 空气基线 | 气体响应 | 恢复 | 类别构成 air / alcohol / acetone |
| --- | ---: | --- | --- | --- | --- |
| `programme_60` | 9 | [0,30) s | [30,40) s | [40,实际末时刻] s | 9 / 0 / 0 |
| `programme_125` | 10 | [0,60) s | [60,65) s | [65,实际末时刻] s | 2 / 4 / 4 |
| `programme_130` | 22 | [0,60) s | [60,70) s | [70,实际末时刻] s | 4 / 9 / 9 |
| `programme_180_window` | 261 | [0,60) s | [60,120) s | [120,min(180,实际末时刻)] s | 71 / 95 / 95 |
| `manual_180_air_22` | 1 | [0,60) s | [60,70) s | [70,min(180,实际末时刻)] s | 1 / 0 / 0 |

`programme_180_window` 包括标称 180 秒及部分略长于 180 秒、约 302 秒的记录；**特征计算只使用前 180 秒**，并没有把更长记录的尾段偷偷并入恢复期。`air_22` 是已记录的人工操作例外，虽属于 180 秒采集程序，响应结束点采用 70 秒。`metadata/acquisition_inventory.csv` 可逐行查到每个气体和样本的标称时长、实际末时刻、所用窗口及 `target_file`。

## 4. 预处理与 54D 特征

对每个样本、每个传感器**单独**计算空气阶段读数中位数 `m`，然后把原始通道 `x(t)` 变为 `z(t) = (x(t) - m) / |m|`。所以空气阶段的中位数归零；这不是强制每条曲线在第一个时刻重合。原始 CSV 中完全空白的行会被忽略；部分缺失行、`m` 过于接近零、时间轴不递增、窗口不足三个观测点或读数非有限值都会报错，而不是静默补值。未做平滑、插值、异常样本删除或跨样本归一化。

六个通道分别提取以下九项，共 `6 × 9 = 54` 项；列名格式为 `s{1..6}_{suffix}`：

| 后缀 | 在基线校正后 `z(t)` 上的定义 |
| --- | --- |
| `exp_mean` | 响应窗口均值。 |
| `exp_absmax` | 响应窗口绝对值最大值。 |
| `exp_span` | 响应窗口最大值减最小值。 |
| `exp_tail` | 响应窗口最后 `ceil(25% × 点数)` 个读数的均值。 |
| `exp_auc_abs` | 响应窗口绝对值的**均值**；这是历史列名，并非数值积分。 |
| `rec_mean` | 恢复窗口均值。 |
| `rec_tail` | 恢复窗口最后 `ceil(25% × 点数)` 个读数的均值。 |
| `rec_slope` | 恢复阶段 `z` 对相对时间（秒）做一元最小二乘拟合的斜率。 |
| `overall_std` | 从 0 秒到恢复窗口终点的总体标准差（`ddof=0`）。 |

`processed/features_54d.csv` **仅完成上述逐样本基线校正与特征提取**，没有预先用所有样本拟合 StandardScaler。复现实验时，应只用该阶段的 `source_train` 估计每一特征的均值与标准差，并将同一个变换应用于允许的训练、验证和测试样本；不能用全体目标域或测试集拟合标准化参数。模型输入仅为这 54 个 `s1_...` 至 `s6_...` 特征列，不能混入 `sample_id`、`label`、`gas_type`、`batch`、日期或阶段/角色字段。原始 CSV 有两种字段集合：30 个 Period I 文件没有后期采集元数据，但六个传感器与时间列均存在。

## 5. 冻结的 S1-S7 划分

成员表的角色列为 `source_train`、`source_val`、`target_labeled_train`、`target_domain`、`target_val`、`target_test`。`target_domain` 只能提供**无标签特征**；`target_test` 标签只能在预测固定后评分。`source_val`、`target_val` 的标签仅用于所声明的 checkpoint/epoch 选择。S1、A-D 使用固定训练轮数；S2-S7 使用对应验证集选 checkpoint，**不把验证样本重新合入训练（no refit）**。

| Split | 有标签训练 | 验证集 | 无标签适应 `target_domain` | 测试 `target_test` |
| --- | --- | --- | --- | --- |
| S1 | Period I 全部 30 | 无 | Batch2-Batch6，全 273 | Batch2-Batch6，全 273 |
| S2 | Period I 固定 25 | Period I 固定 5 | Batch2-Batch6，全 273 | Batch2-Batch6，全 273 |
| S3 | Period I 固定 20 | Period I 固定 10 | Batch2-Batch6，全 273 | Batch2-Batch6，全 273 |
| S4 | Period I 全部 30 | Day 1-Day 2 的目标标签 | Day 3-Day 24，245 | Day 3-Day 24，245 |
| S5 | Period I 30 + 有标签 Day 1（16） | Day 2-Day 3 的目标标签 | Day 4-Day 24，231 | Day 4-Day 24，231 |
| S6 | Period I 30 + 有标签 Day 1-Day 2（28） | Day 3-Day 5 的目标标签 | Day 6-Day 24，205 | Day 6-Day 24，205 |
| S7 | Period I 30 + 有标签 Day 1-Day 3（42） | Day 4-Day 5 的目标标签 | Day 6-Day 24，205 | Day 6-Day 24，205 |

S2/S3 的源域验证成员由固定的独立划分种子 42、按气体分层和文件名自然排序后确定；**模型训练 seed 不改变样本划分**。S1-S3 是无目标标签训练的 UDA；S4-S7 在训练或验证环节使用了明确列出的目标标签，不能与纯无监督设置混称。S1 把全部 Batch2-Batch6 视为一个混合目标域进行对齐。

## 6. 冻结的 A/B/C/D 划分

本组始终只有论文 Batch1 的 30 个样本提供训练标签；目标 Batch2-Batch6 只以无标签特征进入适应。A/B/C 的每个编号阶段各训练一个模型，D 对全部五个目标 batch **只训练一个模型**。多个目标 batch 出现时，A/B/C/D 的对齐按 Batch1-每个目标 batch 分别计算，再对 batch 损失取等权均值，不把它们合成一个无差别目标域。C 的每一步从头初始化，不继承前一步参数或优化器状态。

| 阶段 | 训练时允许看到的无标签 batch | 用来评分的 batch |
| --- | --- | --- |
| A1-A5 | 分别为 Batch2、3、4、5、6 中的**当前一批** | 同一批 |
| B1 | Batch2 | Batch3（训练时未见） |
| B2 | Batch2-Batch3 | Batch4（训练时未见） |
| B3 | Batch2-Batch4 | Batch5（训练时未见） |
| B4 | Batch2-Batch5 | Batch6（训练时未见） |
| C1-C5 | 依次累积 Batch2；Batch2-3；...；Batch2-6 | 每一步的**最新一批** |
| D | 同时使用 Batch2-Batch6，按批分开对齐 | 同一个训练好并冻结的模型分别测 Batch2-Batch6 |

A/C/D 使用待测批次的**无标签**特征，属于声明的转导式无监督适应；B 只用历史无标签批次适应，评估的是完全未见的未来 batch，不能把 B 解释为“已适应当前目标批次”的 UDA。A1 与 C1 的数据成员一致。D 与 S1 看到同样的目标样本，但 D 按 batch 分开计算对齐项，S1 则合并目标域。

A6、C6、D6 是 Batch2-Batch6 的五个**逐 batch 准确率算术平均**；A7、C7、D7 是 Batch3-Batch6 的四个 batch 算术平均；B5 是 B1-B4 的四个结果算术平均。带撇号的 `A'6`、`C'6`、`D'6` 等是各自预测正确总数除以对应测试样本总数的**样本加权准确率**。这些编号仅为汇总指标，不是新的训练/测试划分。跨协议比较时必须同时说明可见目标数据范围、是否见过待测批次及采用哪种平均。

## 7. 复核信息与使用约束

本包的两个成员表已与生成原实验结果的划分代码按 `sample_id` 逐角色对照：S1-S7 的 42 组、A/B/C/D 的 90 组均完全一致。54D 文件和阶段审计文件的 SHA-256 记录在 `metadata/provenance.json`。所有原始文件均在 `metadata/sample_manifest.csv` 中有且只有一个对应样本。

公开数据含真实标签，便于独立评分；**不得**使用目标测试标签训练、选择 epoch、选择超参数、决定预处理或筛选 seeds。建议在运行前固定模型与评价方案，并保留逐 batch 的结果。
