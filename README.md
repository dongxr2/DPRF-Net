# DPRF-Net

DPRF-Net（Decoupled Preference–Reliability Fusion Network）是面向传感器质量退化的电机多源故障诊断模型。仓库包含论文最终实现、ESTOGU 变频驱动子集的论文级衍生特征、六折留一负载协议、深度基线入口、锁定实验结果等。

## 主要结果

在 6 个留出负载、5 个随机种子和 15 种测试工况下，DPRF-Net 获得：

- 正常工况 Macro-F1：99.93%
- 14 种退化工况平均 Macro-F1：98.95%
- 最差退化工况 Macro-F1：94.91%
- 参数量：52,235

这些结果是同一 ESTOGU 数据源上的留一负载内部验证，不构成跨设备外部验证。完整数字见 [results/dprf_locked_confirmation](results/dprf_locked_confirmation)。

## 仓库结构

```text
DPRF-Net_GitHub/
├── configs/                     # 论文锁定配置
├── data/
│   ├── processed/               # 4752 个窗口的压缩特征表
│   └── raw/                     # 原始 ESTOGU 数据下载与目录说明
├── docs/                        # 锁定实验协议
├── dprf/                        # 模型与公共实验组件
├── figures/                     # TikZ、PDF 和 PNG 架构图
├── results/                     # DPRF-Net 与深度基线的论文结果
├── scripts/
│   ├── train_dprf.py            # DPRF-Net/消融训练与六折评估
│   ├── analyze_results.py       # 配对统计、校准与类别稳定性分析
│   ├── build_features.py        # 从原始波形重建特征和退化工况
│   ├── run_classical_baselines.py # 10种传统机器学习对照
│   └── run_deep_baselines.py    # MLP、ResNet、FT-Transformer、TabM、TMC
└── tests/                       # 模型和数据完整性检查
```

## 环境安装

推荐 Python 3.10–3.12。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

若需运行 ResNet、FT-Transformer 和 TabM 对照：

```bash
python -m pip install -e ".[baselines]"
```

## 数据

仓库自带 13 个 `.csv.gz` 特征表：1 个正常工况、6 个增益衰减工况和 6 个零点漂移工况。两种模态缺失由评估脚本在折内标准化后置零生成，因此总计 15 种测试工况。每张表包含 4,752 行、41 维振动特征、55 维电气特征及样本元数据，并通过 `source_file + window_id` 一一配对。

原始波形来自 ESTOGU 的 With Driver 子集：

- Dataset: ESTOGU: A Multimodal Motor Condition Monitoring Dataset for Fault Diagnosis in Electrical Machines
- DOI: https://doi.org/10.5281/zenodo.18222578
- License: CC BY 4.0
- Official record: https://zenodo.org/records/18222578

详见 [data/README.md](data/README.md) 和 [data/DATA_LICENSE.md](data/DATA_LICENSE.md)。

## 快速检查

```bash
python -m pytest -q
```

检查会验证模型输出形状、校准权重归一化、残差开关范围，以及全部特征表的行数、列数和样本键一致性。

## DPRF-Net 烟雾实验

以下命令仅使用一个留出负载、一个随机种子和 2 个训练轮次，用于检查流程，不复现论文数值：

```bash
python scripts/train_dprf.py --run-name smoke --models dprf_full --smoke
```

输出位于 `results/runs/smoke/`。

## 完整论文实验

```bash
python scripts/train_dprf.py \
  --run-name dprf_paper_reproduction \
  --models dprf_full
```

脚本默认使用 `configs/locked.json` 中的 6 个负载、5 个随机种子、80 个训练轮次和单线程确定性设置。若要同时复现消融和同协议内部对照，省略 `--models`。

分析新结果：

```bash
python scripts/analyze_results.py results/runs/dprf_paper_reproduction
```

## 传统机器学习基线

论文表2中的 Logistic Regression、SVM-RBF、KNN、Gaussian NB、Shrinkage LDA、CART、Random Forest、Extra Trees、HistGradientBoosting 和 XGBoost 可通过以下命令复现：

```bash
python scripts/run_classical_baselines.py
```

该脚本采用固定退化训练混合、6个留出负载、5个随机种子和单线程执行；输出写入 `results/runs/classical_reproduction/`。论文结果保存在 `results/classical_baselines/`。

## 深度基线

```bash
python scripts/run_deep_baselines.py \
  --run-name deep_baselines_reproduction \
  --models mlp resnet ft_transformer tabm
```

TMC 依赖其作者的官方实现。由于上游仓库未提供明确的源码许可证，本仓库不复制该源码；获取和放置方法见 [third_party/README.md](third_party/README.md)。论文使用的 TMC 结果已经保存在 `results/deep_baselines/`。

## 从原始波形重建特征

将 Zenodo 的 `With_Driver_Dataset` 解压到 `data/raw/With_Driver_Dataset/`，然后运行：

```bash
python scripts/build_features.py
```

脚本将每条同步记录划分为 12 个长度为 4096 的互不重叠窗口，并对正常、增益和漂移波形重新提取 96 维双源特征。已有 `data/processed` 即该流程的论文版本，不必重复生成。

## 可复现性说明

- 以负载为分组单位执行六折留一负载验证。
- 标准化器仅在每折训练负载的正常样本上拟合。
- 所有退化工况沿用相同的 `source_file + window_id`。
- PyTorch 使用确定性算法，数据加载 `num_workers=0`，CPU线程固定为1。
- 论文统计以 6 个留出负载为配对单位，随机种子不作为独立样本。
- 锁定协议见 [docs/LOCKED_PROTOCOL_zh.md](docs/LOCKED_PROTOCOL_zh.md)。

## 引用

论文正式发表后，将在此补充论文 BibTeX。使用数据时还应引用 ESTOGU 官方 Zenodo 记录及其数据论文。

