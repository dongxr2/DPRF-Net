# 传统机器学习扩展对照协议（结果查看前冻结）

## 目的

扩充独立论文中的表格特征机器学习对照，删除以旧模型版本为参照的沿革式比较。新增结果只用于回答：在相同96维特征、相同留一负载划分和相同连续退化测试条件下，DPRF-Net与常见传统分类器的性能差异如何。

## 数据与划分

- 输入：41维振动特征与55维电气特征直接拼接，共96维。
- 外层划分：6折留一负载，负载为000、111、222、333、444、555。
- 优化/抽样种子：301、302、303、304、305。
- 标准化：仅在当前折的干净训练负载上拟合两模态StandardScaler。
- 测试：干净、两类模态缺失、6类增益衰减和6类零点漂移，共15种工况。
- 模态缺失：在标准化后将指定模态置零，与DPRF-Net测试口径一致。

## 固定退化混合训练

传统机器学习算法不使用渐进式或阶段式训练。每个原始训练样本生成2个训练副本；每个副本独立按固定概率选择输入状态：干净0.4，振动缺失、电气缺失、振动增益0.50、电气增益0.50、振动漂移0.50 RMS和电气漂移0.50 RMS各0.1。状态在模型拟合前一次性抽样并固定，训练样本总数为7 920/折。

## 预定义算法及超参数

1. Logistic Regression：C=1，class_weight=balanced，max_iter=5 000。
2. SVM-RBF：C=10，gamma=scale，class_weight=balanced。
3. KNN：k=5，距离加权，Euclidean距离。
4. Gaussian Naive Bayes：var_smoothing=1e-9。
5. Shrinkage LDA：lsqr求解，shrinkage=auto。
6. CART：min_samples_leaf=2，class_weight=balanced。
7. Random Forest：300棵树，max_features=sqrt，class_weight=balanced_subsample。
8. Extra Trees：300棵树，max_features=sqrt，class_weight=balanced。
9. Histogram Gradient Boosting：200次迭代，learning_rate=0.08，max_leaf_nodes=31，L2=0.1。
10. XGBoost：300棵树，max_depth=5，learning_rate=0.05，subsample=0.9，colsample_bytree=0.9。

## 输出和判读

- 主指标：15种输入工况中的14种退化工况平均Macro-F1。
- 同步报告：干净、缺失族、增益族、漂移族和最差单工况均值。
- 独立统计单位：6个留出负载；5个种子用于表征退化状态抽样及随机模型波动，不作为独立样本量。
- 不依据结果删改算法或超参数；失败算法保留并说明失败原因。
