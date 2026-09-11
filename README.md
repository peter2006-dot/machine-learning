# PR03-01 条件启动子生成

第25组“机器学习综合实践”课程项目代码仓库。

本项目根据目标表达强度生成长度为 50 bp 的大肠杆菌启动子候选序列。计划比较条件 VAE 与条件自回归模型，并使用独立强度预测器以及 motif、GC、k-mer、新颖性和多样性指标统一评价生成结果。

## 当前版本

当前版本建立了所有成员共用的工程基础：

- TOML 配置加载与校验；
- 控制台和文件统一日志；
- 随机种子管理；
- 课程数据读取与合法性检查；
- 1-mer 至 3-mer 及 GC 特征；
- 可保存和加载的岭回归强度预测基线；
- CVAE 与自回归模型共用的生成器接口；
- 不依赖课程数据的端到端检查脚本。

岭回归模型用于验证数据、训练、评价、日志和保存流程是否连通。它是开发基线，不是最终的独立评估器。正式实验将使用相似性聚类划分替代当前的确定性随机划分，并比较更适合序列建模的预测器。

## 环境

需要 Python 3.10 或以上版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 快速检查

```bash
python scripts/smoke_test.py
```

成功时输出：

```text
Smoke test passed
```

## 在课程数据上训练开发基线

数据目录中需要包含：

- `promoter.npy`：一维 DNA 序列数组；
- `gene_expression.npy`：一维正数强度数组。

运行：

```bash
python scripts/train_strength_baseline.py \
  --data-dir /path/to/course/data \
  --output-dir outputs/ridge_strength_dev
```

程序会生成：

- `logs/pipeline.log`：带时间、级别和模块名称的运行日志；
- `outputs/ridge_strength_dev/model.npz`：模型参数；
- `outputs/ridge_strength_dev/metrics.json`：验证集和测试集指标。

## 配置

共享配置位于 `configs/base.toml`。随机种子、数据文件名、划分比例、特征范围、模型正则强度和日志位置均从配置读取。修改实验设置时，应新增配置文件或提交明确的配置修改，避免只在个人命令中保留参数。

## 目录

```text
configs/             共享实验配置
data/                数据说明与后续固定划分
scripts/             可直接运行的训练和检查入口
src/promoter_ml/     数据、模型、评价、日志等公共代码
outputs/             本地生成结果，不进入普通 Git 历史
logs/                本地运行日志，不进入普通 Git 历史
```

## 协作约定

- `main` 保存经过检查、能够运行的阶段版本；
- 每项工作在独立任务分支完成，通过合并请求进入 `main`；
- 开始工作前同步远程更新，当天完成的有效工作当天推送；
- 一个提交只完成一个可以说明清楚的改动；
- 数据接口、固定划分和评价规则的修改必须由另一名成员复核；
- 原始数据、模型权重、缓存和个人环境不直接提交到普通 Git 历史。

推荐提交说明：

```text
feat(data): add similarity-aware dataset split
feat(eval): add independent strength predictor
feat(cvae): implement conditional decoder
feat(ar): add temperature sampling
fix(split): prevent similar sequences crossing datasets
docs(m2): record EDA findings and reproduction commands
```
