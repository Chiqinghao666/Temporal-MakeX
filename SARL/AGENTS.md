# Repository Guidelines

## 项目结构与模块组织
- 根目录存放脚本型入口：`main.py` 负责训练与推理，`dataset.py` 生成子图样本，`algo.py` 汇集训练循环所需的图算法工具；`model.py`、`optimizer.py`、`rule_translator.py` 则分别定义结构感知变换器、调度优化器与规则后处理。
- `encoder/` 与 `decoder/` 目录包含多层结构感知编码器、解码器实现，修改注意保持 `sa_graph_encoder_model.py` 与 `rule_decoder.py` 的接口兼容；实验资产（例如 `results/` 日志）和原始数据默认放在 `DATASET/数据集名/` 下，提交前不要上传体积巨大的中间文件。

## 构建、测试与开发命令
- 预处理：`python dataset.py -data=FB15k-237 -maxN=32 -padding=8 -jump=1` 会在 `DATASET/FB15k-237/` 产生子图缓存（pickle）；调整 `-jump/-padding` 以配合目标关系阶数。
- 训练：`python main.py -data=DATASET/FB15k-237 -jump=1 -padding=8 -batch_size=16 -desc=fb237_run` 将在 `results/fb237_run.log` 写入 hit@k 与 MRR 指标，同时保存检查点到 `exps/`。
- 解码规则或快速验证：`python main.py ... -ckpt=exps/fb237_run/best.pt -decode_rule` 输出候选规则；建议把命令添加到 PR 描述便于复现实验。

## 编码风格与命名约定
- 统一使用 Python 3.8+ 与 PyTorch/Fairseq API，遵循 PEP8、四空格缩进、行宽 100 字符；新增模块务必写明 `__all__` 或导出函数，便于 `StructureAwareGraphTransformer` 组合。
- 变量名以蛇形命名（`tail_indexs`）、类使用帕斯卡命名（`RuleTranslator`），保持参数名与 CLI 选项一致，便于 argparse 自动映射；必要时补充最小 docstring 说明张量形状。
- 引入外部依赖前确认 requirements，优先使用现有工具函数（例如 `ScheduledOptim`）避免重复实现学习率调度。

## 测试指南
- 仓库暂无独立单元测试框架，贡献者需构建最小子图集（例如 FB15k 子集 500 条）并运行 `python main.py ... -epoch=1 --n_warmup_steps=50` 以验证前向、反向可运行。
- 对规则质量的 regression 测试请比较 `results/*.log` 中的 hit@10/MRR 差异，若变化超过 ±0.01，需在 PR 中解释原因与改动。
- 建议在本地 GPU 上固定随机种子（`-seed=31`）并记录显存占用；所有测试命令、指标与日志片段应附在 PR 描述或注释。

## 提交与 Pull Request 规范
- Git 历史以简洁祈使句为主，如 `train: tune sa_graph_encoder_layer dropout`；若同时修改数据与模型，请拆分提交突出单一主题，提交说明中附加最关键的 CLI。
- PR 描述必须包含：变更摘要、受影响的数据集/配置、复现实验命令、关键指标表，以及如有 UI/日志的截屏或文本片段；若引入新文件夹，说明其与 `encoder/decoder` 的依赖关系。
- 在合并前确保 `results/`、`exps/` 仅保留必要的示例配置；对于大文件请改用外部下载链接并在 README 的预处理部分补充说明。

## 数据与配置提示
- 训练依赖 `.pth` 或 `.pt` 检查点，请通过 `-exps` 指定隔离目录避免覆盖历史模型；长时间训练建议每 `-savestep` 轮写一次增量日志。
- `.env` 或敏感凭据不得入库，如需手动设置 GPU 排布，使用 `CUDA_VISIBLE_DEVICES=... python main.py ...` 并在文档中说明。
- 任何涉及新关系 schema 的改动都应更新 `dataset.py` 中的解析逻辑，并给出示例行格式（`head<TAB>relation<TAB>tail`）以便其他贡献者生成一致的子图。
