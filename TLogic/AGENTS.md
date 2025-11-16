# Repository Guidelines

## 项目结构与模块组织
- 根目录包含 `pyproject.toml` 与 `poetry.lock`，用来锁定依赖；研究数据位于 `data/<dataset>`（例如 `icews14`），每个子目录提供 `train.txt`、`valid.txt`、`test.txt` 及 `entity2id.json` 等映射文件。
- 主要源代码集中在 `mycode/`：`learn.py` 负责规则学习、`apply.py` 负责规则应用、`evaluate.py` 负责指标评估，`rule_learning.py`、`rule_application.py`、`temporal_walk.py` 等模块拆分算法细节；`run.txt` 汇总复现实验的命令，`demo.ipynb` 展示端到端流程。
- 将新增组件置于 `mycode/` 下与功能最接近的位置，并同步更新 README 或运行脚本，避免散落脚本难以复用。

## 构建、测试与开发命令
- `poetry install`：一次性安装 Python 3.9–3.10 环境及锁定的依赖。
- `poetry run python learn.py -d icews14 -l 1 2 3 -n 200 -p 16 -s 12`：在 ICEWS14 上学习长度 1–3 的规则（`-p` 可根据 CPU 调整）。
- `poetry run python apply.py -d icews14 -r <rules.json> -l 1 2 3 -w 0 -p 8`：利用生成的规则打分候选。
- `poetry run python evaluate.py -d icews14 -c <cands.json>`：计算 MRR、Hits@k 等链接预测指标；复现实验可直接执行 `mycode/run.txt` 里的三条命令组合。

## 代码风格与命名约定
- 遵循 PEP 8：4 空格缩进、下划线命名函数与变量，类名使用帕斯卡命名。
- 依赖 `numpy`、`pandas`、`joblib`；如新增外部包，请先更新 `pyproject.toml` 并同步 `poetry.lock`。
- 复杂流程（如随机游走或并行调度）需配套注释与类型注解，便于复核与静态检查。

## 测试与验证准则
- 仓库未引入 pytest；验证以端到端复现实验为准，提交前至少在目标数据集上运行一轮 `learn → apply → evaluate`。
- 若优化子模块（如打分函数），请保存旧版候选文件并在同一随机种子下比对指标差异；性能更改建议追加 `--seed` 参数说明。

## Commit 与 Pull Request 指南
- 参考历史记录使用简短祈使句（如 `Update rule sampling logic`）；单次提交聚焦一类改动并附带必要的运行日志。
- PR 描述应包含：变更背景、关键修改点、验证命令及结果摘要；涉及界面或输出格式的改动请附示例（截图或文本片段），并在描述中链接相关 issue。

## 安全与配置提示
- 大规模运行前检查 `data/` 权限，避免意外覆盖原始分割；在多进程模式下优先使用本地 SSD。
- 长任务建议通过 `tmux`/`screen` 持久化会话，并记录 `rules.json` 与 `cands.json` 的命名规则（时间戳＋参数组合）以便复查。
