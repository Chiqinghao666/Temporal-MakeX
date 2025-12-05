# Repository Guidelines

## 项目结构
- `global_explanations/`: 全局规则挖掘与 SARL 相关脚本（如 `run_sarl_mining.py`、模型导出）。
- `local_explanations/`: 局部解释与 Top-K 生成脚本，输出解释子图与 topk CSV。
- `pyMakex/`: C++/PyBind 扩展源码（`pyMakex.cpp`、`include/`）。
- `DataSets/icews14/`: 数据集（`processed/original_graph/icews_v.csv`、`icews_e.csv`、`entity2id.json`、`relation2id.json`）。
- `tools/`: 实用脚本（解释可视化、辅助转换等）。

## 构建与运行
- 编译 C++ 扩展（如需要）：`cd pyMakex && ./clean.sh && python setup.py install`。
- 训练 SARL（示例）：`python global_explanations/train_sarl.py --edge_file ../DataSets/icews14/processed/original_graph/icews_e.csv --num_entities 7128 --num_relations 230 --epochs 10 --cuda`。
- 挖掘路径/规则：`python global_explanations/run_sarl_mining.py --dataset_dir ../DataSets/icews14 --model_path ./sarl_model.pth --output_rep ./rep_sarl.txt --cuda`。
- 生成局部解释：`cd local_explanations && ./run_local_explanation.sh`。

## 代码风格
- Python：4 空格缩进，遵循 PEP 8；长行尽量不超 120 字符。
- C++：C++17，尽量使用 `clang-format`/项目已有格式；避免非 ASCII。
- 命名：文件与目录使用小写加下划线，类名用驼峰，常量全大写。

## 测试准则
- 若有单元测试，保持与源文件同路径/同模块命名（如 `tests/` 下 `test_xxx.py`）。
- 运行测试（示例）：`pytest` 或项目脚本提供的 `./run_tests.sh`（若存在）。
- 新增功能请附带最小可复现的用例。

## 提交与 PR 规范
- 提交信息：前缀动词+简述，例如 `fix: handle empty candidate set`、`feat: add time bin predicate`。
- PR 要求：描述变更目的、主要修改点、测试结果；如有界面/输出变更附截图或示例。
- 关联 issue：在描述中引用 `#123` 等编号（如适用）。

## 额外提示
- 数据路径敏感，请核对脚本中的相对路径（特别是 `DataSets/icews14/processed/original_graph`）。
- GPU/CPU 环境差异较大，运行前确认 `--cuda` 与依赖是否可用；无法用 GPU 时请移除 `--cuda` 参数。
