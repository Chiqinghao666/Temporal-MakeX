## 输出结构说明

为了快速定位 Makex 结果里的关键字段，可按照以下方式阅读：

### Path（路径）
- 存放位置：`Makex-main/global_explanations/rep.txt` 每一行的 **第二个列表项**。
- 含义：列表中的每个元素是一个形如 `[源节点ID, 目标节点ID, 边类型ID]` 的三元组。
- 读取方式：将边类型 ID 通过 `DataSets/icews14/relation2id.json` 映射到事件名称，即可得到“节点 A 通过事件 X 指向节点 B”的语义。

### Star（双星）
- 存放位置：同一行的 **第一、第二个列表项**联合描述：
  - 第一项 `[[节点ID, 节点标签], ...]` 列举了 Pattern 中的所有节点。
  - 第二项（即路径列表）中，凡是 **源节点 ID 等于列表里第一个节点** 的边被视为 “用户星 (User Star)”；源节点或目标节点等于第二个节点的边则组成 “物品星 (Item Star)”。
- 这些星形结构决定了局部拓扑，例如“用户节点 1 需要同时发出边 160、161”。

### Pattern（模式）
- 存放位置：`rep.txt` 每行整体就是一个 Pattern = `Star + Predicates + 统计指标`。
  - 第三个列表项记录谓词（Predicates），如 `['Constant', 1, 'Sector', 'Government', 'string', '=']`。
  - 第四个列表项 `[support, confidence]` 给出支持度与置信度。
- 因此，一个 Pattern 同时约束了 **两颗星形结构** 与 **节点属性**，即当用户星和物品星都满足这些边/属性关系时，该规则可用于解释推荐。

使用脚本：

```bash
# 查看全局规则的中文解释
python tools/interpret_global.py

# 查看 Icews 局部预测的中文说明
python tools/interpret_local.py
```

脚本会自动读取上面提到的字段，并在终端输出可读的报告。***
