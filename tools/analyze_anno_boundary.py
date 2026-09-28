# -*- coding: utf-8 -*-
"""
正确求解 anno-only 子图: 从 template_anno 前向扩展, 只纳入"全部非常量输入都在子图内"的节点
边界 = 子图张量被子图外节点消费的位置 -> 替换为图输入即可把子图移到预处理
"""
import onnx
from collections import Counter

m = onnx.load('models/sutrack_t224.onnx')

producers = {}
consumers = {}
for n in m.graph.node:
    for o in n.output:
        producers[o] = n
    for i in n.input:
        consumers.setdefault(i, []).append(n)
inits = {i.name for i in m.graph.initializer}
# 常量: initializer + Constant 节点的输出
const_tensors = set(inits)
for n in m.graph.node:
    if n.op_type == 'Constant':
        const_tensors.update(n.output)
graph_inputs = {i.name for i in m.graph.input}

# anno-only 子图: 节点全部非常量输入都属于子图(或就是 template_anno)
in_sub_t = {'template_anno'}          # 子图内的张量
in_sub_n = set()                       # 子图内节点 id
changed = True
while changed:
    changed = False
    for n in m.graph.node:
        if id(n) in in_sub_n or n.op_type == 'Constant':
            continue
        inputs = [i for i in n.input if i and i not in const_tensors]
        if not inputs:
            continue
        if all(i in in_sub_t for i in inputs):
            in_sub_n.add(id(n))
            in_sub_t.update(n.output)
            changed = True

sub_nodes = [n for n in m.graph.node if id(n) in in_sub_n]
print(f'anno-only 子图节点数: {len(sub_nodes)}')
print('算子分布:', dict(Counter(n.op_type for n in sub_nodes)))

# 边界
boundary = {}
for t in sorted(in_sub_t - {'template_anno'}):
    outside = [c for c in consumers.get(t, []) if id(c) not in in_sub_n]
    if outside:
        boundary[t] = [(c.op_type, c.name) for c in outside]
print(f'\n边界张量数: {len(boundary)}')
for t, cs in boundary.items():
    print(f'  {t} -> {cs}')

# 边界张量形状
vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
print('\n边界张量形状:')
for t in boundary:
    if t in vi:
        shp = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in vi[t].type.tensor_type.shape.dim]
        dt = vi[t].type.tensor_type.elem_type
        print(f'  {t}: shape={shp} dtype={dt}')
    else:
        print(f'  {t}: (需运行推断形状)')

# 子图节点名单 (供后续剪枝用)
print('\n子图节点:')
for n in sorted(sub_nodes, key=lambda x: x.name):
    print(f'  {n.op_type:10s} {n.name}')
