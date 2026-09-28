# -*- coding: utf-8 -*-
"""
分析 sutrack_t224.onnx 中 Less_42 相关子图:
1. 上游链: Less_42 的输入依赖什么 (是否只依赖 template_anno + 常量)
2. 下游: Less_42 的输出流向哪里 (影响网络多深)
"""
import onnx

m = onnx.load('models/sutrack_t224.onnx')

producers = {}
consumers = {}
for n in m.graph.node:
    for o in n.output:
        producers[o] = n
    for i in n.input:
        consumers.setdefault(i, []).append(n)

inits = {i.name for i in m.graph.initializer}
graph_inputs = {i.name for i in m.graph.input}
print('graph inputs:', sorted(graph_inputs))

# ---- 上游分析 ----
target_op = None
for n in m.graph.node:
    if n.name == 'Less_42':
        target_op = n
print('\ntarget:', target_op.op_type, target_op.name, 'in:', list(target_op.input), 'out:', list(target_op.output))

seen = set()
stack = list(target_op.input)
upstream_nodes = set()
root_tensors = set()
while stack:
    t = stack.pop()
    if t in seen:
        continue
    seen.add(t)
    if t in inits:
        root_tensors.add(('CONST', t))
        continue
    if t in graph_inputs:
        root_tensors.add(('INPUT', t))
        continue
    p = producers.get(t)
    if p is not None:
        upstream_nodes.add((p.op_type, p.name))
        stack.extend(p.input)

print(f'\n上游中间张量数: {len(seen)}, 上游节点数: {len(upstream_nodes)}')
print('上游节点链 (op_type, name):')
for op, name in sorted(upstream_nodes, key=lambda x: x[1]):
    print(f'  {op:12s} {name}')
print('\n根依赖 (该子图的输入来源):')
for kind, t in sorted(root_tensors):
    print(f'  {kind}: {t}')

# ---- 下游分析 ----
out_t = target_op.output[0]
seen2 = set()
stack = [out_t]
downstream_nodes = set()
reaches_output = False
graph_outputs = {o.name for o in m.graph.output}
depth_max = 0
while stack:
    t = stack.pop()
    if t in seen2:
        continue
    seen2.add(t)
    if t in graph_outputs:
        reaches_output = True
    for c in consumers.get(t, []):
        downstream_nodes.add((c.op_type, c.name))
        stack.extend(c.output)

print(f'\n下游节点数: {len(downstream_nodes)}, 是否到达模型输出: {reaches_output}')
print('下游前 25 个节点:')
for op, name in sorted(downstream_nodes, key=lambda x: x[1])[:25]:
    print(f'  {op:12s} {name}')

# ---- 链上张量形状 ----
print('\n关键张量形状:')
vi = {v.name: v for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
import onnx.helper
for t in list(target_op.input) + [out_t]:
    if t in vi:
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in vi[t].type.tensor_type.shape.dim]
        dtype = vi[t].type.tensor_type.elem_type
        print(f'  {t}: shape={shape} dtype={dtype}')
    else:
        print(f'  {t}: (shape unknown)')
