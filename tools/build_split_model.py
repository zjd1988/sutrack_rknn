# -*- coding: utf-8 -*-
"""
构建 mask 预处理拆分模型 (v2, 基于 anno-only 子图边界分析):
- 边界张量 onnx::Mul_592 / onnx::Mul_596 -> 新图输入 mask_0 / mask_1
- 剪掉 82 节点 anno-only 子图 (Less_42 等全部逻辑算子随之移除)
- ORT 验证输出与原始模型完全一致

输出: models/sutrack_t224_maskin.onnx
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import sys

sys.path.insert(0, 'python')
from verify_rknn_accuracy import make_test_inputs

MODEL = 'models/sutrack_t224.onnx'
OUT = 'models/sutrack_t224_maskin.onnx'
BOUNDARY = ['onnx::Mul_592', 'onnx::Mul_596']
NEW_INPUTS = ['mask_0', 'mask_1']

# 1. 提取 anno 子模型 (template_anno -> 2 个边界张量), 推断形状
onnx.utils.extract_model(MODEL, 'tmp_onnx/anno_submodel.onnx',
                         input_names=['template_anno'], output_names=BOUNDARY)
inputs = make_test_inputs(seed=0)
sess_sub = ort.InferenceSession('tmp_onnx/anno_submodel.onnx', providers=['CPUExecutionProvider'])
masks = sess_sub.run(None, {'template_anno': inputs['template_anno']})
for t, arr in zip(BOUNDARY, masks):
    print(f'{t}: shape={arr.shape} dtype={arr.dtype}')

# 2. 改接 + 剪枝
m = onnx.load(MODEL)
for ni, arr in zip(NEW_INPUTS, masks):
    m.graph.input.append(helper.make_tensor_value_info(
        ni, TensorProto.FLOAT, list(arr.shape)))
n_rewired = 0
for c in m.graph.node:
    for i, t in enumerate(c.input):
        if t in BOUNDARY:
            c.input[i] = NEW_INPUTS[BOUNDARY.index(t)]
            n_rewired += 1
print(f'rewired {n_rewired} input(s)')

out_names = [o.name for o in m.graph.output]
onnx.save(m, 'tmp_onnx/rewired2.onnx')
onnx.utils.extract_model('tmp_onnx/rewired2.onnx', OUT,
                         input_names=['template', 'search'] + NEW_INPUTS,
                         output_names=out_names)

m_out = onnx.load(OUT)
print('\n拆分后模型:')
print('  inputs :', [i.name for i in m_out.graph.input])
print('  outputs:', [o.name for o in m_out.graph.output])
n_orig, n_new = len(m.graph.node), len(m_out.graph.node)
print(f'  nodes: {n_orig} -> {n_new} (剪掉 {n_orig - n_new})')
logic = [n.name for n in m_out.graph.node if n.op_type in ('Less', 'Not', 'And', 'Or')]
print('  剩余逻辑算子:', logic if logic else '无')

# 3. ORT 对比验证
sess_a = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
sess_b = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
feed = {'template': inputs['template'], 'search': inputs['search'],
        'mask_0': masks[0], 'mask_1': masks[1]}
base = sess_a.run(None, inputs)
outs = sess_b.run(None, feed)
ok = True
for name, a, b in zip(out_names, base, outs):
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    ok &= (d == 0)
    print(f'  {name}: max diff = {d:.2e}')
print('验证通过: 输出完全一致' if ok else '存在差异!')
