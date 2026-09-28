# -*- coding: utf-8 -*-
"""
ONNX 数值实验: 印证 Less_42 链对精度的影响 + 预处理拆分的可行性

实验 A: 提取 template_anno -> onnx::Not_450 子模型, 单独运行得到 mask
实验 B: 破坏 Less_42 输出 (置零), 对比最终输出掉点是否类似 INT8
实验 C: 把链输出替换为新图输入 (mask 作为预处理输入), 验证输出与基线完全一致
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import sys

sys.path.insert(0, 'python')
from verify_rknn_accuracy import make_test_inputs, cal_bbox, box_iou

MODEL = 'models/sutrack_t224.onnx'
FEAT_SZ = 14

inputs = make_test_inputs(seed=0)
anno = inputs['template_anno']

# ---------- 实验 A: 提取 mask 子模型 ----------
print('===== A. 提取 mask 子模型 =====')
onnx.utils.extract_model(MODEL, 'tmp_onnx/mask_model.onnx',
                         input_names=['template_anno'],
                         output_names=['onnx::Not_450'])
sess_mask = ort.InferenceSession('tmp_onnx/mask_model.onnx', providers=['CPUExecutionProvider'])
mask = sess_mask.run(None, {'template_anno': anno})[0]
print('mask shape:', mask.shape, 'dtype:', mask.dtype, 'true ratio: %.4f' % mask.mean())

# ---------- 基线 ----------
print('\n===== 基线 (原始模型) =====')
sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
base = sess.run(None, inputs)
base_names = [o.name for o in sess.get_outputs()]
box_b, score_b, idx_b = cal_bbox(np.asarray(base[0]).reshape(FEAT_SZ, FEAT_SZ),
                                 np.asarray(base[1]), np.asarray(base[2]), FEAT_SZ)
print(f'argmax idx={idx_b} score={score_b:.4f} bbox={np.round(box_b, 4).tolist()}')

def compare(outs, tag):
    print(f'--- {tag} ---')
    for name, b, o in zip(base_names, base, outs):
        bf, of = np.asarray(b).flatten(), np.asarray(o).flatten()
        cos = float(np.dot(bf, of) / (np.linalg.norm(bf) * np.linalg.norm(of) + 1e-12))
        print(f'  {name:12s} cos_sim={cos:.6f}')
    box, score, idx = cal_bbox(np.asarray(outs[0]).reshape(FEAT_SZ, FEAT_SZ),
                               np.asarray(outs[1]), np.asarray(outs[2]), FEAT_SZ)
    print(f'  argmax idx={idx} (base={idx_b}, match={idx == idx_b}) IoU={box_iou(box_b, box):.4f}')

# ---------- 实验 B: 破坏 Less_42 输出 ----------
print('\n===== B. 破坏 Less_42 输出 (置零) =====')
m = onnx.load(MODEL)
consumers = [n for n in m.graph.node if 'onnx::Not_450' in n.input]
print('onnx::Not_450 的消费者:', [(c.op_type, c.name) for c in consumers])
zero_init = helper.make_tensor('mask_corrupt', TensorProto.BOOL, list(mask.shape),
                               np.zeros_like(mask).flatten().tolist())
m.graph.initializer.append(zero_init)
for c in consumers:
    for i, t in enumerate(c.input):
        if t == 'onnx::Not_450':
            c.input[i] = 'mask_corrupt'
onnx.save(m, 'tmp_onnx/model_corrupt.onnx')
sess_c = ort.InferenceSession('tmp_onnx/model_corrupt.onnx', providers=['CPUExecutionProvider'])
compare(sess_c.run(None, inputs), 'mask 置零后的输出')

# ---------- 实验 C: mask 变为图输入 (预处理拆分) ----------
print('\n===== C. mask 作为新输入 (预处理拆分验证) =====')
m2 = onnx.load(MODEL)
mask_vi = helper.make_tensor_value_info('mask_in', TensorProto.BOOL, list(mask.shape))
m2.graph.input.append(mask_vi)
consumers2 = [n for n in m2.graph.node if 'onnx::Not_450' in n.input]
for c in consumers2:
    for i, t in enumerate(c.input):
        if t == 'onnx::Not_450':
            c.input[i] = 'mask_in'
onnx.save(m2, 'tmp_onnx/model_split.onnx')
sess_s = ort.InferenceSession('tmp_onnx/model_split.onnx', providers=['CPUExecutionProvider'])
feed = dict(inputs)
feed['mask_in'] = mask
outs_s = sess_s.run(None, feed)
diffs = [float(np.max(np.abs(np.asarray(b) - np.asarray(o)))) for b, o in zip(base, outs_s)]
print('与基线最大逐元素差异:', [f'{d:.2e}' for d in diffs])
print('拆分模型输出与基线完全一致' if all(d == 0 for d in diffs) else '存在差异(数值误差级)' )
