# -*- coding: utf-8 -*-
"""
拆分模型 (4 输入) INT8 校准数据生成 — 合成数据版 (无真实视频时使用)

用 make_test_inputs 生成符合归一化分布的 template/search, template_anno 随机采样,
mask_0/mask_1 由 anno_submodel.onnx 计算 (与部署时预处理一致)。

注意: 合成数据仅用于服务器端快速验证量化链路; 正式部署应使用
tools/gen_calib_maskin_pc.py 在真实视频上采样的校准数据。

用法:
    python tools/gen_calib_maskin_synth.py --num 60 --out calib_maskin
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from verify_rknn_accuracy import make_test_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anno-sub', default='models/anno_submodel.onnx')
    ap.add_argument('--out', default='calib_maskin')
    ap.add_argument('--num', type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sess_mask = ort.InferenceSession(args.anno_sub, providers=['CPUExecutionProvider'])

    lines = []
    for i in range(args.num):
        inputs = make_test_inputs(seed=i)
        m0, m1 = sess_mask.run(None, {'template_anno': inputs['template_anno']})
        ft = os.path.join(args.out, f'{i:03d}_template.npy')
        fs = os.path.join(args.out, f'{i:03d}_search.npy')
        f0 = os.path.join(args.out, f'{i:03d}_mask0.npy')
        f1 = os.path.join(args.out, f'{i:03d}_mask1.npy')
        np.save(ft, inputs['template'])
        np.save(fs, inputs['search'])
        np.save(f0, m0.astype(np.float32))
        np.save(f1, m1.astype(np.float32))
        lines.append(f'{os.path.basename(ft)} {os.path.basename(fs)} '
                     f'{os.path.basename(f0)} {os.path.basename(f1)}')
        if (i + 1) % 10 == 0:
            print(f'[{i + 1}/{args.num}]', flush=True)

    with open(os.path.join(args.out, 'dataset.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'done: {args.num} samples -> {args.out}/dataset.txt')


if __name__ == '__main__':
    main()
