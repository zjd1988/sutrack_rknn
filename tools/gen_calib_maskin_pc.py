# -*- coding: utf-8 -*-
"""
PC 端生成拆分模型 (4 输入) 的 INT8 校准数据

用 ONNX Runtime 跑原始模型在 vtest.avi 上真实跟踪, 每 --stride 帧保存
实际送入拆分模型的 4 个输入 (template, search, mask_0, mask_1) 为 npy,
mask 由 anno_submodel.onnx 从 template_anno 计算 (即部署时的预处理)。

用法 (PC):
    .venv/python.exe tools/gen_calib_maskin_pc.py --num 60 --stride 10 --out calib_maskin
"""
import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, 'python')
from video_track_rknn import (sample_target, process, transform_image_to_crop,
                              clip_box, hann2d, cal_bbox, map_box_back)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='models/vtest.avi')
    ap.add_argument('--onnx', default='models/sutrack_t224.onnx')
    ap.add_argument('--anno-sub', default='models/anno_submodel.onnx')
    ap.add_argument('--bbox', default='496,156,38,76')
    ap.add_argument('--out', default='calib_maskin')
    ap.add_argument('--num', type=int, default=60)
    ap.add_argument('--stride', type=int, default=10)
    args = ap.parse_args()

    state = [float(v) for v in args.bbox.split(',')]
    os.makedirs(args.out, exist_ok=True)

    search_size, template_size = 224, 112
    feat_sz = search_size // 16
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    sess_mask = ort.InferenceSession(args.anno_sub, providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(args.video)
    ret, first = cap.read()
    assert ret
    template_factor, search_factor = 2.0, 4.0
    update_intervals, update_threshold = 25, 0.7
    output_window = hann2d((feat_sz, feat_sz), centered=True)

    first_rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    z, rf = sample_target(first_rgb, state, template_factor, output_sz=template_size)
    template = process(z)
    template_list = [template, template]
    anno = transform_image_to_crop(state, state, rf, template_size)
    anno_list = [anno, anno]

    lines = []
    saved = 0
    fid = 0
    while saved < args.num:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        H, W, _ = frame.shape
        frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, rf = sample_target(frgb, state, search_factor, output_sz=search_size)
        search = process(x)

        template_input = np.stack(template_list, 0)[np.newaxis, ...]
        search_input = np.stack([search], 0)[np.newaxis, ...]
        anno_input = np.stack(anno_list, 0)[np.newaxis, ...].astype(np.float32)

        outs = sess.run(None, {'template': template_input, 'search': search_input,
                               'template_anno': anno_input})
        resp = np.asarray(outs[0]).reshape(feat_sz, feat_sz) * output_window.reshape(feat_sz, feat_sz)
        pred, conf = cal_bbox(resp, np.asarray(outs[1]).reshape(1, 2, feat_sz, feat_sz),
                              np.asarray(outs[2]).reshape(1, 2, feat_sz, feat_sz), feat_sz)
        pred = (pred * search_size / rf).tolist()
        state = clip_box(map_box_back(pred, state, rf, search_size), H, W, margin=10)

        if fid % update_intervals == 0 and conf > update_threshold:
            z, rf2 = sample_target(frgb, state, template_factor, output_sz=template_size)
            template_list[1] = process(z)
            anno_list[1] = transform_image_to_crop(state, state, rf2, template_size)
            anno_input = np.stack(anno_list, 0)[np.newaxis, ...].astype(np.float32)

        if fid % args.stride == 0:
            m0, m1 = sess_mask.run(None, {'template_anno': anno_input})
            ft = os.path.join(args.out, f'{saved:03d}_template.npy')
            fs = os.path.join(args.out, f'{saved:03d}_search.npy')
            f0 = os.path.join(args.out, f'{saved:03d}_mask0.npy')
            f1 = os.path.join(args.out, f'{saved:03d}_mask1.npy')
            np.save(ft, template_input)
            np.save(fs, search_input)
            np.save(f0, m0)
            np.save(f1, m1)
            lines.append(f'{os.path.basename(ft)} {os.path.basename(fs)} '
                         f'{os.path.basename(f0)} {os.path.basename(f1)}')
            saved += 1
            print(f'[{saved}/{args.num}] frame {fid}, conf={conf:.3f}', flush=True)

    with open(os.path.join(args.out, 'dataset.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    cap.release()
    print(f'done: {saved} samples -> {args.out}/dataset.txt')


if __name__ == '__main__':
    main()
