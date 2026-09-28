# -*- coding: utf-8 -*-
"""
INT8 量化校准数据生成脚本

用 FP16 RKNN 模型在真实视频上跑跟踪管线 (与 video_track_rknn.py 一致),
每隔 --stride 帧把实际送入模型的三个输入张量保存为 npy, 并生成 dataset.txt
(rknn build(do_quantization=True) 的校准数据集格式: 每行一个样本, 多输入空格分隔)。

用法 (在开发板上执行):
    python gen_calibration.py --video vtest.avi --rknn sutrack_t224_rk3576.rknn \
        --bbox 496,156,38,76 --out calib --num 60 --stride 10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_track_rknn import (sample_target, process, transform_image_to_crop,
                              clip_box, hann2d, cal_bbox, map_box_back)


def main():
    parser = argparse.ArgumentParser(description='Generate INT8 calibration dataset from real video')
    parser.add_argument('--video', type=str, default='./vtest.avi')
    parser.add_argument('--rknn', type=str, default='./sutrack_t224_rk3576.rknn')
    parser.add_argument('--bbox', type=str, required=True, help='初始框 x,y,w,h')
    parser.add_argument('--out', type=str, default='./calib', help='输出目录')
    parser.add_argument('--num', type=int, default=60, help='采样样本数')
    parser.add_argument('--stride', type=int, default=10, help='每隔多少帧采一个样本')
    args = parser.parse_args()

    import cv2

    init_bbox = [float(v) for v in args.bbox.split(',')]
    os.makedirs(args.out, exist_ok=True)

    search_size = 384 if '384' in args.rknn else 224
    template_size = 192 if '384' in args.rknn else 112
    feat_sz = search_size // 16

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    assert rknn.load_rknn(args.rknn) == 0
    assert rknn.init_runtime() == 0

    cap = cv2.VideoCapture(args.video)
    ret, first_frame = cap.read()
    assert ret, 'cannot read video'

    template_factor, search_factor = 2.0, 4.0
    num_templates, update_intervals, update_threshold = 2, 25, 0.7
    output_window = hann2d((feat_sz, feat_sz), centered=True)

    first_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    state = list(init_bbox)
    z_patch, rf = sample_target(first_rgb, state, template_factor, output_sz=template_size)
    template = process(z_patch)
    template_list = [template] * num_templates
    anno = transform_image_to_crop(state, state, rf, template_size)
    template_anno_list = [anno] * num_templates

    dataset_lines = []
    saved = 0
    frame_id = 0
    while saved < args.num:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_patch, rf = sample_target(frame_rgb, state, search_factor, output_sz=search_size)
        search = process(x_patch)

        template_input = np.stack(template_list, axis=0)[np.newaxis, ...]
        search_input = np.stack([search], axis=0)[np.newaxis, ...]
        anno_input = np.stack(template_anno_list, axis=0)[np.newaxis, ...].astype(np.float32)

        outs = rknn.inference(inputs=[template_input, search_input, anno_input], data_format='nchw')
        score_map = np.asarray(outs[0], dtype=np.float32).reshape(feat_sz, feat_sz) * output_window.reshape(feat_sz, feat_sz)
        pred_box, conf = cal_bbox(score_map, np.asarray(outs[1]).reshape(1, 2, feat_sz, feat_sz),
                                  np.asarray(outs[2]).reshape(1, 2, feat_sz, feat_sz), feat_sz)
        pred_box = (pred_box * search_size / rf).tolist()
        state = clip_box(map_box_back(pred_box, state, rf, search_size), H, W, margin=10)

        if frame_id % update_intervals == 0 and conf > update_threshold:
            z_patch, rf2 = sample_target(frame_rgb, state, template_factor, output_sz=template_size)
            template_list[1] = process(z_patch)
            template_anno_list[1] = transform_image_to_crop(state, state, rf2, template_size)

        if frame_id % args.stride == 0:
            ft = os.path.join(args.out, f'{saved:03d}_template.npy')
            fs = os.path.join(args.out, f'{saved:03d}_search.npy')
            fa = os.path.join(args.out, f'{saved:03d}_anno.npy')
            np.save(ft, template_input)
            np.save(fs, search_input)
            np.save(fa, anno_input)
            dataset_lines.append(f'{os.path.basename(ft)} {os.path.basename(fs)} {os.path.basename(fa)}')
            saved += 1
            print(f'[{saved}/{args.num}] frame {frame_id}, conf={conf:.3f}', flush=True)

    with open(os.path.join(args.out, 'dataset.txt'), 'w') as f:
        f.write('\n'.join(dataset_lines) + '\n')

    cap.release()
    rknn.release()
    print(f'done: {saved} samples -> {args.out}/dataset.txt')


if __name__ == '__main__':
    main()
