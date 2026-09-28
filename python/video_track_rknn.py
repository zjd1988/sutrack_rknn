# -*- coding: utf-8 -*-
"""
SUTrack RKNN 开发板视频跟踪测速脚本

在 RKNN 开发板上对真实视频做端到端目标跟踪, 统计各阶段耗时与帧率:
    - 视频解码 / 预处理 / NPU 推理 / 后处理 分项耗时
    - 端到端平均帧率 (评估能否实时)

前/后处理与官方 video_track_onnx.py (SUTrack-ONNX) 完全一致。

用法 (在开发板上执行):
    python video_track_rknn.py --video vtest.avi --rknn sutrack_t224_rk3576.rknn \
        --bbox 496,156,38,76 --save track_out.mp4
"""
import argparse
import math
import os
import time

import cv2
import numpy as np


def sample_target(im, target_bb, search_area_factor, output_sz):
    x, y, w, h = target_bb
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1:
        raise Exception('Too small bounding box.')
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz
    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    resize_factor = output_sz / crop_sz
    im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
    return im_crop_padded, resize_factor


def process(img_arr):
    img_arr = np.concatenate([img_arr, img_arr], axis=-1)  # (H, W, 6)
    img_tensor = img_arr.astype(np.float32).transpose(2, 0, 1)  # (6, H, W)
    mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32).reshape(6, 1, 1)
    std = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32).reshape(6, 1, 1)
    return ((img_tensor / 255.0) - mean) / std


def transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz):
    box_extract_center = np.array(box_extract[0:2]) + 0.5 * np.array(box_extract[2:4])
    box_in_center = np.array(box_in[0:2]) + 0.5 * np.array(box_in[2:4])
    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = np.array(box_in[2:4]) * resize_factor
    box_out = np.concatenate((box_out_center - 0.5 * box_out_wh, box_out_wh))
    return box_out / (crop_sz - 1)


def clip_box(box, H, W, margin):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


def hann1d(sz, centered=True):
    if centered:
        return 0.5 * (1 - np.cos((2 * math.pi / (sz + 1)) * np.arange(1, sz + 1, dtype=np.float32)))


def hann2d(sz, centered=True):
    h1_0 = hann1d(sz[0], centered).reshape(1, 1, -1, 1)
    h1_1 = hann1d(sz[1], centered).reshape(1, 1, 1, -1)
    return h1_0 * h1_1


def cal_bbox(score_map_ctr, size_map, offset_map, feat_sz):
    score_map_flat = score_map_ctr.flatten()
    idx = int(np.argmax(score_map_flat))
    max_score = float(score_map_flat[idx])
    idx_y = idx // feat_sz
    idx_x = idx % feat_sz
    size = size_map.reshape(2, -1)[:, idx]
    offset = offset_map.reshape(2, -1)[:, idx]
    bbox = np.array([(idx_x + offset[0]) / feat_sz,
                     (idx_y + offset[1]) / feat_sz,
                     size[0], size[1]], dtype=np.float32)
    return bbox, max_score


def map_box_back(pred_box, state, resize_factor, search_size):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def main():
    parser = argparse.ArgumentParser(description='SUTrack RKNN on-board video tracking benchmark')
    parser.add_argument('--video', type=str, default='./vtest.avi')
    parser.add_argument('--rknn', type=str, default='./sutrack_t224_rk3576.rknn')
    parser.add_argument('--bbox', type=str, required=True, help='初始框 x,y,w,h (第一帧)')
    parser.add_argument('--save', type=str, default=None, help='保存标注结果视频 (如 track_out.mp4)')
    parser.add_argument('--max-frames', type=int, default=0, help='最多处理帧数 (0=全部)')
    parser.add_argument('--split', action='store_true', default=None,
                        help='拆分模型模式 (4 输入含 mask; 默认按文件名含 maskin 自动判断)')
    parser.add_argument('--anno-sub', type=str, default='./anno_submodel.onnx',
                        help='mask 预处理子模型路径 (split 模式需要)')
    args = parser.parse_args()

    if args.split is None:
        args.split = 'maskin' in os.path.basename(args.rknn)

    for p in (args.video, args.rknn):
        if not os.path.exists(p):
            print(f'Error: file not found: {p}')
            return
    init_bbox = [float(v) for v in args.bbox.split(',')]
    assert len(init_bbox) == 4, '--bbox 格式: x,y,w,h'

    search_size = 384 if '384' in args.rknn else 224
    template_size = 192 if '384' in args.rknn else 112
    feat_sz = search_size // 16

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    print(f'==> Loading RKNN model: {args.rknn}')
    assert rknn.load_rknn(args.rknn) == 0, 'load_rknn failed'
    print('==> Init runtime on NPU ...')
    assert rknn.init_runtime() == 0, 'init_runtime failed'

    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if not ret:
        print('Error: cannot read video')
        return
    print(f'video: {total_frames} frames, {video_fps:.1f} fps, {first_frame.shape[1]}x{first_frame.shape[0]}')

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, video_fps if video_fps > 0 else 25,
                                 (first_frame.shape[1], first_frame.shape[0]))

    template_factor = 2.0
    search_factor = 4.0
    num_templates = 2
    update_intervals = 25
    update_threshold = 0.7

    output_window = hann2d((feat_sz, feat_sz), centered=True)

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    state = list(init_bbox)

    z_patch_arr, resize_factor = sample_target(first_frame_rgb, state, template_factor, output_sz=template_size)
    template = process(z_patch_arr)
    template_list = [template] * num_templates
    prev_box_crop = transform_image_to_crop(state, state, resize_factor, template_size)
    template_anno_list = [prev_box_crop] * num_templates

    # split 模式: mask 预处理子模型 (输入 template_anno, 输出 mask_0/mask_1)
    sess_mask = None
    masks = [None, None]
    if args.split:
        if not os.path.exists(args.anno_sub):
            print(f'Error: anno sub model not found: {args.anno_sub}')
            return
        import onnxruntime as ort
        sess_mask = ort.InferenceSession(args.anno_sub, providers=['CPUExecutionProvider'])
        print(f'split mode: mask from {args.anno_sub}')

    def update_masks():
        if sess_mask is not None:
            anno_input = np.stack(template_anno_list, axis=0)[np.newaxis, ...].astype(np.float32)
            masks[0], masks[1] = sess_mask.run(None, {'template_anno': anno_input})

    update_masks()

    if writer is not None:
        b = [int(v) for v in state]
        cv2.rectangle(first_frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 0), 3)
        writer.write(first_frame)

    t_decode = t_pre = t_infer = t_post = 0.0
    frame_id = 0
    t_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t1 = time.perf_counter()

        frame_id += 1
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x_patch_arr, resize_factor = sample_target(frame_rgb, state, search_factor, output_sz=search_size)
        search = process(x_patch_arr)

        template_input = np.stack(template_list, axis=0)[np.newaxis, ...]
        search_input = np.stack([search], axis=0)[np.newaxis, ...]
        t2 = time.perf_counter()

        if args.split:
            rknn_inputs = [template_input, search_input, masks[0], masks[1]]
        else:
            template_anno_input = np.stack(template_anno_list, axis=0)[np.newaxis, ...].astype(np.float32)
            rknn_inputs = [template_input, search_input, template_anno_input]
        rknn_outs = rknn.inference(inputs=rknn_inputs, data_format='nchw')
        t3 = time.perf_counter()

        score_map = np.asarray(rknn_outs[0], dtype=np.float32).reshape(1, 1, feat_sz, feat_sz)
        size_map = np.asarray(rknn_outs[1], dtype=np.float32).reshape(1, 2, feat_sz, feat_sz)
        offset_map = np.asarray(rknn_outs[2], dtype=np.float32).reshape(1, 2, feat_sz, feat_sz)

        response = score_map * output_window
        pred_box, conf_score = cal_bbox(response, size_map, offset_map, feat_sz)
        pred_box = (pred_box * search_size / resize_factor).tolist()
        state = clip_box(map_box_back(pred_box, state, resize_factor, search_size), H, W, margin=10)

        if frame_id % update_intervals == 0 and conf_score > update_threshold:
            z_patch_arr, rf = sample_target(frame_rgb, state, template_factor, output_sz=template_size)
            template = process(z_patch_arr)
            template_list.append(template)
            if len(template_list) > num_templates:
                template_list.pop(1)
            prev_box_crop = transform_image_to_crop(state, state, rf, template_size)
            template_anno_list.append(prev_box_crop)
            if len(template_anno_list) > num_templates:
                template_anno_list.pop(1)
            update_masks()

        if writer is not None:
            b = [int(v) for v in state]
            cv2.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 0), 3)
            cv2.putText(frame, f'{conf_score:.2f}', (b[0], max(20, b[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(frame)
        t4 = time.perf_counter()

        t_decode += t1 - t0
        t_pre += t2 - t1
        t_infer += t3 - t2
        t_post += t4 - t3

        if frame_id % 100 == 0:
            el = time.perf_counter() - t_start
            print(f'  frame {frame_id}/{total_frames}: {frame_id / el:.1f} FPS, conf={conf_score:.3f}')

        if args.max_frames and frame_id >= args.max_frames:
            break

    elapsed = time.perf_counter() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    rknn.release()

    print('\n========== 性能统计 ==========')
    print(f'处理帧数: {frame_id}, 总耗时: {elapsed:.2f} s')
    print(f'端到端平均: {frame_id / elapsed:.2f} FPS ({elapsed / max(frame_id, 1) * 1000:.1f} ms/帧)')
    print(f'  解码读取: {t_decode / max(frame_id, 1) * 1000:6.1f} ms/帧')
    print(f'  预处理  : {t_pre / max(frame_id, 1) * 1000:6.1f} ms/帧')
    print(f'  NPU 推理: {t_infer / max(frame_id, 1) * 1000:6.1f} ms/帧')
    print(f'  后处理  : {t_post / max(frame_id, 1) * 1000:6.1f} ms/帧')
    print(f'源视频帧率: {video_fps:.1f} fps -> 实时倍率: {frame_id / elapsed / max(video_fps, 1e-6):.2f}x'
          f' ({"可达到实时" if frame_id / elapsed >= video_fps else "无法实时"})')
    if args.save:
        print(f'标注结果已保存: {args.save}')


if __name__ == '__main__':
    main()
