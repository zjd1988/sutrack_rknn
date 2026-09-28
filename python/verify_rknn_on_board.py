# -*- coding: utf-8 -*-
"""
RKNN 板端 NPU 精度验证脚本

在 RKNN 开发板 (如鲁班猫3 / RK3576) 上使用 rknn-toolkit-lite2 加载 .rknn 模型,
通过板端 NPU 推理, 与同机 ONNX Runtime 的结果进行对比:

    - 各输出的余弦相似度 / 最大绝对误差 / 平均绝对误差 / 相对 L2 误差
    - score_map 的 argmax 位置是否一致
    - 按 SUTrack 后处理解码出的 bbox 差异 (IoU / 中心点偏移)
    - NPU 单次推理耗时

用法 (在开发板上执行):
    python verify_rknn_on_board.py --onnx sutrack_t224.onnx --rknn sutrack_t224_rk3576.rknn --num-tests 5
"""
import argparse
import os
import sys
import time

import numpy as np


def make_test_inputs(seed=0, template_size=112, search_size=224):
    """构造符合真实数据分布的测试输入 (与 video_track_onnx.py 的 process() 输出分布一致)"""
    rng = np.random.RandomState(seed)

    def fake_image_tensor(h, w):
        img = rng.randint(0, 256, size=(h, w, 3)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img6 = np.concatenate([img, img], axis=-1)  # (H, W, 6)
        return img6.transpose(2, 0, 1).astype(np.float32)  # (6, H, W)

    template = np.stack([np.stack([fake_image_tensor(template_size, template_size),
                                   fake_image_tensor(template_size, template_size)], 0)],
                        0)  # (1, 2, 6, H, W)
    search = np.stack([np.stack([fake_image_tensor(search_size, search_size)], 0)], 0)  # (1, 1, 6, H, W)
    template_anno = rng.rand(1, 2, 4).astype(np.float32)
    return {'template': template, 'search': search, 'template_anno': template_anno}


def cal_bbox(score_map_ctr, size_map, offset_map, feat_sz):
    """与 video_track_onnx.py 一致的后处理"""
    score_map_flat = score_map_ctr.flatten()
    idx = int(np.argmax(score_map_flat))
    max_score = float(score_map_flat[idx])
    idx_y, idx_x = idx // feat_sz, idx % feat_sz
    size = size_map.reshape(2, -1)[:, idx]
    offset = offset_map.reshape(2, -1)[:, idx]
    bbox = np.array([(idx_x + offset[0]) / feat_sz,
                     (idx_y + offset[1]) / feat_sz,
                     size[0], size[1]], dtype=np.float32)
    return bbox, max_score, idx


def box_iou(a, b):
    """cx, cy, w, h (归一化) -> IoU"""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    ax2, ay2 = ax1 + a[2], ay1 + a[3]
    bx2, by2 = bx1 + b[2], by1 + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def compare_outputs(onnx_outs, rknn_outs, names):
    results = []
    for name, o, r in zip(names, onnx_outs, rknn_outs):
        o = np.asarray(o, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32).reshape(o.shape)
        of, rf = o.flatten(), r.flatten()
        cos = float(np.dot(of, rf) / (np.linalg.norm(of) * np.linalg.norm(rf) + 1e-12))
        abs_diff = np.abs(of - rf)
        results.append(dict(name=name, shape=list(o.shape), cos_sim=cos,
                            max_abs=float(abs_diff.max()), mean_abs=float(abs_diff.mean()),
                            rel_l2=float(np.linalg.norm(of - rf) / (np.linalg.norm(of) + 1e-12))))
    return results


def main():
    parser = argparse.ArgumentParser(description='RKNN on-board NPU accuracy verification')
    parser.add_argument('--onnx', type=str, default='./sutrack_t224.onnx')
    parser.add_argument('--rknn', type=str, default='./sutrack_t224_rk3576.rknn')
    parser.add_argument('--num-tests', type=int, default=5, help='随机测试样本数')
    parser.add_argument('--repeat', type=int, default=20, help='测速重复次数 (取最后一组输入)')
    parser.add_argument('--split', action='store_true', default=None,
                        help='拆分模型模式 (4 输入: template/search/mask_0/mask_1; 默认按文件名含 maskin 自动判断)')
    parser.add_argument('--anno-sub', type=str, default='./anno_submodel.onnx',
                        help='mask 预处理子模型路径 (split 模式需要)')
    args = parser.parse_args()

    if args.split is None:
        args.split = 'maskin' in os.path.basename(args.rknn)

    for p in (args.onnx, args.rknn):
        if not os.path.exists(p):
            print(f'Error: file not found: {p}')
            sys.exit(1)

    search_size = 384 if '384' in args.onnx else 224
    template_size = 192 if '384' in args.onnx else 112
    feat_sz = search_size // 16
    print(f'search_size={search_size}, template_size={template_size}, feat_sz={feat_sz}')

    import onnxruntime as ort
    ort_session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    out_names = [o.name for o in ort_session.get_outputs()]
    print('ONNX outputs:', out_names)

    sess_mask = None
    if args.split:
        if not os.path.exists(args.anno_sub):
            print(f'Error: anno sub model not found: {args.anno_sub}')
            sys.exit(1)
        sess_mask = ort.InferenceSession(args.anno_sub, providers=['CPUExecutionProvider'])
        print(f'split mode: mask from {args.anno_sub}')

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    print(f'==> Loading RKNN model: {args.rknn}')
    assert rknn.load_rknn(args.rknn) == 0, 'load_rknn failed'
    print('==> Init runtime on NPU ...')
    assert rknn.init_runtime() == 0, 'init_runtime failed'

    agg = {}
    bbox_ok = 0
    last_inputs = None
    for i in range(args.num_tests):
        inputs = make_test_inputs(seed=i, template_size=template_size, search_size=search_size)
        last_inputs = inputs

        onnx_outs = ort_session.run(None, inputs)
        if args.split:
            m0, m1 = sess_mask.run(None, {'template_anno': inputs['template_anno']})
            rknn_inputs = [inputs['template'], inputs['search'], m0, m1]
        else:
            rknn_inputs = [inputs['template'], inputs['search'], inputs['template_anno']]
        rknn_outs = rknn.inference(inputs=rknn_inputs, data_format='nchw')

        results = compare_outputs(onnx_outs, rknn_outs, out_names)

        box_o, score_o, idx_o = cal_bbox(np.asarray(onnx_outs[0]).reshape(feat_sz, feat_sz),
                                         np.asarray(onnx_outs[1]), np.asarray(onnx_outs[2]), feat_sz)
        box_r, score_r, idx_r = cal_bbox(np.asarray(rknn_outs[0]).reshape(feat_sz, feat_sz),
                                         np.asarray(rknn_outs[1]).reshape(1, 2, feat_sz, feat_sz),
                                         np.asarray(rknn_outs[2]).reshape(1, 2, feat_sz, feat_sz), feat_sz)
        iou = box_iou(box_o, box_r)
        center_shift = float(np.linalg.norm(box_o[:2] - box_r[:2]))
        same_idx = idx_o == idx_r
        bbox_ok += bool(same_idx)

        print(f'\n--- sample {i} ---')
        for r in results:
            print(f"  {r['name']:12s} shape={str(r['shape']):20s} cos_sim={r['cos_sim']:.6f} "
                  f"max_abs={r['max_abs']:.6f} mean_abs={r['mean_abs']:.6f} rel_l2={r['rel_l2']:.6f}")
            for k in ('cos_sim', 'max_abs', 'mean_abs', 'rel_l2'):
                agg.setdefault((r['name'], k), []).append(r[k])
        print(f'  score_map argmax idx: onnx={idx_o} rknn={idx_r} match={same_idx} '
              f'score(onnx={score_o:.4f}, rknn={score_r:.4f})')
        print(f'  bbox onnx={np.round(box_o, 4).tolist()}')
        print(f'  bbox rknn={np.round(box_r, 4).tolist()}  IoU={iou:.4f} center_shift={center_shift:.6f}')

    print('\n========== 汇总 (mean over samples) ==========')
    for (name, k), v in agg.items():
        print(f'  {name:12s} {k:9s} = {np.mean(v):.6f}')
    print(f'  argmax 位置一致率: {bbox_ok}/{args.num_tests}')

    # NPU 推理耗时
    if args.split:
        m0, m1 = sess_mask.run(None, {'template_anno': last_inputs['template_anno']})
        rknn_inputs = [last_inputs['template'], last_inputs['search'], m0, m1]
    else:
        rknn_inputs = [last_inputs['template'], last_inputs['search'], last_inputs['template_anno']]
    rknn.inference(inputs=rknn_inputs, data_format='nchw')  # warmup
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        rknn.inference(inputs=rknn_inputs, data_format='nchw')
    dt = (time.perf_counter() - t0) / args.repeat
    print(f'\nNPU 推理耗时: {dt * 1000:.1f} ms/帧 ({1.0 / dt:.1f} FPS, {args.repeat} 次平均)')

    rknn.release()


if __name__ == '__main__':
    main()
