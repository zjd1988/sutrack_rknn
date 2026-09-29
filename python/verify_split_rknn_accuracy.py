# -*- coding: utf-8 -*-
"""
拆分模型 (mask 输入, 4 输入) 的 RKNN PC 仿真器精度验证脚本

对比链路:
    原始 ONNX (template, search, template_anno)            -> 参考输出
    拆分 RKNN (template, search, mask_0, mask_1) 仿真器推理 -> 待测输出
    其中 mask_0/mask_1 由 anno_submodel.onnx 从 template_anno 计算 (即部署时的预处理)

用法:
    # FP16
    python verify_split_rknn_accuracy.py --platform rk3576 \
        --export ../models/sutrack_t224_maskin_rk3576.rknn
    # INT8 (需先准备校准数据 dataset.txt, 每行: template.npy search.npy mask0.npy mask1.npy)
    python verify_split_rknn_accuracy.py --platform rk3576 --quant --dataset ../calib_maskin/dataset.txt \
        --export ../models/sutrack_t224_maskin_rk3576_i8.rknn
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_rknn_accuracy import make_test_inputs, cal_bbox, box_iou, compare_outputs


def main():
    parser = argparse.ArgumentParser(description='Split-model RKNN simulator accuracy verification')
    parser.add_argument('--onnx', type=str, default='./sutrack_t224.onnx', help='原始 ONNX (参考)')
    parser.add_argument('--split-onnx', type=str, default='./sutrack_t224_maskin.onnx', help='拆分后 ONNX')
    parser.add_argument('--anno-sub', type=str, default='./anno_submodel.onnx', help='mask 预处理子模型')
    parser.add_argument('--platform', type=str, default='rk3576')
    parser.add_argument('--quant', action='store_true', help='INT8 量化 (需 --dataset)')
    parser.add_argument('--hybrid', action='store_true',
                        help='混合量化 (hybrid_quantization_step1/2, 敏感层保留浮点, 需 --dataset)')
    parser.add_argument('--hybrid-level', type=int, default=0,
                        help='自动混合量化等级 (0=全量化; 越大保留越多浮点层, 配合 --quant 使用)')
    parser.add_argument('--dataset', type=str, default=None, help='量化校准数据集列表文件')
    parser.add_argument('--qdtype', type=str, default='w8a8',
                        choices=['w8a8', 'w8a16', 'w16a16i', 'w16a16i_dfp', 'w4a16'],
                        help='量化数据类型 (默认 w8a8; w8a16=权重int8激活int16, 精度显著好于 w8a8)')
    parser.add_argument('--qalgo', type=str, default='normal',
                        choices=['normal', 'mmse', 'kl_divergence', 'gdq'],
                        help='量化算法 (默认 normal)')
    parser.add_argument('--export', type=str, default=None, help='导出 .rknn 路径 (可选)')
    parser.add_argument('--num-tests', type=int, default=5)
    args = parser.parse_args()

    for p in (args.onnx, args.split_onnx, args.anno_sub):
        if not os.path.exists(p):
            print(f'Error: file not found: {p}')
            sys.exit(1)

    search_size = 384 if '384' in args.onnx else 224
    template_size = 192 if '384' in args.onnx else 112
    feat_sz = search_size // 16
    print(f'search_size={search_size}, template_size={template_size}, feat_sz={feat_sz}')

    import onnxruntime as ort
    sess_ref = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    sess_mask = ort.InferenceSession(args.anno_sub, providers=['CPUExecutionProvider'])
    out_names = [o.name for o in sess_ref.get_outputs()]
    print('ONNX outputs:', out_names)

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    print(f'==> config target_platform={args.platform} quant={args.quant} '
          f'qdtype={args.qdtype} qalgo={args.qalgo} hybrid_level={args.hybrid_level}')
    assert rknn.config(target_platform=args.platform,
                       quantized_dtype=args.qdtype,
                       quantized_algorithm=args.qalgo,
                       quantized_hybrid_level=args.hybrid_level) == 0, 'config failed'
    print(f'==> load_onnx: {args.split_onnx}')
    assert rknn.load_onnx(model=args.split_onnx) == 0, 'load_onnx failed'
    if args.hybrid:
        assert args.dataset and os.path.exists(args.dataset), '--hybrid 需要有效的 --dataset'
        model_name = os.path.splitext(os.path.basename(args.split_onnx))[0]
        data_input = f'{model_name}.data'
        quant_cfg = f'{model_name}.quantization.cfg'
        print(f'==> hybrid_quantization_step1 (dataset={args.dataset}, proposal=True)')
        assert rknn.hybrid_quantization_step1(dataset=args.dataset, proposal=True) == 0, 'step1 failed'
        for f in (data_input, quant_cfg):
            assert os.path.exists(f), f'expected step1 output not found: {f}'
        print(f'==> hybrid_quantization_step2 (cfg={quant_cfg})')
        assert rknn.hybrid_quantization_step2(model_input=args.split_onnx,
                                              data_input=data_input,
                                              model_quantization_cfg=quant_cfg) == 0, 'step2 failed'
    elif args.quant:
        assert args.dataset and os.path.exists(args.dataset), '--quant 需要有效的 --dataset'
        print(f'==> build (INT8, dataset={args.dataset})')
        assert rknn.build(do_quantization=True, dataset=args.dataset) == 0, 'build failed'
    else:
        print('==> build (FP16)')
        assert rknn.build(do_quantization=False) == 0, 'build failed'

    if args.export:
        assert rknn.export_rknn(args.export) == 0, 'export failed'
        print(f'==> exported: {args.export}')

    # PC 仿真器只能跑本会话 build 的模型
    print(f'==> init PC simulator (target_platform={args.platform})')
    assert rknn.init_runtime(target=None) == 0, 'init_runtime failed'

    agg = {}
    bbox_ok = 0
    ious = []
    for i in range(args.num_tests):
        inputs = make_test_inputs(seed=i, template_size=template_size, search_size=search_size)
        m0, m1 = sess_mask.run(None, {'template_anno': inputs['template_anno']})

        onnx_outs = sess_ref.run(None, inputs)
        rknn_outs = rknn.inference(inputs=[inputs['template'], inputs['search'],
                                           m0.astype(np.float32), m1.astype(np.float32)],
                                   data_format='nchw')

        results = compare_outputs(onnx_outs, rknn_outs, out_names)

        box_o, score_o, idx_o = cal_bbox(np.asarray(onnx_outs[0]).reshape(feat_sz, feat_sz),
                                         np.asarray(onnx_outs[1]), np.asarray(onnx_outs[2]), feat_sz)
        box_r, score_r, idx_r = cal_bbox(np.asarray(rknn_outs[0]).reshape(feat_sz, feat_sz),
                                         np.asarray(rknn_outs[1]).reshape(1, 2, feat_sz, feat_sz),
                                         np.asarray(rknn_outs[2]).reshape(1, 2, feat_sz, feat_sz), feat_sz)
        iou = box_iou(box_o, box_r)
        ious.append(iou)
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
        print(f'  bbox rknn={np.round(box_r, 4).tolist()}  IoU={iou:.4f}')

    print('\n========== 汇总 (mean over samples) ==========')
    for (name, k), v in agg.items():
        print(f'  {name:12s} {k:9s} = {np.mean(v):.6f}')
    print(f'  argmax 位置一致率: {bbox_ok}/{args.num_tests}')
    print(f'  bbox IoU: min={min(ious):.4f} mean={np.mean(ious):.4f}')

    rknn.release()


if __name__ == '__main__':
    main()
