# -*- coding: utf-8 -*-
"""
SUTrack ONNX -> RKNN 转换脚本

用法:
    python convert_onnx_to_rknn.py --onnx sutrack_t224.onnx --platform rk3588
    python convert_onnx_to_rknn.py --onnx sutrack_t224.onnx --platform rk3588 --quant --dataset dataset.txt

说明:
    - 输入 template/search 已经是预处理后的 float32 归一化张量(6 通道),
      因此转换时不配置 mean/std,RKNN 端不做额外预处理。
    - 默认不量化(float16),精度最接近原始 ONNX; 加 --quant 启用 int8 量化。
"""
import argparse
import os
import sys

from rknn.api import RKNN


def _patch_ort_for_auto_hybrid():
    """绕开 toolkit 2.3.2 自动混合量化 (hybrid_level>0) 的 bug:
    敏感度分析时把 (1,2,4) 的 template_anno 错误 squeeze 成 (2,4) 喂给
    onnxruntime, 报 'Got invalid dimensions for input: template_anno_int8'。
    在 InferenceSession.run 入口把被 squeeze 的输入补回 batch 维。"""
    import numpy as np
    import onnxruntime as ort
    _orig_run = ort.InferenceSession.run

    def _patched_run(self, output_names, input_feed, run_options=None):
        for k, v in list(input_feed.items()):
            if 'template_anno' in k:
                print(f'[patch] {k}: type={type(v).__name__}, shape={getattr(v, "shape", None)}', flush=True)
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    input_feed[k] = v[np.newaxis, ...]
                    print(f'[patch] {k} expanded to {input_feed[k].shape}', flush=True)
        return _orig_run(self, output_names, input_feed, run_options)

    ort.InferenceSession.run = _patched_run


def main():
    parser = argparse.ArgumentParser(description='SUTrack ONNX -> RKNN Converter')
    parser.add_argument('--onnx', type=str, default='./sutrack_t224.onnx', help='输入 ONNX 模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出 RKNN 路径 (默认与 onnx 同名)')
    parser.add_argument('--platform', type=str, default='rk3588',
                        choices=['rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588', 'rv1106', 'rv1103', 'rk2118'],
                        help='目标平台')
    parser.add_argument('--quant', action='store_true', help='启用 int8 量化 (默认关闭, 使用 float16)')
    parser.add_argument('--dataset', type=str, default=None, help='量化校准数据集列表文件 (txt)')
    parser.add_argument('--qdtype', type=str, default='w8a8',
                        choices=['w8a8', 'w8a16', 'w16a16i', 'w16a16i_dfp', 'w4a16'],
                        help='量化数据类型 (默认 w8a8; w8a16=权重int8激活int16, 精度显著好于 w8a8)')
    parser.add_argument('--qalgo', type=str, default='normal',
                        choices=['normal', 'mmse', 'kl_divergence', 'gdq'],
                        help='量化算法 (默认 normal; kl_divergence/mmse 通常精度更好)')
    parser.add_argument('--hybrid-level', type=int, default=0,
                        help='自动混合量化等级 (0=全量化; 越大保留越多浮点层)')
    parser.add_argument('--flash-attn', action='store_true', help='启用 flash attention 加速 (Transformer 模型)')
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f'Error: ONNX model not found: {args.onnx}')
        sys.exit(1)

    output_path = args.output or os.path.splitext(args.onnx)[0] + f'_{args.platform}.rknn'
    if args.quant:
        suffix = '_i8' if args.qdtype == 'w8a8' else f'_{args.qdtype}'
        if args.hybrid_level > 0:
            suffix += f'h{args.hybrid_level}'
        output_path = output_path.replace('.rknn', f'{suffix}.rknn')

    rknn = RKNN(verbose=True)

    print(f'==> Config (target_platform={args.platform}, quant={args.quant}, '
          f'qdtype={args.qdtype}, qalgo={args.qalgo}, hybrid_level={args.hybrid_level}, '
          f'flash_attn={args.flash_attn})')
    ret = rknn.config(target_platform=args.platform,
                      quantized_dtype=args.qdtype,
                      quantized_algorithm=args.qalgo,
                      quantized_hybrid_level=args.hybrid_level,
                      enable_flash_attention=args.flash_attn)
    if ret != 0:
        print('Config failed!')
        sys.exit(ret)

    print(f'==> Loading ONNX model: {args.onnx}')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print('Load ONNX failed!')
        sys.exit(ret)

    print('==> Building RKNN model ...')
    if args.quant:
        if not args.dataset or not os.path.exists(args.dataset):
            print('Error: --quant 需要提供 --dataset 校准数据列表文件')
            sys.exit(1)
        if args.hybrid_level > 0:
            _patch_ort_for_auto_hybrid()
        ret = rknn.build(do_quantization=True, dataset=args.dataset)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build failed!')
        sys.exit(ret)

    print(f'==> Exporting RKNN model: {output_path}')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export failed!')
        sys.exit(ret)

    rknn.release()
    print('Done.')


if __name__ == '__main__':
    main()
