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


def main():
    parser = argparse.ArgumentParser(description='SUTrack ONNX -> RKNN Converter')
    parser.add_argument('--onnx', type=str, default='./sutrack_t224.onnx', help='输入 ONNX 模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出 RKNN 路径 (默认与 onnx 同名)')
    parser.add_argument('--platform', type=str, default='rk3588',
                        choices=['rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588', 'rv1106', 'rv1103', 'rk2118'],
                        help='目标平台')
    parser.add_argument('--quant', action='store_true', help='启用 int8 量化 (默认关闭, 使用 float16)')
    parser.add_argument('--dataset', type=str, default=None, help='量化校准数据集列表文件 (txt)')
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f'Error: ONNX model not found: {args.onnx}')
        sys.exit(1)

    output_path = args.output or os.path.splitext(args.onnx)[0] + f'_{args.platform}.rknn'
    if args.quant:
        output_path = output_path.replace('.rknn', '_i8.rknn')

    rknn = RKNN(verbose=True)

    print(f'==> Config (target_platform={args.platform}, quant={args.quant})')
    ret = rknn.config(target_platform=args.platform)
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
