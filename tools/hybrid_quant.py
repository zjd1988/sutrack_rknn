# -*- coding: utf-8 -*-
"""
RKNN 混合量化 (hybrid quantization) 脚本

step1: 加载 ONNX + 校准数据集, 自动分析 (proposal=True) 生成量化配置
       <model>.quantization.cfg (哪些层量化/哪些层保持浮点)
step2: 按配置构建混合精度模型并导出 .rknn

用法 (在开发板上执行, 注意 dataset 相对路径以 CWD 解析):
    cd calib && python ../hybrid_quant.py --onnx ../sutrack_t224.onnx --platform rk3576
    # 输出: ../sutrack_t224_rk3576_hybrid.rknn
"""
import argparse
import os
import sys

from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser(description='SUTrack hybrid quantization')
    parser.add_argument('--onnx', type=str, default='../sutrack_t224.onnx')
    parser.add_argument('--dataset', type=str, default='dataset.txt')
    parser.add_argument('--platform', type=str, default='rk3576')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no-proposal', action='store_true', help='关闭自动敏感层分析 (全量化)')
    parser.add_argument('--step', type=int, default=0, choices=[0, 1, 2],
                        help='0=两步都跑(默认), 1=只跑step1生成cfg, 2=用已有cfg跑step2')
    parser.add_argument('--cfg', type=str, default=None, help='step2 使用的 quantization.cfg 路径')
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f'Error: ONNX model not found: {args.onnx}')
        sys.exit(1)
    if not os.path.exists(args.dataset):
        print(f'Error: dataset not found: {args.dataset}')
        sys.exit(1)

    output = args.output or os.path.splitext(args.onnx)[0] + f'_{args.platform}_hybrid.rknn'
    model_name = os.path.splitext(os.path.basename(args.onnx))[0]
    data_input = f'{model_name}.data'
    quant_cfg = args.cfg or f'{model_name}.quantization.cfg'

    rknn = RKNN(verbose=True)
    print(f'==> Config (target_platform={args.platform})')
    assert rknn.config(target_platform=args.platform) == 0, 'config failed'

    print(f'==> Loading ONNX: {args.onnx}')
    assert rknn.load_onnx(model=args.onnx) == 0, 'load_onnx failed'

    if args.step in (0, 1):
        print(f'==> hybrid_quantization_step1 (dataset={args.dataset}, proposal={not args.no_proposal})')
        ret = rknn.hybrid_quantization_step1(dataset=args.dataset,
                                             proposal=not args.no_proposal)
        assert ret == 0, 'step1 failed'

    if args.step == 1:
        rknn.release()
        print(f'step1 done, cfg: {quant_cfg}')
        return

    for f in (data_input, quant_cfg):
        if not os.path.exists(f):
            print(f'Error: expected step1 output not found: {f}')
            print('CWD files:', [x for x in os.listdir('.') if x.startswith(model_name)])
            sys.exit(1)

    print(f'==> hybrid_quantization_step2 (cfg={quant_cfg})')
    ret = rknn.hybrid_quantization_step2(model_input=args.onnx,
                                         data_input=data_input,
                                         model_quantization_cfg=quant_cfg)
    assert ret == 0, 'step2 failed'

    print(f'==> Exporting: {output}')
    assert rknn.export_rknn(output) == 0, 'export failed'

    rknn.release()
    print('Done.')


if __name__ == '__main__':
    main()
