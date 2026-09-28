#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quantization.cfg 外科手术式修改: 把指定张量的 dtype 改为 float32
(按行编辑, 不动其他任何内容 — YAML 整体读写会破坏 float32 精度导致 step2 校验失败)

用法:
    python fix_quant_cfg.py --cfg calib/sutrack_t224.quantization.cfg \
        --tensors onnx::Less_449,onnx::Mul_431
"""
import argparse
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--tensors', required=True, help='逗号分隔的张量名')
    args = ap.parse_args()

    targets = set(args.tensors.split(','))
    with open(args.cfg) as f:
        lines = f.readlines()

    out = []
    i = 0
    changed = []
    block_re = re.compile(r'^    (\S+):\s*$')
    while i < len(lines):
        m = block_re.match(lines[i])
        if not m or m.group(1) not in targets:
            out.append(lines[i])
            i += 1
            continue
        name = m.group(1)
        # 复制块内容, 修改 dtype / scale / zero_point
        out.append(lines[i])
        i += 1
        while i < len(lines) and not block_re.match(lines[i]) and not lines[i].startswith(('custom_', 'quantize_')):
            line = lines[i]
            if re.match(r'^\s+dtype:\s*int(8|16)\s*$', line):
                out.append(re.sub(r'int(8|16)', 'float32', line))
                changed.append(name)
            elif re.match(r'^\s+(scale|zero_point):\s*$', line):
                # "scale:" 单独成行, 下一行是 "-   value", 合并为空列表并跳过值行
                out.append(line.rstrip('\n') + ' []\n')
                if i + 1 < len(lines) and re.match(r'^\s+-\s+', lines[i + 1]):
                    i += 1
            else:
                out.append(line)
            i += 1

    with open(args.cfg, 'w') as f:
        f.writelines(out)
    for n in changed:
        print(f'  {n} -> float32')
    print(f'修改 {len(changed)} 个张量, cfg saved')


if __name__ == '__main__':
    main()
