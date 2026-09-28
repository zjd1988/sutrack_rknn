#!/bin/bash
# 混合量化全流程: 修 cfg -> step2 构建 -> 精度验证 -> 视频测速
# 用法: nohup bash run_hybrid_pipeline.sh > hybrid_pipeline.log 2>&1 &
set -e
cd /home/cat/sutrack
echo $$ > /home/cat/sutrack/pipeline.pid
trap 'echo FAILED > /home/cat/sutrack/pipeline.status' ERR
PY=~/miniconda3/envs/rknn/bin/python

echo "=== [1/4] patch cfg: custom_quantize_layers ==="
cd /home/cat/sutrack/calib
cp sutrack_t224.quantization.cfg.bak sutrack_t224.quantization.cfg
python3 - <<'EOF'
p = '/home/cat/sutrack/calib/sutrack_t224.quantization.cfg'
txt = open(p).read()
custom = """custom_quantize_layers:
    onnx::Mul_431: float16
    onnx::Less_449: float16"""
assert 'custom_quantize_layers: {}' in txt
txt = txt.replace('custom_quantize_layers: {}', custom)
open(p, 'w').write(txt)
print('patched custom_quantize_layers: onnx::Mul_431, onnx::Less_449 -> float16')
EOF

echo "=== [2/4] hybrid step2 build ==="
$PY -u ../hybrid_quant.py --onnx ../sutrack_t224.onnx --platform rk3576 --step 2
ls -lh /home/cat/sutrack/sutrack_t224_rk3576_hybrid.rknn

echo "=== [3/4] accuracy verify (hybrid vs onnxruntime) ==="
cd /home/cat/sutrack
$PY -u verify_rknn_on_board.py --onnx sutrack_t224.onnx --rknn sutrack_t224_rk3576_hybrid.rknn --num-tests 5

echo "=== [4/4] e2e speed test (200 frames) ==="
$PY -u video_track_rknn.py --video vtest.avi --rknn sutrack_t224_rk3576_hybrid.rknn --bbox 496,156,38,76 --max-frames 200

echo "=== ALL DONE ==="
echo DONE > /home/cat/sutrack/pipeline.status
