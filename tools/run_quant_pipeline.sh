#!/bin/bash
# 量化变体全流程: 转换 -> 精度验证 -> 视频测速
# 用法: nohup bash run_quant_pipeline.sh [qdtype] [qalgo] [hybrid_level] > quant_pipeline.log 2>&1 &
set -e
cd /home/cat/sutrack
echo $$ > /home/cat/sutrack/pipeline.pid
trap 'echo FAILED > /home/cat/sutrack/pipeline.status' ERR
PY=~/miniconda3/envs/rknn/bin/python

QDTYPE=${1:-w8a16}
QALGO=${2:-kl_divergence}
HLEVEL=${3:-0}
if [ "$QDTYPE" = "w8a8" ]; then TAG="i8"; else TAG="$QDTYPE"; fi
[ "$HLEVEL" != "0" ] && TAG="${TAG}h${HLEVEL}"
RKNN=/home/cat/sutrack/sutrack_t224_rk3576_${TAG}.rknn

echo "=== [1/3] convert ($QDTYPE, $QALGO, hybrid_level=$HLEVEL) ==="
cd /home/cat/sutrack/calib
$PY -u ../convert_onnx_to_rknn.py --onnx ../sutrack_t224.onnx --platform rk3576 \
    --quant --dataset dataset.txt --qdtype $QDTYPE --qalgo $QALGO --hybrid-level $HLEVEL
ls -lh $RKNN

echo "=== [2/3] accuracy verify ==="
cd /home/cat/sutrack
$PY -u verify_rknn_on_board.py --onnx sutrack_t224.onnx --rknn $RKNN --num-tests 5

echo "=== [3/3] e2e speed test (200 frames) ==="
$PY -u video_track_rknn.py --video vtest.avi --rknn $RKNN --bbox 496,156,38,76 --max-frames 200

echo "=== ALL DONE ==="
echo DONE > /home/cat/sutrack/pipeline.status
