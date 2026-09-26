# 真实视频跟踪性能实测 (鲁班猫3 / RK3576)

- 测试视频: OpenCV 官方测试片段 `vtest.avi` (768x576, 10 fps, DivX, 室外监控行人场景)
- 跟踪目标: 第一帧指定初始框 `496,156,38,76` (中间行人)
- 模型: `sutrack_t224_rk3576.rknn` (FP16)
- 管线: 与官方 `video_track_onnx.py` 完全一致 (模板更新间隔 25 帧, 阈值 0.7, Hann 窗)
- 命令: `python python/video_track_rknn.py --video vtest.avi --rknn sutrack_t224_rk3576.rknn --bbox 496,156,38,76 --save track_out.mp4`

## 结果 (Python 版, librknnrt 2.1.0)

- 处理 647 帧, 总耗时 91.73 s, **端到端 7.05 FPS (141.8 ms/帧)**
- 全程稳定跟踪目标行人 (置信度 0.45 ~ 0.87, 多人穿行场景未丢目标), 标注结果见 `track_out.mp4`

## 结果 (C++ 版, librknnrt 2.3.2, core-mask=3)

- 处理 647 帧, 总耗时 83.84 s, **端到端 7.72 FPS (129.6 ms/帧)**, 跟踪目标与 Python 版一致
- 相比 Python 版提升约 9.5%: 预处理更快 (7.7 -> 3.2 ms), NPU 推理 109.5 -> 98.2 ms (librknnrt 升级 + C++ 开销更低)

## C++ 版构建与运行

```bash
g++ -O2 -std=c++14 cpp/video_track_rknn.cpp -o video_track_rknn \
    -I./runtime $(pkg-config --cflags --libs opencv4) -lrknnrt
./video_track_rknn --video vtest.avi --model sutrack_t224_rk3576.rknn \
    --bbox 496,156,38,76 --core-mask 3 --save track_out_cpp.mp4
```

> 注: 两版均只解码出 647/795 帧 — 视频文件尾部有损坏块 (msmpeg4 decode error), 属正常现象。

## 分项耗时 (每帧平均)

| 阶段 | Python 版 | C++ 版 |
| :--- | ---: | ---: |
| 视频解码读取 | 3.6 ms | 4.3 ms |
| 预处理 (裁剪/resize/归一化) | 7.7 ms | 3.2 ms |
| **NPU 推理** | **109.5 ms** | **98.2 ms** |
| 后处理 (解码 bbox + 画框写视频) | 21.0 ms | 23.9 ms |
| **端到端** | **7.05 FPS** | **7.72 FPS** |

## 实时性结论

源视频 10 fps, 实测 7.05 FPS, **实时倍率 0.71x —— 达不到实时**, 瓶颈在 NPU 推理 (109.5 ms, 占 77%)。

说明: SUTrack 为 Transformer 结构, 部分算子 (如 SDPA/注意力相关) 在 RK3576 NPU 上会回退 CPU 执行, 这是推理耗时偏高的主要原因。

## 可选优化方向

1. **INT8 量化**: 准备校准数据后用 `convert_onnx_to_rknn.py --quant --dataset dataset.txt` 转 INT8, NPU 推理通常可提速 2~3 倍, 有望达到实时; 再用 `verify_rknn_on_board.py` 复测精度损失。
2. **流水线并行**: 解码/预处理 (CPU) 与 NPU 推理用生产者-消费者线程重叠, 可隐藏约 30 ms/帧, 端到端约可提升到 9 FPS。
3. **去掉结果写盘**: 不测速演示时去掉 `--save`, 后处理耗时可降一半以上。
4. **更新板端 librknnrt**: 当前系统自带 2.1.0 与 toolkit 2.3.2 不匹配, 替换为 wheel 内置的 2.3.2 运行时可能改善调度。
