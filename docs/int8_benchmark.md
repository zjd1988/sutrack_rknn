# INT8 量化实测记录 (鲁班猫3 / RK3576)

- 校准数据: `tools/gen_calibration.py` 用 FP16 模型在 vtest.avi 上真实跟踪, 每 10 帧采样一组实际输入张量 (template/search/template_anno), 共 60 组 npy + dataset.txt
- 转换: `convert_onnx_to_rknn.py --onnx sutrack_t224.onnx --platform rk3576 --quant --dataset calib/dataset.txt`
- 产物: `sutrack_t224_rk3576_i8.rknn` (38MB, FP16 为 63MB)

## 速度: 大幅提升 ✅

| 指标 | FP16 | INT8 | 提升 |
| :--- | ---: | ---: | ---: |
| NPU 推理 | 109.5 ms/帧 | **14.6 ms/帧 (68.5 FPS)** | **7.5x** |
| 端到端视频跟踪 (Python, 无写盘) | 7.05 FPS | **38.39 FPS** | **5.4x** |
| 实时倍率 (@10fps 源) | 0.71x | **3.84x** | 可实时 |

INT8 端到端分项: 解码 3.5 ms + 预处理 7.5 ms + NPU 14.8 ms + 后处理 0.2 ms = 26.0 ms/帧。
**速度上足以支撑 25~30fps 实时跟踪** — 说明 FP16 慢的根因确实是算子回退 CPU, INT8 化后算子落回 NPU。

## 精度: 崩溃 ❌

| 输出 | cos_sim (FP16 参考) | cos_sim (INT8) |
| :--- | ---: | ---: |
| score_map | 0.999981 | 0.868550 |
| size_map | 0.999997 | 0.985377 |
| offset_map | 0.999987 | **-0.420595** |

- **argmax 位置一致率 0/5**, 解码 bbox 完全错误 (IoU=0), 跟踪不可用
- 根因: 推理日志大量报错 `ElementwiseLogical: unsupported A type: FLOAT, B type: INT8! Op type:Less, name: Less:Less_42, fallback cpu failed` — `Less_42` 算子两输入一个被量化成 INT8 一个保持 FLOAT, 类型不匹配且 CPU 回退失败, 输出错误向后传播

## 结论与下一步

朴素 INT8 全量化**不可直接用于本模型** (速度 7.5x 但精度归零)。可行路径是**混合量化 (hybrid quantization)**: 用 `rknn.hybrid_quantization_step1/step2` 让量化器自动分析敏感层, 把 `Less` 等敏感算子保留 FP16, 其余 INT8, 预期在精度和速度之间取得平衡 (速度介于 14.6~110 ms 之间, 精度接近 FP16)。
