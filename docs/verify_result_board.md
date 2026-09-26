# RKNN 板端 NPU 实测验证记录 (鲁班猫3 / RK3576)

- 开发板: 鲁班猫3 (RK3576, 8 核, 8GB RAM), Debian 12 (bookworm), 内核 6.1.99-rk3576
- 模型: `sutrack_t224.onnx` (106MB) -> `sutrack_t224_rk3576.rknn` (63MB, FP16, do_quantization=False)
- 工具链: **转换与验证均在开发板上完成** — rknn-toolkit2 2.3.2 (manylinux aarch64) + rknn-toolkit-lite2 2.3.2, Python 3.11 (conda 环境 `rknn`)
- 对比基准: 板端 ONNX Runtime 1.26.0 (CPUExecutionProvider)
- NPU 运行时: librknnrt 2.1.0 (板端系统自带), RKNPU 驱动 0.9.8
- 连接方式: 板端网口 <-> PC USB 转网口直连, PC 开启 ICS 共享上网 (PC 侧 192.168.137.1, 板端 DHCP 得 192.168.137.130)

## 环境搭建 (板端)

```bash
conda create -y -n rknn python=3.11
conda activate rknn
pip install rknn-toolkit2==2.3.2 onnxruntime
pip install rknn-toolkit-lite2==2.3.2 'setuptools<81'
```

注意:

- rknn-toolkit2 从 2.x 起提供 manylinux aarch64 wheel, 因此可以直接在 RK3576 开发板上完成 ONNX -> RKNN 转换, 无需 x86 PC。
- 清华等国内镜像未收录 rknn-toolkit2, 需用官方 PyPI 源。
- `onnxoptimizer` 无 aarch64 预编译 wheel, pip 会现场 cmake 编译 (依赖 build-essential + cmake, 鲁班猫 Debian 镜像已自带), 编译约 10 分钟。
- 新版 setuptools (>=81) 移除了 `pkg_resources`, 必须 `pip install 'setuptools<81'`, 否则 `RKNN()` 初始化报 `ModuleNotFoundError: No module named 'pkg_resources'`。

## 转换 (板端)

```bash
python convert_onnx_to_rknn.py --onnx sutrack_t224.onnx --platform rk3576
# 输出: sutrack_t224_rk3576.rknn (63MB)
```

## 逐样本结果 (NPU vs ONNX Runtime)

### sample 0
| 输出 | shape | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| score_map | [1,1,14,14] | 0.999990 | 0.002312 | 0.000384 | 0.004528 |
| size_map | [1,2,14,14] | 0.999998 | 0.004334 | 0.000809 | 0.002010 |
| offset_map | [1,2,14,14] | 0.999994 | 0.032632 | 0.008282 | 0.003481 |

argmax idx: onnx=95, rknn=95 (match), score onnx=0.2432 / rknn=0.2434
bbox IoU=0.9899, center_shift=0.001579

### sample 1
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999982 | 0.002092 | 0.000540 | 0.006150 |
| size_map | 0.999997 | 0.006210 | 0.001017 | 0.002592 |
| offset_map | 0.999995 | 0.040753 | 0.009350 | 0.003576 |

argmax idx: onnx=177, rknn=177 (match), score onnx=0.2094 / rknn=0.2080
bbox IoU=0.9931, center_shift=0.000926

### sample 2
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999968 | 0.003790 | 0.000548 | 0.008103 |
| size_map | 0.999997 | 0.005403 | 0.001080 | 0.002536 |
| offset_map | 0.999980 | 0.072725 | 0.015045 | 0.006592 |

argmax idx: onnx=108, rknn=108 (match), score onnx=0.2684 / rknn=0.2683
bbox IoU=0.9964, center_shift=0.000151

### sample 3
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999975 | 0.002516 | 0.000696 | 0.007432 |
| size_map | 0.999993 | 0.008304 | 0.001368 | 0.003859 |
| offset_map | 0.999973 | 0.100310 | 0.015387 | 0.007458 |

argmax idx: onnx=109, rknn=109 (match), score onnx=0.3143 / rknn=0.3118
bbox IoU=0.9504, center_shift=0.006083

### sample 4
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999990 | 0.002820 | 0.000367 | 0.004593 |
| size_map | 0.999999 | 0.003336 | 0.000587 | 0.001438 |
| offset_map | 0.999992 | 0.041009 | 0.010118 | 0.004070 |

argmax idx: onnx=167, rknn=167 (match), score onnx=0.2645 / rknn=0.2617
bbox IoU=0.9859, center_shift=0.002220

## 汇总 (5 样本均值)

| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999981 | 0.002706 | 0.000507 | 0.006161 |
| size_map | 0.999997 | 0.005517 | 0.000972 | 0.002487 |
| offset_map | 0.999987 | 0.057486 | 0.011636 | 0.005035 |

- argmax 位置一致率: **5/5**
- 解码 bbox IoU: 0.95 ~ 0.996, 中心点偏移 < 0.007 (归一化坐标)
- **NPU 推理耗时: 115.9 ms/帧 (8.6 FPS)**, 20 次平均

## 结论

FP16 转换的 RKNN 模型在 RK3576 真实 NPU 上推理结果与 ONNX Runtime 高度一致 (余弦相似度 > 0.9999, 跟踪取点完全一致), 板端部署精度可靠。
相比 PC 仿真器 (rel_l2 约 0.1-0.2%), 真实 NPU 误差略大 (rel_l2 约 0.2-0.6%), 属正常水平, 不影响跟踪取点。

备注: 板端系统自带 librknnrt 2.1.0 与 toolkit 2.3.2 不完全匹配 (有 warning 但可正常运行); 如需消除可将 rknn-toolkit-lite2 wheel 内自带的 `librknnrt.so` 替换到系统库路径。
