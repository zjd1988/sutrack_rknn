# RKNN PC 仿真器精度验证实测记录

- 模型: `sutrack_t224.onnx` (106MB) -> `sutrack_t224_rk3588.rknn` (58MB, FP16, do_quantization=False)
- 工具链: rknn-toolkit2 2.3.2, onnx 1.17.0, Python 3.10
- 环境: Linux x86_64 PC, RKNN 仿真器 (`init_runtime(target=None)`)
- 对比基准: ONNX Runtime (CPUExecutionProvider)
- 命令: `python verify_rknn_accuracy.py --onnx sutrack_t224.onnx --rknn sutrack_t224_rk3588.rknn --platform rk3588 --num-tests 5`

## 逐样本结果

### sample 0
| 输出 | shape | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| score_map | [1,1,14,14] | 0.999999 | 0.000678 | 0.000140 | 0.001621 |
| size_map | [1,2,14,14] | 1.000000 | 0.000989 | 0.000255 | 0.000589 |
| offset_map | [1,2,14,14] | 0.999999 | 0.016710 | 0.004049 | 0.001726 |

argmax idx: onnx=95, rknn=95 (match), score onnx=0.2432 / rknn=0.2429
bbox IoU=0.9944, center_shift=0.000860

### sample 1
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999999 | 0.000508 | 0.000118 | 0.001434 |
| size_map | 1.000000 | 0.000838 | 0.000224 | 0.000504 |
| offset_map | 0.999999 | 0.017263 | 0.003410 | 0.001476 |

argmax idx: onnx=177, rknn=177 (match), score onnx=0.2094 / rknn=0.2090
bbox IoU=0.9973, center_shift=0.000677

### sample 2
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999996 | 0.001123 | 0.000211 | 0.002792 |
| size_map | 0.999999 | 0.002418 | 0.000409 | 0.001037 |
| offset_map | 0.999998 | 0.017947 | 0.005041 | 0.002064 |

argmax idx: onnx=108, rknn=108 (match), score onnx=0.2684 / rknn=0.2678
bbox IoU=0.9956, center_shift=0.000729

### sample 3
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999998 | 0.000980 | 0.000147 | 0.001942 |
| size_map | 1.000000 | 0.000896 | 0.000234 | 0.000550 |
| offset_map | 1.000000 | 0.007999 | 0.002236 | 0.000933 |

argmax idx: onnx=109, rknn=109 (match), score onnx=0.3143 / rknn=0.3152
bbox IoU=0.9977, center_shift=0.000166

### sample 4
| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999999 | 0.000901 | 0.000139 | 0.001755 |
| size_map | 1.000000 | 0.001022 | 0.000216 | 0.000509 |
| offset_map | 0.999999 | 0.010307 | 0.002779 | 0.001148 |

argmax idx: onnx=167, rknn=167 (match), score onnx=0.2645 / rknn=0.2642
bbox IoU=0.9985, center_shift=0.000229

## 汇总 (5 样本均值)

| 输出 | cos_sim | max_abs | mean_abs | rel_l2 |
| :--- | ---: | ---: | ---: | ---: |
| score_map | 0.999998 | 0.000838 | 0.000151 | 0.001909 |
| size_map | 1.000000 | 0.001232 | 0.000268 | 0.000638 |
| offset_map | 0.999999 | 0.014045 | 0.003503 | 0.001469 |

- argmax 位置一致率: **5/5**
- 结论: FP16 转换基本无损。
