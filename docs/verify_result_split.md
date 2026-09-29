# 拆分模型 (mask 输入) 转换与精度验证记录 — x86 服务器 PC 仿真器

- 环境: x86_64 服务器, Python 3.10 venv + rknn-toolkit2 2.3.2 + onnxruntime (CPU), 目标平台 rk3576
- 验证方式: RKNN PC 仿真器 (`init_runtime(target=None)`, 现场 load_onnx+build) vs ONNX Runtime 参考输出
- 测试输入: `make_test_inputs` 合成 5 组样本; mask_0/mask_1 由 `anno_submodel.onnx` 从 template_anno 计算 (与部署预处理一致)

## 拆分 (tools/build_split_model.py)

- 边界张量 `onnx::Mul_592` / `onnx::Mul_596` (形状均为 `(2,49,1)` float32) 改为图输入 `mask_0` / `mask_1`
- 节点数 1289 -> 1150 (剪掉 139 节点 anno-only 子图), **剩余逻辑算子: 无** (Less/Not/And/Or 全部移除)
- ORT 等价验证: 三个输出 max diff = 0.00e+00, 与原模型完全一致
- 产物: `models/sutrack_t224_maskin.onnx` (4 输入), `models/anno_submodel.onnx` (18KB 预处理子模型)

## FP16 (拆分模型) — 精度可靠 ✅

产物: `models/sutrack_t224_maskin_rk3576.rknn` (63MB)

| 输出 | cos_sim | rel_l2 |
| :--- | ---: | ---: |
| score_map | 0.999998 | 0.19% |
| size_map | 1.000000 | 0.06% |
| offset_map | 0.999999 | 0.15% |

- argmax 位置一致率 **5/5**, bbox IoU min=0.9944 mean=0.9967
- 与原 3 输入模型 FP16 的板端结果 (cos_sim ≥ 0.99998) 同一水平 — **拆分本身不损失精度**

## INT8 w8a8 (normal) — 仍然崩溃 ❌

产物: `models/sutrack_t224_maskin_rk3576_i8.rknn` (38MB); 校准: `tools/gen_calib_maskin_synth.py` 合成 60 组

| 输出 | cos_sim | rel_l2 |
| :--- | ---: | ---: |
| score_map | 0.721895 | 97.4% |
| size_map | 0.982833 | 19.4% |
| offset_map | 0.763819 | 64.9% |

- argmax 一致率 **0/5**, bbox IoU mean=0.26, 不可用
- 与拆分前板端 w8a8 的失败相比, **本次无任何算子回退报错** — 说明拆分成功消除了 `Less_42` 类型不匹配问题,
  但崩溃根因不止于此: Transformer 激活动态范围大, w8a8 逐层量化误差本身已足以毁掉输出

## INT8 w8a8 + kl_divergence — 改善但仍不可用 ❌

产物: `models/sutrack_t224_maskin_rk3576_i8_kld.rknn`

| 输出 | cos_sim | rel_l2 |
| :--- | ---: | ---: |
| score_map | 0.866506 | 55.0% |
| size_map | 0.990547 | 13.8% |
| offset_map | 0.916953 | 41.7% |

- argmax 一致率 1/5, bbox IoU mean=0.40 — 量化算法有改善但不够

## 其他量化配置

- `mmse`: 全模型逐层优化 >60min 未收敛, 中止 (kl_divergence 结果已说明量化算法只能改善不能根治)
- `hybrid_quantization_step1(proposal=True)` / `quantized_hybrid_level>0`: **均不可用** —
  自动敏感层分析要求模型支持 expand batch, 而拆分模型 mask 输入形状为 `(2,49,1)` 无 batch 维,
  报 `This model does not support expand batch, Therefore, the 'proposal' function cannot be used!`

### quantized_dtype 平台支持矩阵 (toolkit 2.3.2 实测探测, PyPI 最新即 2.3.2)

| dtype | rk3562 | rk3566/68 | rk3576 | rk3588 | rv1106/03 | rk2118 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| w8a8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| w8a16 | ✅ | ❌ | ❌ | **❌** | ❌ | ❌ |
| w16a16i(_dfp) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| w4a16 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |

- **w8a16 在 rk3588 上 config 直接拒绝** (`not support in 'rk3588'`), 无法测试
- 唯一支持 w8a16 的 rk3562 上, 本模型 build 阶段原生代码确定性崩溃
  (`malloc(): unsorted double linked list corrupted`, exit 134, 换 qalgo/线程/TMPDIR 均复现) — w8a16 对本模型不可评估

### w16a16i_dfp (16bit 权重+激活, rk3588 & rk3576) — 基本可用 ✅

产物: `models/sutrack_t224_maskin_rk3588_w16a16idfp.rknn` (59MB),
`models/sutrack_t224_maskin_rk3576_w16a16idfp.rknn` (两平台仿真器数值结果完全一致)

| 输出 | cos_sim | rel_l2 |
| :--- | ---: | ---: |
| score_map | 0.999867 | 1.63% |
| size_map | 0.999984 | 0.55% |
| offset_map | 0.999937 | 1.08% |

- argmax 一致率 **4/5**, bbox IoU min=0.586 mean=0.887
- 唯一失配样本 (seed=1) score_map 本身接近平坦 (conf≈0.21, 合成噪声输入), 微小扰动即翻转 argmax;
  真实跟踪 conf 0.45~0.87, 预期稳定性更好。精度远好于 w8a8, 代价是模型体积/算力接近 FP16

## 结论

1. **拆分转换本身精度无损**: ORT 逐元素 0 差异; FP16 RKNN 仿真器 cos_sim ≥ 0.999998, argmax 5/5, IoU ≥ 0.99,
   与原 3 输入模型 FP16 同一水平。
2. 拆分成功消除了 w8a8 时的 `Less` 算子类型不匹配/CPU 回退报错, 但 **w8a8 精度崩溃的根因是
   Transformer 激活量化误差本身** (normal/kl_divergence 均不可用)。
3. **w8a16 走不通**: rk3588/rk3576 均不支持; 唯一支持的 rk3562 上 toolkit 对本模型原生崩溃。
4. **w16a16i_dfp 精度基本可用** (rk3588/rk3576 均支持, cos_sim ≥ 0.99987, argmax 4/5),
   但 16bit 权重+激活的加速收益有限, 适合作为"精度优先"的量化兜底方案。
5. rk3576 上 INT8 大幅提速的可行路径只剩**手动混合量化**: `hybrid_quantization_step1(proposal=False)` 生成
   全量化 cfg 后, 人工将敏感层填入 `custom_quantize_layers: float16` 再 step2 (参考 `tools/run_hybrid_pipeline.sh`
   对原模型的做法)。敏感层选择需要逐层实验, 且建议使用真实视频校准数据, 适合在板端进行。

## 复现命令

```bash
PY=/workspace/SUTrack-ONNX/.venv-rknn/bin/python   # python3.10 + rknn-toolkit2 2.3.2
# 1. 拆分
$PY tools/build_split_model.py
# 2. 合成校准数据 (无真实视频时)
$PY tools/gen_calib_maskin_synth.py --num 60 --out calib_maskin
# 3. FP16 转换+验证
cd models && $PY ../python/verify_split_rknn_accuracy.py --platform rk3576 \
    --export sutrack_t224_maskin_rk3576.rknn --num-tests 5
# 4. INT8 转换+验证 (在 calib_maskin/ 下, dataset.txt 相对路径解析)
cd calib_maskin && $PY ../python/verify_split_rknn_accuracy.py \
    --onnx ../models/sutrack_t224.onnx --split-onnx ../models/sutrack_t224_maskin.onnx \
    --anno-sub ../models/anno_submodel.onnx --platform rk3576 \
    --quant --qdtype w8a8 --qalgo kl_divergence --dataset dataset.txt --num-tests 5
```

## 注意

- 本服务器无 vtest.avi, INT8 校准用的是**合成数据**; 正式部署应改用
  `tools/gen_calib_maskin_pc.py` 在真实视频上采样 (需要 vtest.avi)。
- PC 仿真器只能跑现场 build 的模型, 因此验证脚本内联了转换过程并顺带导出 .rknn。
