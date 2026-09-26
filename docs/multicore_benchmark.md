# RK3576 多核 NPU 实测记录 (鲁班猫3)

- 模型: `sutrack_t224_rk3576.rknn` (FP16)
- 运行时: librknnrt **2.3.2** (已从系统自带 2.1.0 升级, 版本不匹配的 warning 消除)
- 脚本: `tools/bench_multicore.py` (30 次推理取平均)

## 结果

| 模式 | 耗时/帧 | 吞吐 | 相对基线 |
| :--- | ---: | ---: | ---: |
| A. 单实例 core_mask=AUTO | 109.8 ms | 9.10 FPS | 1.00x |
| B. 单实例 core_mask=NPU_CORE_0_1 | 106.7 ms | 9.37 FPS | 1.03x |
| C. 双实例双线程并发 (CORE_0_1) | - | 10.36 FPS (聚合) | 1.14x |

## 结论

1. **默认 (AUTO) 单实例推理只跑在一个 NPU 核上**。
2. **单模型单实例设置 core_mask=NPU_CORE_0_1 基本不降延迟 (1.03x)** — RKNN 单实例推理任务不会被拆分到多个核上执行。
3. **双实例双线程并发, 聚合吞吐也只有 1.14x** — 远达不到理论 2x。原因是该模型为 Transformer 结构, 相当部分算子 (注意力/SDPA 相关) 回退 CPU 执行, 两个推理线程在 CPU 部分互相争抢, NPU 双核无法充分体现。
4. 因此对 **本模型**, 多核 NPU 对性能提升有限; 提升性能的有效路径是 **INT8 量化** 或 **减少 CPU 回退算子**。
   - 注: 对于纯 CNN 类模型 (算子全部落 NPU), 双核并发通常可以获得接近 2x 吞吐; 多核更适合"多路视频流各起一个推理实例"的场景。

## librknnrt 升级方法

```bash
# 从 rknn-toolkit2 仓库获取 2.3.2 运行时 (已存于 runtime/librknnrt.so)
sudo cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak   # 备份
sudo cp runtime/librknnrt.so /usr/lib/librknnrt.so
sudo ldconfig
```

升级后验证日志中 `librknnrt version: 2.3.2`, 与 toolkit 2.3.2 匹配。
