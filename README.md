# SUTrack-RKNN

将 [SUTrack](https://github.com/chenxin-dlut/SUTrack) 视觉跟踪模型(ONNX 版本,来自 [whyb/SUTrack-ONNX](https://github.com/whyb/SUTrack-ONNX))转换为 **RKNN** 格式(瑞芯微 NPU),并完成了从转换、精度验证到真实视频跟踪测速的全流程验证。

- 实测平台: **鲁班猫3 开发板 (RK3576, 双核 NPU, Debian 12)**
- 转换与验证可**全部在开发板上完成**(rknn-toolkit2 2.x 提供 aarch64 支持),PC 仿真器验证为可选路径(x86 Linux)
- 提供 **Python / C++ 双实现**的视频跟踪测速程序,管线与官方 `video_track_onnx.py` 一致

> 面向 AI 代理/开发者的操作手册(环境搭建、命令、开发板接入、踩坑记录)见 [AGENTS.md](AGENTS.md)。

## 功能特性

- **ONNX -> RKNN 转换**: 支持 FP16 / INT8 量化,目标平台 rk3562 ~ rk3588(含 rk3576)
- **三级精度验证**: PC 仿真器 vs ONNX Runtime、板端 NPU vs ONNX Runtime(余弦相似度 / argmax 一致性 / bbox IoU)
- **真实视频跟踪测速**: 分项耗时统计(解码/预处理/NPU 推理/后处理),Python 与 C++ 双实现
- **多核 NPU 实测**: 单核/双核/双实例并发吞吐对比

## 模型说明

SUTrack ONNX 模型输入输出(以 `sutrack_t224` 为例):

| 输入 | 形状 | 说明 |
| :--- | :--- | :--- |
| `template` | `[1, 2, 6, 112, 112]` float32 | 2 帧模板, 6 通道归一化张量 |
| `search` | `[1, 1, 6, 224, 224]` float32 | 搜索区域, 6 通道归一化张量 |
| `template_anno` | `[1, 2, 4]` float32 | 模板对应的归一化标注框 |

| 输出 | 形状 |
| :--- | :--- |
| `score_map` | `[1, 1, 14, 14]` |
| `size_map` | `[1, 2, 14, 14]` |
| `offset_map` | `[1, 2, 14, 14]` |

模型 zoo(Tiny/Base/Large,224/384 分辨率,本流程对 5 个变体均适用)及下载见 [SUTrack-ONNX Releases](https://github.com/whyb/SUTrack-ONNX/releases)。模型与视频文件存放于 `models/`(不纳入仓库)。

## 快速开始

```bash
# 1. 板端(或 x86 Linux)安装工具链并转换, 详见 AGENTS.md
python python/convert_onnx_to_rknn.py --onnx models/sutrack_t224.onnx --platform rk3576

# 2. 板端 NPU 精度验证 (vs ONNX Runtime)
python python/verify_rknn_on_board.py --onnx sutrack_t224.onnx --rknn sutrack_t224_rk3576.rknn

# 3. 真实视频跟踪测速 (Python 与 C++ 双实现, 命令见 AGENTS.md)
python python/video_track_rknn.py --video vtest.avi --rknn sutrack_t224_rk3576.rknn --bbox 496,156,38,76 --save track_out.mp4
```

## 实测结果 (鲁班猫3 / RK3576)

### 精度: 板端 NPU vs ONNX Runtime (FP16, 5 组样本)

| 输出 | 余弦相似度 | 相对 L2 误差 |
| :--- | :---: | :---: |
| `score_map` | 0.999981 | 0.62% |
| `size_map` | 0.999997 | 0.25% |
| `offset_map` | 0.999987 | 0.50% |

- argmax 位置一致率 **5/5**,解码 bbox IoU 0.95 ~ 0.996 — FP16 转换在真实 NPU 上精度可靠

### 性能: 真实视频跟踪 (vtest.avi 768x576@10fps)

| 实现 | 端到端帧率 | NPU 推理 | 实时性 |
| :--- | ---: | ---: | :--- |
| Python | 7.05 FPS | 109.5 ms/帧 | 0.71x |
| C++ | 7.72 FPS | 98.2 ms/帧 | 0.77x |

- 全程 647 帧稳定跟踪目标行人;**FP16 暂不能实时**,瓶颈在 NPU 推理(Transformer 注意力算子回退 CPU)
- 多核 NPU 对本模型提升有限(双核并发仅 1.14x),有效提速路径: **INT8 量化**

### 拆分模型 (mask 输入) 与量化探索 (x86 服务器, PC 仿真器, 目标 rk3576/rk3588)

- **拆分转换无损**: anno-only 子图(含全部逻辑算子)剪为预处理子模型, 主模型变 4 输入; FP16 cos_sim ≥ 0.999998, argmax 5/5, IoU ≥ 0.99
- **w8a8 全量化不可用** (拆分后仍 argmax 0~1/5, 根因是激活量化误差而非算子回退); `w8a16` 在 rk3576/rk3588 均不被 toolkit 2.3.2 支持
- **w16a16i_dfp 基本可用** (rk3588/rk3576): cos_sim ≥ 0.99987, argmax 4/5, IoU mean 0.89 — 精度优先的量化兜底方案
- 自动混合量化对拆分模型不可用 (mask 输入无 batch 维); INT8 级提速只剩手动混合量化路径, 详见 [docs/verify_result_split.md](docs/verify_result_split.md)

## 实测记录文档

| 文档 | 内容 |
| :--- | :--- |
| [docs/verify_result.md](docs/verify_result.md) | PC 仿真器精度验证 (RK3588, x86 Linux) |
| [docs/verify_result_board.md](docs/verify_result_board.md) | 板端 NPU 精度验证 + 板端环境搭建记录 |
| [docs/video_benchmark.md](docs/video_benchmark.md) | 真实视频跟踪测速 (Python / C++ 对比) |
| [docs/multicore_benchmark.md](docs/multicore_benchmark.md) | 多核 NPU 吞吐实测 + librknnrt 升级方法 |
| [docs/verify_result_split.md](docs/verify_result_split.md) | 拆分模型 (mask 输入) 转换 + PC 仿真器精度验证 (x86 服务器) |

## 目录结构

```
sutrack_rknn/
├── python/      # 转换 / 仿真器验证 / 板端验证 / 视频跟踪 (Python)
├── cpp/         # 视频跟踪 (C++, rknn_api + OpenCV)
├── runtime/     # rknn_api.h + librknnrt.so v2.3.2
├── tools/       # 板端 SSH/SFTP 工具, 多核吞吐实测脚本
├── docs/        # 各阶段实测记录
├── models/      # 模型与测试视频 (不入库)
├── AGENTS.md    # 开发操作手册 (环境/命令/板端信息/踩坑)
└── README.md
```

## 参考

- SUTrack 原始实现: https://github.com/chenxin-dlut/SUTrack
- ONNX 模型来源: https://github.com/whyb/SUTrack-ONNX
- RKNN Toolkit2: https://github.com/airockchip/rknn-toolkit2
