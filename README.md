# SUTrack-RKNN

将 [SUTrack](https://github.com/chenxin-dlut/SUTrack) 视觉跟踪模型(ONNX 版本,来自 [whyb/SUTrack-ONNX](https://github.com/whyb/SUTrack-ONNX))转换为 **RKNN** 格式(瑞芯微 NPU,如 RK3588),并在 **PC 上通过 RKNN 仿真器**验证转换后的模型精度。

## 项目结构

```
sutrack_rknn/
├── convert_onnx_to_rknn.py   # ONNX -> RKNN 转换脚本 (支持 FP16 / INT8 量化)
├── verify_rknn_accuracy.py   # PC 仿真器精度验证脚本 (与 ONNX Runtime 对比)
├── requirements.txt          # Python 依赖
└── README.md
```

> 模型文件(.onnx / .rknn)体积较大,不纳入仓库,请按下文指引下载或自行转换生成。

## 1. 模型说明

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

模型 zoo 及下载链接见 [SUTrack-ONNX Releases](https://github.com/whyb/SUTrack-ONNX/releases)(Tiny/Base/Large,224/384 分辨率,本流程对 5 个变体均适用)。

```bash
# 下载最小的 Tiny 模型 (106MB)
wget https://github.com/whyb/SUTrack-ONNX/releases/download/onnx/sutrack_t224.onnx
```

## 2. 环境搭建

rknn-toolkit2 对 Python 版本和依赖版本比较敏感,**推荐 Python 3.10**:

```bash
# 创建虚拟环境 (以 uv 为例, conda/venv 亦可)
uv venv --python 3.10 .venv
source .venv/bin/activate

pip install -r requirements.txt
```

注意事项(实测踩坑记录):

- rknn-toolkit2 2.3.2 最高支持 Python 3.12,但 `onnx` 需要 **≤ 1.17**(新版移除了 `onnx.mapping`),而 onnx ≤1.14 没有 cp312 预编译包,因此 Python 3.12 + onnx 组合不可行,**直接用 Python 3.10 最省事**。
- 新版 setuptools (≥81) 移除了 `pkg_resources`,需锁定 `setuptools<81`。

## 3. ONNX -> RKNN 转换

```bash
# FP16 (不量化, 精度最高), 目标平台 RK3588
python convert_onnx_to_rknn.py --onnx sutrack_t224.onnx --platform rk3588
# 输出: sutrack_t224_rk3588.rknn

# INT8 量化 (需要提供校准数据集列表 txt)
python convert_onnx_to_rknn.py --onnx sutrack_t224.onnx --platform rk3588 --quant --dataset dataset.txt
```

支持的 `--platform`:`rk3562 / rk3566 / rk3568 / rk3576 / rk3588 / rv1103 / rv1106 / rk2118`。

说明:模型输入 `template` / `search` 已经是预处理后的 float32 归一化张量(6 通道),因此转换时**不配置 mean/std**,RKNN 端不做额外预处理,预处理逻辑保持在应用侧(参考 [video_track_onnx.py](https://github.com/whyb/SUTrack-ONNX/blob/main/video_track_onnx.py))。

## 4. PC 仿真器精度验证

```bash
python verify_rknn_accuracy.py --onnx sutrack_t224.onnx --rknn sutrack_t224_rk3588.rknn --platform rk3588 --num-tests 5
```

验证内容:

- 各输出张量的**余弦相似度 / 最大绝对误差 / 平均绝对误差 / 相对 L2 误差**
- `score_map` 的 **argmax 位置**是否一致(跟踪取点)
- 按 SUTrack 后处理解码出的 **bbox IoU / 中心点偏移**

> 注意:RKNN PC 仿真器(`init_runtime(target=None)`)要求从原始 ONNX 现场 `load_onnx` + `build`;通过 `load_rknn` 加载的预编译 `.rknn` 文件**不能在仿真器上运行**。导出的 `.rknn` 文件用于真实 NPU 设备部署(rknn-lite 或 `init_runtime(target='rk3588')` 连板推理)。

## 5. 实测验证结果

`sutrack_t224` → RK3588 FP16,5 组随机样本,RKNN PC 仿真器 vs ONNX Runtime:

| 输出 | 余弦相似度 (均值) | 相对 L2 误差 (均值) |
| :--- | :---: | :---: |
| `score_map` | 0.999998 | 0.19% |
| `size_map` | 1.000000 | 0.06% |
| `offset_map` | 0.999999 | 0.15% |

- **score_map argmax 位置一致率: 5/5**(跟踪取点完全一致)
- **解码 bbox IoU ≥ 0.994**,中心点偏移 < 0.001(归一化坐标)

结论:FP16 转换基本无损,转换后的模型精度可靠。如需进一步压缩提速,可准备校准数据转 INT8 后用同一脚本复测精度。

## 6. 部署到 RK3588

将生成的 `.rknn` 文件拷贝到开发板,使用 rknn-toolkit-lite2:

```python
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn('sutrack_t224_rk3588.rknn')
rknn.init_runtime()  # 板端 NPU
outputs = rknn.inference(inputs=[template, search, template_anno], data_format='nchw')
```

前/后处理逻辑与 `verify_rknn_accuracy.py` 及原版 `video_track_onnx.py` 完全一致。

## 参考

- SUTrack 原始实现: https://github.com/chenxin-dlut/SUTrack
- ONNX 模型来源: https://github.com/whyb/SUTrack-ONNX
- RKNN Toolkit2: https://github.com/airockchip/rknn-toolkit2
