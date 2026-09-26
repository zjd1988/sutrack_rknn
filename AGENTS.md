# AGENTS.md — SUTrack-RKNN 项目操作手册

本文件面向在本仓库工作的 AI 代理/开发者,记录环境约束、常用命令、开发板信息与已踩过的坑。项目介绍与实测结论见 [README.md](README.md),详细实测数据见 `docs/`。

## 项目概述

将 SUTrack 视觉跟踪模型 (ONNX) 转换为 RKNN 格式,在瑞芯微 NPU 上部署验证。目标开发板: **鲁班猫3 (RK3576, 双核 NPU, 8GB RAM, Debian 12 bookworm, 内核 6.1.99-rk3576)**。转换与验证可全部在开发板上完成 (rknn-toolkit2 2.x 提供 manylinux aarch64 wheel),PC 仿真器验证为可选项。

## 目录结构与约定

```
sutrack_rknn/
├── python/
│   ├── convert_onnx_to_rknn.py    # ONNX -> RKNN 转换 (FP16 / INT8)
│   ├── verify_rknn_accuracy.py    # PC 仿真器精度验证 (vs ONNX Runtime)
│   ├── verify_rknn_on_board.py    # 板端 NPU 精度验证 (rknn-lite2, 含测速)
│   └── video_track_rknn.py        # 板端视频跟踪测速 (Python 实现)
├── cpp/
│   ├── video_track_rknn.cpp       # 板端视频跟踪测速 (C++ 实现, rknn_api + OpenCV)
│   └── CMakeLists.txt
├── runtime/
│   ├── rknn_api.h                 # RKNN C API 头文件 (v2.3.2)
│   └── librknnrt.so               # RKNN aarch64 运行时 v2.3.2 (升级板端系统库用)
├── tools/
│   ├── board_ssh.py               # SSH 远程执行: python tools/board_ssh.py "<cmd>"
│   ├── board_push.py              # SFTP 推送: MSYS_NO_PATHCONV=1 python tools/board_push.py <files...> <remote_dir>
│   └── bench_multicore.py         # 多核 NPU 吞吐实测 (在板端运行)
├── docs/                          # 实测记录 (改动行为后需同步更新)
├── models/                        # 模型与测试视频 (*.onnx/*.rknn/*.avi/*.mp4, 全部不入库)
└── requirements.txt               # PC 端 Python 依赖
```

约定:

- **大文件不入库**: `models/` 及所有模型/视频文件已 gitignore; `runtime/librknnrt.so` 也不入库(需从 rknn-toolkit2 仓库 v2.3.2 标签重新下载)。
- 新增实测后,在 `docs/` 写记录并在 README 更新结论性数字。
- Python 与 C++ 两套跟踪实现的前/后处理必须与官方 [video_track_onnx.py](https://github.com/whyb/SUTrack-ONNX/blob/main/video_track_onnx.py) 保持一致(sample_target / process / cal_bbox / map_box_back 等),改动时需两边同步。

## 工具链与环境约束

PC 端 (转换 + 仿真器验证, 可选):

- **Python 3.10 最省事**; rknn-toolkit2 2.3.2 支持到 Python 3.12,但 onnx 需 ≤1.17 且无 cp312 预编译包,3.12 组合不可行。
- rknn-toolkit2 **PyPI 无 Windows wheel**(只有 manylinux x86_64 / aarch64),Windows PC 上不能装;需要 x86 Linux 或直接在开发板上转换。
- `pip install -r requirements.txt`,注意锁定 `setuptools<81`。

板端 (推荐路径):

```bash
conda create -y -n rknn python=3.11 && conda activate rknn
pip install rknn-toolkit2==2.3.2 onnxruntime        # 必须官方 PyPI 源, 清华等镜像未收录
pip install rknn-toolkit-lite2==2.3.2 'setuptools<81'
```

- `setuptools<81` 必须,否则 `RKNN()` 初始化报 `No module named 'pkg_resources'`。
- `onnxoptimizer` 无 aarch64 wheel,pip 现场 cmake 编译 (约 10 分钟,需 build-essential + cmake,鲁班猫镜像已自带)。
- 板端系统自带 librknnrt 2.1.0 与 toolkit 2.3.2 不匹配,**已升级为 2.3.2**:`sudo cp runtime/librknnrt.so /usr/lib/librknnrt.so && sudo ldconfig`(原库备份在 `/usr/lib/librknnrt.so.2.1.0.bak`)。

## 常用命令

转换 (板端或 x86 Linux):

```bash
python python/convert_onnx_to_rknn.py --onnx models/sutrack_t224.onnx --platform rk3576          # FP16
python python/convert_onnx_to_rknn.py --onnx models/sutrack_t224.onnx --platform rk3576 --quant --dataset dataset.txt  # INT8
```

精度验证:

```bash
python python/verify_rknn_accuracy.py --onnx models/sutrack_t224.onnx --rknn <rknn> --platform rk3588   # PC 仿真器 (x86 Linux)
python python/verify_rknn_on_board.py --onnx sutrack_t224.onnx --rknn sutrack_t224_rk3576.rknn          # 板端 NPU
```

视频跟踪测速 (板端):

```bash
# Python
python python/video_track_rknn.py --video vtest.avi --rknn sutrack_t224_rk3576.rknn --bbox 496,156,38,76 --save track_out.mp4
# C++ (板端编译)
g++ -O2 -std=c++14 cpp/video_track_rknn.cpp -o video_track_rknn -I./runtime $(pkg-config --cflags --libs opencv4) -lrknnrt
./video_track_rknn --video vtest.avi --model sutrack_t224_rk3576.rknn --bbox 496,156,38,76 --core-mask 3 --save out.mp4
# 多核吞吐实测
python tools/bench_multicore.py --rknn sutrack_t224_rk3576.rknn --repeat 30
```

## 开发板接入信息

- 地址: `192.168.137.130` (DHCP, 可能变化, 扫 192.168.137.x 的 22 端口可找到); 账号 `cat` / `temppwd` (sudo 同密码)。
- 连接方式: 板端网口 <-> PC USB 转网口直连,PC 对接板网卡开启了 **ICS 共享上网**(PC 侧 192.168.137.1,提供 DHCP+NAT,板端可上外网)。板端重启后自动获取 192.168.137.x。
- 板端工作目录: `/home/cat/sutrack/` (模型、脚本、输出均在此);conda 环境 `rknn` 在 `~/miniconda3/envs/rknn/`。
- PC 为 Windows + Git Bash: 调 `tools/board_push.py` 传远程路径时**必须**加 `MSYS_NO_PATHCONV=1`,否则 `/home/...` 会被 MSYS 转成 Windows 路径。
- GitHub 直连不稳,大文件用代理: `https://gh.ddlc.top/` (快) 或 `https://gh-proxy.com/` (慢但稳);`raw.githubusercontent.com` 通常可直连。
- 板端 SSH 长任务要用 `nohup ... > log 2>&1 &` + 轮询日志;paramiko 直连超时(300s)会断开并可能留下孤儿进程,注意 `pgrep -af 'pip install'` 清理。
- `pkill -f <pattern>` 会匹配到发起它的 bash 自身命令行,远程执行时用 `pgrep -f` 精确 PID 后 `kill`。

## 关键技术事实 (避免重复踩坑)

- **RKNN PC 仿真器** (`init_runtime(target=None)`) 只能跑现场 `load_onnx`+`build` 的模型;`load_rknn` 的预编译模型**不能**在仿真器运行。`.rknn` 预编译文件用于真实 NPU。
- **RKNN 模型按平台编译**: rk3588 平台产物不要直接拿去 rk3576 跑,用 `--platform rk3576` 重新转换。
- **多核 NPU 对本模型无效**: 单实例推理只落单核,`core_mask=NPU_CORE_0_1` 延迟仅改善 3%,双实例并发吞吐仅 1.14x — 瓶颈是 Transformer 注意力算子回退 CPU,提速靠 INT8 量化。详见 `docs/multicore_benchmark.md`。
- **当前性能**: FP16 端到端约 7.05 (Python) / 7.72 (C++) FPS @ vtest.avi 10fps,不能实时;NPU 推理 ~100ms/帧 占 77%。
- `vtest.avi` 尾部有损坏块,只能解码 647/795 帧,属正常现象,不是代码 bug。
- RKNN 推理输入顺序与 ONNX 一致 (template, search, template_anno),`data_format='nchw'`;转换时不配 mean/std(输入已是归一化张量)。
- 跟踪初始框: vtest.avi 第一帧中间行人 `496,156,38,76`。
- 板端 ping 不可用 (cat 用户无 cap_net_raw),测网络用 `curl -sI`。

## 验证基线 (改动回归用)

- 板端精度: cos_sim ≥ 0.9999,argmax 一致率 5/5,bbox IoU ≥ 0.95 (详见 `docs/verify_result_board.md`)。
- 视频跟踪: vtest.avi 全程 647 帧不丢目标,conf 0.45~0.87,端到端 ≥ 7 FPS。
