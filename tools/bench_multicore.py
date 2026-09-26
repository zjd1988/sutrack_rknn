# -*- coding: utf-8 -*-
"""
RK3576 多核 NPU 吞吐实测脚本

对比三种模式的推理吞吐:
    A. 单实例, core_mask=AUTO (基线)
    B. 单实例, core_mask=NPU_CORE_0_1 (验证单模型能否利用双核降延迟)
    C. 双实例 + 双线程并发, core_mask=NPU_CORE_0_1 (验证多核并行吞吐)

用法:
    python bench_multicore.py --rknn sutrack_t224_rk3576.rknn --repeat 30
"""
import argparse
import threading
import time

import numpy as np


def make_inputs():
    rng = np.random.RandomState(0)
    template = rng.randn(1, 2, 6, 112, 112).astype(np.float32)
    search = rng.randn(1, 1, 6, 224, 224).astype(np.float32)
    anno = rng.rand(1, 2, 4).astype(np.float32)
    return [template, search, anno]


def new_instance(rknn_path, core_mask):
    from rknnlite.api import RKNNLite
    r = RKNNLite(verbose=False)
    assert r.load_rknn(rknn_path) == 0
    assert r.init_runtime(core_mask=core_mask) == 0
    return r


def bench_sequential(rknn_path, core_mask, repeat):
    r = new_instance(rknn_path, core_mask)
    inputs = make_inputs()
    r.inference(inputs=inputs, data_format='nchw')  # warmup
    t0 = time.perf_counter()
    for _ in range(repeat):
        r.inference(inputs=inputs, data_format='nchw')
    dt = (time.perf_counter() - t0) / repeat
    r.release()
    return dt


def bench_parallel(rknn_path, core_mask, repeat, n_workers):
    barrier = threading.Barrier(n_workers)
    results = [None] * n_workers

    def worker(i):
        r = new_instance(rknn_path, core_mask)
        inputs = make_inputs()
        r.inference(inputs=inputs, data_format='nchw')
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(repeat):
            r.inference(inputs=inputs, data_format='nchw')
        results[i] = time.perf_counter() - t0
        r.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    total_inferences = repeat * n_workers
    return wall, total_inferences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rknn', type=str, default='./sutrack_t224_rk3576.rknn')
    parser.add_argument('--repeat', type=int, default=30)
    args = parser.parse_args()

    from rknnlite.api import RKNNLite

    print('===== A. 单实例 core_mask=AUTO =====')
    dt_a = bench_sequential(args.rknn, RKNNLite.NPU_CORE_AUTO, args.repeat)
    print(f'  {dt_a * 1000:.1f} ms/帧, {1.0 / dt_a:.2f} FPS')

    print('===== B. 单实例 core_mask=NPU_CORE_0_1 =====')
    dt_b = bench_sequential(args.rknn, RKNNLite.NPU_CORE_0_1, args.repeat)
    print(f'  {dt_b * 1000:.1f} ms/帧, {1.0 / dt_b:.2f} FPS (相对 A 加速 {dt_a / dt_b:.2f}x)')

    print('===== C. 双实例双线程 core_mask=NPU_CORE_0_1 =====')
    wall, total = bench_parallel(args.rknn, RKNNLite.NPU_CORE_0_1, args.repeat, 2)
    fps_c = total / wall
    print(f'  聚合吞吐 {fps_c:.2f} FPS (相对 A 吞吐加速 {fps_c * dt_a:.2f}x)')


if __name__ == '__main__':
    main()
