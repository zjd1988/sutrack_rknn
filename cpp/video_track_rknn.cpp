/*
 * SUTrack RKNN 开发板视频跟踪测速 (C++ 版)
 *
 * 功能与 python/video_track_rknn.py 完全一致:
 *   - 读取视频, 首帧给定初始框, 逐帧跟踪
 *   - 统计 解码/预处理/NPU推理/后处理 分项耗时与端到端帧率
 *   - 支持 --core-mask 设置 NPU 核掩码 (0=AUTO, 1=CORE0, 2=CORE1, 3=CORE_0_1)
 *   - --save 保存标注结果视频
 *
 * 编译 (在开发板上):
 *   g++ -O2 -std=c++14 video_track_rknn.cpp -o video_track_rknn \
 *       -I../runtime $(pkg-config --cflags --libs opencv4) -lrknnrt
 *
 * 运行:
 *   ./video_track_rknn --video vtest.avi --model sutrack_t224_rk3576.rknn \
 *       --bbox 496,156,38,76 --save track_out_cpp.mp4
 */
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"

static const float MEAN[6] = {0.485f, 0.456f, 0.406f, 0.485f, 0.456f, 0.406f};
static const float STDV[6] = {0.229f, 0.224f, 0.225f, 0.229f, 0.224f, 0.225f};

struct BBox { float x, y, w, h; };

// 与 python 版 sample_target 一致: 以目标中心裁剪正方形区域, 边界补 0, resize 到 output_sz
static cv::Mat sample_target(const cv::Mat& im, const BBox& bb, float factor, int output_sz,
                             float& resize_factor) {
    float crop_sz_f = std::ceil(std::sqrt(bb.w * bb.h) * factor);
    int crop_sz = (int)crop_sz_f;
    if (crop_sz < 1) { fprintf(stderr, "Too small bounding box.\n"); exit(1); }
    int x1 = (int)std::round(bb.x + 0.5f * bb.w - crop_sz * 0.5f);
    int y1 = (int)std::round(bb.y + 0.5f * bb.h - crop_sz * 0.5f);
    int x2 = x1 + crop_sz, y2 = y1 + crop_sz;
    int x1_pad = std::max(0, -x1), x2_pad = std::max(x2 - im.cols + 1, 0);
    int y1_pad = std::max(0, -y1), y2_pad = std::max(y2 - im.rows + 1, 0);

    cv::Rect roi(x1 + x1_pad, y1 + y1_pad,
                 std::min(crop_sz - x1_pad - x2_pad, im.cols - x1 - x1_pad),
                 std::min(crop_sz - y1_pad - y2_pad, im.rows - y1 - y1_pad));
    cv::Mat crop = im(roi).clone();
    cv::Mat padded;
    cv::copyMakeBorder(crop, padded, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT);
    resize_factor = (float)output_sz / crop_sz;
    cv::Mat out;
    cv::resize(padded, out, cv::Size(output_sz, output_sz));
    return out;
}

// 与 python 版 process 一致: HWC uint8 -> CHW float32, 6 通道 (两次拼接), 归一化
static void process(const cv::Mat& img, std::vector<float>& out /* (6,H,W) */) {
    cv::Mat f;
    img.convertTo(f, CV_32FC3, 1.0 / 255.0);
    int H = img.rows, W = img.cols;
    out.resize(6 * H * W);
    for (int c = 0; c < 6; ++c) {
        int sc = c % 3;
        for (int y = 0; y < H; ++y) {
            const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
            float* dst = out.data() + (size_t)c * H * W + (size_t)y * W;
            for (int x = 0; x < W; ++x)
                dst[x] = (row[x][sc] - MEAN[c]) / STDV[c];
        }
    }
}

// 与 python 版 transform_image_to_crop 一致, 输出归一化 [0,1] 的 cx,cy,w,h 形式 anno
static void transform_image_to_crop(const BBox& box_in, const BBox& box_extract,
                                    float resize_factor, int crop_sz, float anno[4]) {
    float ecx = box_extract.x + 0.5f * box_extract.w, ecy = box_extract.y + 0.5f * box_extract.h;
    float icx = box_in.x + 0.5f * box_in.w, icy = box_in.y + 0.5f * box_in.h;
    float ocx = (crop_sz - 1) / 2.0f + (icx - ecx) * resize_factor;
    float ocy = (crop_sz - 1) / 2.0f + (icy - ecy) * resize_factor;
    float ow = box_in.w * resize_factor, oh = box_in.h * resize_factor;
    anno[0] = (ocx - 0.5f * ow) / (crop_sz - 1);
    anno[1] = (ocy - 0.5f * oh) / (crop_sz - 1);
    anno[2] = ow / (crop_sz - 1);
    anno[3] = oh / (crop_sz - 1);
}

static BBox clip_box(const BBox& b, int H, int W, int margin) {
    float x1 = b.x, y1 = b.y, x2 = b.x + b.w, y2 = b.y + b.h;
    x1 = std::min(std::max(0.f, x1), (float)W - margin);
    x2 = std::min(std::max((float)margin, x2), (float)W);
    y1 = std::min(std::max(0.f, y1), (float)H - margin);
    y2 = std::min(std::max((float)margin, y2), (float)H);
    return {x1, y1, std::max((float)margin, x2 - x1), std::max((float)margin, y2 - y1)};
}

// feat_sz x feat_sz 二维 Hann 窗
static std::vector<float> hann2d(int feat_sz) {
    std::vector<float> win(feat_sz * feat_sz);
    for (int i = 0; i < feat_sz; ++i)
        for (int j = 0; j < feat_sz; ++j) {
            float wy = 0.5f * (1 - std::cos(2 * M_PI / (feat_sz + 1) * (i + 1)));
            float wx = 0.5f * (1 - std::cos(2 * M_PI / (feat_sz + 1) * (j + 1)));
            win[i * feat_sz + j] = wy * wx;
        }
    return win;
}

// 与 python 版 cal_bbox 一致: 输出归一化 cx,cy,w,h
static void cal_bbox(const float* score, const float* size_map, const float* offset_map,
                     int feat_sz, BBox& out, float& max_score) {
    int idx = 0;
    max_score = score[0];
    int n = feat_sz * feat_sz;
    for (int i = 1; i < n; ++i)
        if (score[i] > max_score) { max_score = score[i]; idx = i; }
    int idx_y = idx / feat_sz, idx_x = idx % feat_sz;
    out.x = (idx_x + offset_map[idx]) / feat_sz;          // offset_map channel0
    out.y = (idx_y + offset_map[n + idx]) / feat_sz;      // offset_map channel1
    out.w = size_map[idx];                                // size_map channel0
    out.h = size_map[n + idx];                            // size_map channel1
}

static BBox map_box_back(const BBox& pred, const BBox& state, float resize_factor, int search_size) {
    float cx_prev = state.x + 0.5f * state.w, cy_prev = state.y + 0.5f * state.h;
    float half_side = 0.5f * search_size / resize_factor;
    float cx_real = pred.x + (cx_prev - half_side);
    float cy_real = pred.y + (cy_prev - half_side);
    return {cx_real - 0.5f * pred.w, cy_real - 0.5f * pred.h, pred.w, pred.h};
}

static std::vector<char> read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "read error\n"); exit(1); }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    std::string video_path, model_path, save_path;
    BBox init_bb{0, 0, 0, 0};
    bool has_bbox = false;
    int core_mask = RKNN_NPU_CORE_AUTO;
    int max_frames = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--video") video_path = next();
        else if (a == "--model") model_path = next();
        else if (a == "--save") save_path = next();
        else if (a == "--core-mask") core_mask = std::stoi(next());
        else if (a == "--max-frames") max_frames = std::stoi(next());
        else if (a == "--bbox") {
            float v[4];
            if (sscanf(next().c_str(), "%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3]) == 4) {
                init_bb = {v[0], v[1], v[2], v[3]};
                has_bbox = true;
            }
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }
    if (video_path.empty() || model_path.empty() || !has_bbox) {
        fprintf(stderr, "usage: %s --video X --model X --bbox x,y,w,h [--save out.mp4] [--core-mask N] [--max-frames N]\n", argv[0]);
        return 1;
    }

    int search_size = model_path.find("384") != std::string::npos ? 384 : 224;
    int template_size = model_path.find("384") != std::string::npos ? 192 : 112;
    int feat_sz = search_size / 16;

    // 加载 RKNN 模型
    std::vector<char> model = read_file(model_path.c_str());
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model.data(), model.size(), 0, nullptr);
    if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_init failed: %d\n", ret); return 1; }
    rknn_set_core_mask(ctx, (rknn_core_mask)core_mask);

    rknn_sdk_version ver;
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("librknnrt: %s, driver: %s\n", ver.api_version, ver.drv_version);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) { fprintf(stderr, "cannot open video\n"); return 1; }
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv::Mat frame, first_frame;
    if (!cap.read(first_frame)) { fprintf(stderr, "cannot read video\n"); return 1; }
    printf("video: %d frames, %.1f fps, %dx%d\n", total_frames, video_fps, first_frame.cols, first_frame.rows);

    cv::VideoWriter writer;
    if (!save_path.empty())
        writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    video_fps > 0 ? video_fps : 25, first_frame.size());

    const float template_factor = 2.0f, search_factor = 4.0f;
    const int num_templates = 2, update_intervals = 25;
    const float update_threshold = 0.7f;

    std::vector<float> hann = hann2d(feat_sz);

    cv::Mat first_rgb;
    cv::cvtColor(first_frame, first_rgb, cv::COLOR_BGR2RGB);
    BBox state = init_bb;

    // 模板: 2 份, 每份 (6, template_size, template_size)
    std::vector<std::vector<float>> templates(num_templates);
    std::vector<std::array<float, 4>> template_annos(num_templates);
    float rf;
    cv::Mat z_patch = sample_target(first_rgb, state, template_factor, template_size, rf);
    process(z_patch, templates[0]);
    transform_image_to_crop(state, state, rf, template_size, template_annos[0].data());
    templates[1] = templates[0];
    template_annos[1] = template_annos[0];

    if (writer.isOpened()) {
        cv::rectangle(first_frame, cv::Rect((int)state.x, (int)state.y, (int)state.w, (int)state.h),
                      cv::Scalar(0, 255, 0), 3);
        writer.write(first_frame);
    }

    // 推理输入缓冲 (复用)
    std::vector<float> template_input((size_t)2 * 6 * template_size * template_size);
    std::vector<float> search_input((size_t)6 * search_size * search_size);
    std::vector<float> anno_input(8);

    rknn_input inputs[3];
    memset(inputs, 0, sizeof(inputs));
    for (int i = 0; i < 3; ++i) {
        inputs[i].index = i;
        inputs[i].type = RKNN_TENSOR_FLOAT32;
        inputs[i].fmt = RKNN_TENSOR_NCHW;
        inputs[i].pass_through = 0;
    }
    inputs[0].size = template_input.size() * sizeof(float);
    inputs[0].buf = template_input.data();
    inputs[1].size = search_input.size() * sizeof(float);
    inputs[1].buf = search_input.data();
    inputs[2].size = anno_input.size() * sizeof(float);
    inputs[2].buf = anno_input.data();

    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; ++i) outputs[i].want_float = 1;

    double t_decode = 0, t_pre = 0, t_infer = 0, t_post = 0;
    int frame_id = 0;
    auto t_start = std::chrono::steady_clock::now();

    while (true) {
        auto t0 = std::chrono::steady_clock::now();
        if (!cap.read(frame)) break;
        auto t1 = std::chrono::steady_clock::now();

        frame_id++;
        int H = frame.rows, W = frame.cols;
        cv::Mat frame_rgb;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        cv::Mat x_patch = sample_target(frame_rgb, state, search_factor, search_size, rf);
        process(x_patch, search_input);

        // 拼接双模板与 anno
        size_t tsz = templates[0].size();
        memcpy(template_input.data(), templates[0].data(), tsz * sizeof(float));
        memcpy(template_input.data() + tsz, templates[1].data(), tsz * sizeof(float));
        memcpy(anno_input.data(), template_annos[0].data(), 4 * sizeof(float));
        memcpy(anno_input.data() + 4, template_annos[1].data(), 4 * sizeof(float));
        auto t2 = std::chrono::steady_clock::now();

        ret = rknn_inputs_set(ctx, 3, inputs);
        if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_inputs_set failed: %d\n", ret); break; }
        ret = rknn_run(ctx, nullptr);
        if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_run failed: %d\n", ret); break; }
        ret = rknn_outputs_get(ctx, 3, outputs, nullptr);
        if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_outputs_get failed: %d\n", ret); break; }
        auto t3 = std::chrono::steady_clock::now();

        // outputs: score_map [1,1,fs,fs], size_map [1,2,fs,fs], offset_map [1,2,fs,fs]
        const float* score_map = (const float*)outputs[0].buf;
        const float* size_map = (const float*)outputs[1].buf;
        const float* offset_map = (const float*)outputs[2].buf;

        // score * hann 窗
        int n = feat_sz * feat_sz;
        std::vector<float> response(n);
        for (int i = 0; i < n; ++i) response[i] = score_map[i] * hann[i];

        BBox pred;
        float conf;
        cal_bbox(response.data(), size_map, offset_map, feat_sz, pred, conf);
        pred.x *= search_size / rf; pred.y *= search_size / rf;
        pred.w *= search_size / rf; pred.h *= search_size / rf;
        state = clip_box(map_box_back(pred, state, rf, search_size), H, W, 10);
        rknn_outputs_release(ctx, 3, outputs);

        if (frame_id % update_intervals == 0 && conf > update_threshold) {
            cv::Mat zp = sample_target(frame_rgb, state, template_factor, template_size, rf);
            process(zp, templates[1]);
            transform_image_to_crop(state, state, rf, template_size, template_annos[1].data());
        }

        if (writer.isOpened()) {
            cv::Rect box((int)state.x, (int)state.y, (int)state.w, (int)state.h);
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 3);
            char txt[32];
            snprintf(txt, sizeof(txt), "%.2f", conf);
            cv::putText(frame, txt, cv::Point(box.x, std::max(20, box.y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            writer.write(frame);
        }
        auto t4 = std::chrono::steady_clock::now();

        t_decode += std::chrono::duration<double>(t1 - t0).count();
        t_pre += std::chrono::duration<double>(t2 - t1).count();
        t_infer += std::chrono::duration<double>(t3 - t2).count();
        t_post += std::chrono::duration<double>(t4 - t3).count();

        if (frame_id % 100 == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            printf("  frame %d/%d: %.1f FPS, conf=%.3f\n", frame_id, total_frames, frame_id / el, conf);
        }
        if (max_frames > 0 && frame_id >= max_frames) break;
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    cap.release();
    if (writer.isOpened()) writer.release();
    rknn_destroy(ctx);

    printf("\n========== 性能统计 (C++) ==========\n");
    printf("处理帧数: %d, 总耗时: %.2f s\n", frame_id, elapsed);
    printf("端到端平均: %.2f FPS (%.1f ms/帧)\n", frame_id / elapsed, elapsed / std::max(frame_id, 1) * 1000);
    printf("  解码读取: %6.1f ms/帧\n", t_decode / std::max(frame_id, 1) * 1000);
    printf("  预处理  : %6.1f ms/帧\n", t_pre / std::max(frame_id, 1) * 1000);
    printf("  NPU 推理: %6.1f ms/帧\n", t_infer / std::max(frame_id, 1) * 1000);
    printf("  后处理  : %6.1f ms/帧\n", t_post / std::max(frame_id, 1) * 1000);
    printf("源视频帧率: %.1f fps -> 实时倍率: %.2fx (%s)\n",
           video_fps, frame_id / elapsed / std::max(video_fps, 1e-6),
           frame_id / elapsed >= video_fps ? "可达到实时" : "无法实时");
    if (!save_path.empty()) printf("标注结果已保存: %s\n", save_path.c_str());
    return 0;
}
