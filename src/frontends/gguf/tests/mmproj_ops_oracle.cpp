// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Offline ggml oracle for the multimodal op tests. Link against the pinned ggml CPU libraries.
// Usage: mmproj_ops_oracle <output directory>; writes one raw F32 file per expectation.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

// Window partition/unpartition and SAM relative positions.
int geometry(const std::filesystem::path& directory) {
    auto* ctx = ggml_init({64 * 1024 * 1024, nullptr, false});
    auto* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 5, 3, 1);
    for (int i = 0; i < 45; ++i)
        static_cast<float*>(input->data)[i] = std::sin(float(i) * .13f);
    auto* windows = ggml_win_part(ctx, input, 2);
    auto* restored = ggml_win_unpart(ctx, windows, 5, 3, 2);
    // SAM decomposed relative positions: resize a 3-entry [L, C] table to 2 * 3 - 1 rows, gather q - k + 2.
    auto* table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    for (int i = 0; i < 12; ++i)
        static_cast<float*>(table->data)[i] = std::sin(float(i) * .37f);
    auto* indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 9);
    for (int q = 0; q < 3; ++q)
        for (int k = 0; k < 3; ++k)
            static_cast<int32_t*>(indices->data)[q * 3 + k] = q - k + 2;
    auto* lengths = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, table)), 3, 1, 4);
    auto* resized = ggml_cont(
        ctx,
        ggml_transpose(
            ctx,
            ggml_reshape_2d(ctx, ggml_interpolate(ctx, lengths, 5, 1, 4, 1, GGML_SCALE_MODE_BILINEAR), 5, 4)));
    auto* relative = ggml_get_rows(ctx, resized, indices);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, restored);
    ggml_build_forward_expand(graph, relative);
    if (ggml_graph_compute_with_ctx(ctx, graph, 2) != GGML_STATUS_SUCCESS)
        return 3;
    const auto save = [&](const char* name, ggml_tensor* t) {
        std::ofstream out(directory / name, std::ios::binary);
        out.write(static_cast<char*>(t->data), ggml_nbytes(t));
    };
    save("window_input.bin", input);
    save("windows.bin", windows);
    save("restored.bin", restored);
    save("relative_table.bin", table);
    save("relative.bin", relative);
    ggml_free(ctx);
    return 0;
}

// One op per mode: vision/imrope RoPE, antialiased bilinear resize variants, or 2D im2col.
int single_op(const std::string& mode, const std::filesystem::path& output) {
    const bool vision = mode == "vision";
    const bool resize = mode.find("resize") == 0;
    const bool downsample = mode.find("resize_down") == 0;
    const bool corners = mode.find("corners") != std::string::npos;
    const bool im2col = mode.find("im2col") == 0;
    const int image_width = mode == "im2col7" ? 7 : 9;
    const int image_height = image_width == 7 ? 5 : 4;
    auto* ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto* x = im2col   ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, image_width, image_height, 2, 1)
              : resize ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, downsample ? 12 : 3, downsample ? 8 : 2, 2, 1)
                       : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 3, 2, 1);
    auto* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8);
    int vision_sections[4] = {16, 16, 16, 16};
    int text_sections[4] = {8, 7, 9, 8};
    auto* y = im2col   ? ggml_im2col(ctx,
                                   ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 2, 1),
                                   x,
                                   2,
                                   1,
                                   1,
                                   0,
                                   1,
                                   1,
                                   true,
                                   GGML_TYPE_F32)
              : resize ? ggml_interpolate(ctx,
                                          x,
                                          5,
                                          4,
                                          2,
                                          1,
                                          GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS |
                                              (corners ? GGML_SCALE_FLAG_ALIGN_CORNERS : 0))
                       : ggml_rope_multi(ctx,
                                         x,
                                         positions,
                                         nullptr,
                                         vision ? 32 : 64,
                                         vision ? vision_sections : text_sections,
                                         vision ? GGML_ROPE_TYPE_VISION : GGML_ROPE_TYPE_IMROPE,
                                         32768,
                                         10000.f,
                                         1.f,
                                         0.f,
                                         1.f,
                                         32.f,
                                         1.f);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto backend = ggml_backend_cpu_init();
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> data(ggml_nelements(x));
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = std::sin(float(i) * 0.13f);
    const int32_t pos[8] = {3, 7, 11, 2, 5, 13, 17, 19};
    ggml_backend_tensor_set(x, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(positions, pos, 0, sizeof(pos));
    const auto status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        data.resize(ggml_nelements(y));
        ggml_backend_tensor_get(y, data.data(), 0, data.size() * sizeof(float));
        std::ofstream out(output, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    int status = geometry(directory);
    for (const char* mode :
         {"vision", "imrope", "resize", "resize_corners", "resize_down", "resize_down_corners", "im2col7", "im2col9"})
        status = status ? status : single_op(mode, directory / (std::string(mode) + ".bin"));
    return status;
}
