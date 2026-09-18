// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Offline oracle: link against the pinned ggml CPU libraries.
// Usage: multimodal_rope_oracle vision|imrope|resize|im2col7|im2col9 output.f32
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    const bool vision = std::string(argv[1]) == "vision";
    const bool resize = std::string(argv[1]) == "resize";
    const bool im2col = std::string(argv[1]).find("im2col") == 0;
    const int image_width = std::string(argv[1]) == "im2col7" ? 7 : 9;
    const int image_height = image_width == 7 ? 5 : 4;
    auto* ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    auto* x = im2col   ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, image_width, image_height, 2, 1)
              : resize ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 2, 1)
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
              : resize ? ggml_interpolate(ctx, x, 5, 4, 2, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS)
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
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS ? 0 : 1;
}
